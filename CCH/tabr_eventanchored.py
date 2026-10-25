"""
tabr_eventanchored.py — TabR, event-anchored tabular

TabR: Gorishniy et al., "TabR: Unlocking the Power of Retrieval-Augmented
Tabular Deep Learning", ICLR 2024 (Yandex Research).

核心創新: 在推論時從訓練集「取回 (retrieve)」k 個最近鄰，
把它們的 embedding + label 透過 attention 整合進預測，
結合了 parametric DL 和 non-parametric k-NN 的優點。

架構:
  1. Feature Encoder: MLP → d_embed 維 embedding
  2. 每個 epoch 起始: 對全訓練集 encode（stop gradient）→ embedding cache
  3. 每個 batch: cosine similarity 找每個 query 的 k 個最近鄰
  4. Attention: query 對 [cand_emb || label_emb] 加權求和
  5. Prediction head: Linear([query_emb || context] → 1)

用法 (從 CCH/ 執行):
  python tabr_eventanchored.py --seed 42
  python tabr_eventanchored.py --seed 42 --n-boot 500
"""
import os
import sys
import math
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
import pipeline_common as pc
import eventanchor_common as eac

# ── device ─────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--seed',         type=int, default=42)
parser.add_argument('--horizon',      type=int, default=28)
parser.add_argument('--n-boot',       type=int, default=500)
parser.add_argument('--max-obs-days', type=int, default=28,
                    help='Only use lab features within this many days before T')
args = parser.parse_args()
SEED, HORIZON, N_BOOT, MAX_OBS_DAYS = args.seed, args.horizon, args.n_boot, args.max_obs_days

torch.manual_seed(SEED)
np.random.seed(SEED)

THRESHOLDS = [1.5, 1.0, 0.5]
LABELS = {1.5: '<1500 (mild)', 1.0: '<1000 (moderate)', 0.5: '<500 (severe/FN)'}

print(f"TabR | event-anchored h{HORIZON} block primary seed{SEED} max_obs_days={MAX_OBS_DAYS} | device={DEVICE}")


# ══════════════════════════════════════════════════════════════════════════════
#  Model (faithfully follows Gorishniy et al., ICLR 2024)
# ══════════════════════════════════════════════════════════════════════════════

class TabR(nn.Module):
    """
    Retrieval-Augmented Tabular Deep Learning.

    Hyperparameters (paper defaults for medium datasets):
      d_embed  = 128   feature encoder output dim
      d_label  = 8     label embedding dim (for candidate labels)
      K        = 96    candidates retrieved per query
    """
    def __init__(self, n_features: int, d_embed: int = 128, d_label: int = 8):
        super().__init__()
        self.d_embed = d_embed
        self.K = 96

        # Feature encoder: n_features → d_embed
        self.encoder = nn.Sequential(
            nn.Linear(n_features, d_embed),
            nn.ReLU(),
            nn.Linear(d_embed, d_embed),
        )

        # Label embedding: 0 / 1 → d_label
        self.label_emb = nn.Embedding(2, d_label)

        # Prediction head: [query_emb || context] → logit
        # context dim = d_embed + d_label
        self.head = nn.Sequential(
            nn.LayerNorm(d_embed + (d_embed + d_label)),
            nn.Linear(d_embed + (d_embed + d_label), d_embed),
            nn.ReLU(),
            nn.Linear(d_embed, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode features → normalised embedding."""
        e = self.encoder(x)
        return e / (e.norm(dim=-1, keepdim=True) + 1e-8)

    def forward(self,
                x:           torch.Tensor,   # [B, n_features]
                cand_emb:    torch.Tensor,   # [N_cand, d_embed] — frozen
                cand_labels: torch.Tensor,   # [N_cand] int64 — training labels
                ) -> torch.Tensor:           # [B] logits

        q = self.encode(x)                               # [B, d_embed]

        # ── Per-sample retrieval ──────────────────────────────────────────────
        # Cosine similarity (embeddings already L2-normalised)
        sim = q @ cand_emb.T                             # [B, N_cand]
        K = min(self.K, cand_emb.size(0))
        _, top_idx = sim.topk(K, dim=1)                  # [B, K]

        # Gather top-K candidates
        c_emb = cand_emb[top_idx]                        # [B, K, d_embed]
        c_lbl = self.label_emb(cand_labels[top_idx])     # [B, K, d_label]
        c_kv  = torch.cat([c_emb, c_lbl], dim=-1)        # [B, K, d_embed+d_label]

        # ── Attention: query × candidates ────────────────────────────────────
        # scores: dot(q, c_emb) / sqrt(d) — shape [B, K]
        attn_scores = (q.unsqueeze(1) * c_emb).sum(dim=-1) / math.sqrt(self.d_embed)
        attn_weights = torch.softmax(attn_scores, dim=-1)   # [B, K]

        # Weighted sum of key-value pairs → context [B, d_embed+d_label]
        context = (attn_weights.unsqueeze(-1) * c_kv).sum(dim=1)

        # ── Prediction ───────────────────────────────────────────────────────
        combined = torch.cat([q, context], dim=-1)       # [B, d_embed + d_embed+d_label]
        return self.head(combined).squeeze(-1)            # [B]


# ══════════════════════════════════════════════════════════════════════════════
#  Training helper
# ══════════════════════════════════════════════════════════════════════════════

def train_tabr(Xtr, ytr, Xva, yva, n_features, pos_weight_val,
               max_epochs=200, patience=20, batch_size=256, lr=1e-4, wd=0.0):

    model   = TabR(n_features).to(DEVICE)
    opt     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    pw      = torch.tensor([pos_weight_val], device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    Xtr_t  = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t  = torch.tensor(ytr, dtype=torch.float32)
    ytr_i  = torch.tensor(ytr, dtype=torch.long, device=DEVICE)   # for label embedding
    Xva_t  = torch.tensor(Xva, dtype=torch.float32, device=DEVICE)

    loader = DataLoader(TensorDataset(Xtr_t, ytr_t),
                        batch_size=batch_size, shuffle=True)

    best_auc, best_epoch, best_state, no_improve = 0.0, 0, None, 0

    for epoch in range(1, max_epochs + 1):

        # ── Re-encode full training set (stop gradient → embedding cache) ────
        model.eval()
        with torch.no_grad():
            chunks = []
            for i in range(0, len(Xtr_t), 512):
                chunks.append(model.encode(Xtr_t[i:i+512].to(DEVICE)))
            all_emb = torch.cat(chunks, dim=0)   # [N_train, d_embed] on DEVICE

        # ── Training pass ─────────────────────────────────────────────────────
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb, all_emb, ytr_i)
            loss_fn(logits, yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # ── Validation AUC ────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            logits_va = model(Xva_t, all_emb, ytr_i)
        proba_va = torch.sigmoid(logits_va).cpu().numpy()
        try:
            auc = roc_auc_score(yva, proba_va)
        except Exception:
            auc = 0.0

        if auc > best_auc:
            best_auc, best_epoch = auc, epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_auc, best_epoch, all_emb, ytr_i


def predict_tabr(model, Xte, train_emb, train_labels):
    model.eval()
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        logits = model(Xte_t, train_emb, train_labels)
    return torch.sigmoid(logits).cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
#  Data
# ══════════════════════════════════════════════════════════════════════════════

df       = pc.load_lab_df(exclude_heme_mode='leukemia_only_adj')
chemo_df = pc.load_chemo_filtered()
cm_info  = eac.build_cm_info(pd.read_csv(pc.CANCER_MAP_PATH))

pats = df['Newid'].unique()
tr_idx, te_idx = next(
    GroupShuffleSplit(test_size=0.2, random_state=SEED).split(pats, groups=pats))
train_pat, test_pat = set(pats[tr_idx]), set(pats[te_idx])
markers   = sorted(df[df['Newid'].isin(train_pat)]['NAME'].dropna().unique().tolist())
top_drugs = pc.select_top_k_drugs(chemo_df, train_pat, pc.TOP_K_DRUGS)

data = eac.build_anchor_dataset(df, chemo_df, cm_info, top_drugs, markers,
                                horizon=HORIZON, gap=10, regimen_days=0,
                                regimen_mode='block', max_obs_days=MAX_OBS_DAYS)
fc   = [c for c in data.columns if c not in ('Newid', 'min_win_anc')]
tr   = data['Newid'].isin(train_pat)
te   = data['Newid'].isin(test_pat)
Xtr_raw  = data.loc[tr, fc].values
Xte_raw  = data.loc[te, fc].values
anc_tr   = data.loc[tr, 'min_win_anc'].values
anc_te   = data.loc[te, 'min_win_anc'].values
pid_te   = data.loc[te, 'Newid'].values

sc  = StandardScaler().fit(Xtr_raw)
Xtr = sc.transform(Xtr_raw).astype(np.float32)
Xte = sc.transform(Xte_raw).astype(np.float32)

print(f"Train: {len(Xtr)} | Test: {len(Xte)} | Features: {Xtr.shape[1]}")


# ══════════════════════════════════════════════════════════════════════════════
#  Bootstrap CI
# ══════════════════════════════════════════════════════════════════════════════

def boot_ci(y_true, y_score, metric_fn):
    if N_BOOT < 1:
        return None, None
    rng  = np.random.RandomState(SEED)
    vals = []
    for _ in range(N_BOOT):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if y_true[idx].sum() < 2 or y_true[idx].sum() == len(idx):
            continue
        vals.append(metric_fn(y_true[idx], y_score[idx]))
    if len(vals) < 10:
        return None, None
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def fmt(y_true, y_score, metric_fn):
    pt = metric_fn(y_true, y_score)
    lo, hi = boot_ci(y_true, y_score, metric_fn)
    return f"{pt:.3f}" if lo is None else f"{pt:.3f} [{lo:.3f}-{hi:.3f}]"


# ══════════════════════════════════════════════════════════════════════════════
#  Main loop
# ══════════════════════════════════════════════════════════════════════════════

CUT = int(0.85 * len(Xtr))
_PRED_OUT = os.path.join(_HERE, f'predictions_tabr_h{HORIZON}_noleukadj_seed{SEED}.npz')
save_dict: dict = {}

for thr in THRESHOLDS:
    ytr = (anc_tr < thr).astype(np.float32)
    yte = (anc_te < thr).astype(int)
    n_pos = int(ytr.sum()); n_neg = len(ytr) - n_pos
    if n_pos < 10:
        continue

    pos_weight = float(n_neg) / n_pos

    print(f"\n=== {LABELS[thr]} | test n_pos={int(yte.sum())} / n={len(yte)} "
          f"(prev={yte.mean():.1%}) ===")
    print(f"  {'Model':<16}  {'AUROC (95% CI)':<24}  {'PR-AUC (95% CI)':<24}")
    print(f"  {'-'*16}  {'-'*24}  {'-'*24}")

    model, val_auc, best_ep, train_emb, train_lbl = train_tabr(
        Xtr[:CUT], ytr[:CUT].astype(int),
        Xtr[CUT:], ytr[CUT:].astype(int),
        Xtr.shape[1], pos_weight,
    )
    p_tabr = predict_tabr(model, Xte, train_emb, train_lbl)
    print(f"  {'TabR':<16}  {fmt(yte, p_tabr, roc_auc_score):<24}  "
          f"{fmt(yte, p_tabr, average_precision_score):<24}"
          f"  [best_epoch={best_ep}, val_auc={val_auc:.4f}]")
    save_dict[f'thr{thr}_pid']   = pid_te
    save_dict[f'thr{thr}_pred']  = p_tabr
    save_dict[f'thr{thr}_truth'] = anc_te   # raw ANC; binarize with < thr downstream

save_dict['_thresholds'] = np.array(THRESHOLDS)
save_dict['_horizon']    = np.array([HORIZON])
save_dict['_seed']       = np.array([SEED])
np.savez_compressed(_PRED_OUT, **save_dict)
print(f"\n✓ 預測存到 {_PRED_OUT}")

print(f"""
TabR config (paper defaults for medium datasets):
  d_embed=128, K=96 (retrieved candidates), d_label=8
  lr=1e-4, weight_decay=0, batch_size=256, patience=20
  pos_weight = n_neg/n_pos  (mirrors LGB/XGB scale_pos_weight)
  Training: re-encode full training set every epoch (stop gradient)
  device={DEVICE}

Reference: Gorishniy et al., ICLR 2024
  "TabR: Unlocking the Power of Retrieval-Augmented Tabular Deep Learning"
  https://arxiv.org/abs/2307.14338
""")
