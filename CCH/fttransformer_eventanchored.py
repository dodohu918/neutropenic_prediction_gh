"""
fttransformer_eventanchored.py — FT-Transformer vs TabNet vs MLP, event-anchored tabular

FT-Transformer: Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data",
NeurIPS 2021 (Yandex Research). 當前 tabular DL 的 consensus best baseline。

架構:
  Feature Tokenizer: 每個 numerical feature → per-feature linear projection → d_token embedding
  [CLS] token 前置，N 個 Pre-LN Transformer block，CLS → binary classification head。

用法 (從 CCH/ 執行):
  python fttransformer_eventanchored.py --seed 42
  python fttransformer_eventanchored.py --seed 42 --n-boot 500

與 mlp_vs_lgb_eventanchored.py 分開執行 (Apple Silicon PyTorch/LightGBM OpenMP 衝突)。
"""
import os
import sys
import math
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
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

print(f"FT-Transformer | event-anchored h{HORIZON} block primary seed{SEED} max_obs_days={MAX_OBS_DAYS} | device={DEVICE}")


# ══════════════════════════════════════════════════════════════════════════════
#  Model definition (faithfully follows Gorishniy et al. NeurIPS 2021)
# ══════════════════════════════════════════════════════════════════════════════

class FeatureTokenizer(nn.Module):
    """Per-feature linear projection + prepended [CLS] token."""
    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias   = nn.Parameter(torch.zeros(n_features, d_token))
        self.cls    = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, n_features]
        tokens = x.unsqueeze(-1) * self.weight + self.bias   # [B, n_feat, d_token]
        cls = self.cls.expand(x.size(0), -1, -1)             # [B, 1, d_token]
        return torch.cat([cls, tokens], dim=1)                # [B, n_feat+1, d_token]


class FTTransformerBlock(nn.Module):
    """Pre-LN Transformer block (attention + FFN with GELU)."""
    def __init__(self, d_token: int, n_heads: int, ffn_d_hidden: int,
                 attn_dropout: float, ffn_dropout: float, residual_dropout: float):
        super().__init__()
        self.norm1  = nn.LayerNorm(d_token)
        self.attn   = nn.MultiheadAttention(d_token, n_heads,
                                            dropout=attn_dropout, batch_first=True)
        self.drop1  = nn.Dropout(residual_dropout)
        self.norm2  = nn.LayerNorm(d_token)
        self.ffn    = nn.Sequential(
            nn.Linear(d_token, ffn_d_hidden),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(ffn_d_hidden, d_token),
            nn.Dropout(ffn_dropout),
        )
        self.drop2  = nn.Dropout(residual_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN attention
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + self.drop1(h)
        # Pre-LN FFN
        h = self.norm2(x)
        h = self.ffn(h)
        x = x + self.drop2(h)
        return x


class FTTransformer(nn.Module):
    """
    Feature Tokenizer + Transformer classifier.
    Default hyperparameters from the paper (medium-dataset config).
    """
    def __init__(self, n_features: int, d_token: int = 192, n_blocks: int = 3,
                 n_heads: int = 8, ffn_d_hidden: int = 256,
                 attn_dropout: float = 0.2, ffn_dropout: float = 0.1,
                 residual_dropout: float = 0.0):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_token)
        self.blocks    = nn.ModuleList([
            FTTransformerBlock(d_token, n_heads, ffn_d_hidden,
                               attn_dropout, ffn_dropout, residual_dropout)
            for _ in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)          # [B, n_feat+1, d_token]
        for block in self.blocks:
            tokens = block(tokens)
        cls = self.norm(tokens[:, 0])       # [B, d_token] — CLS token
        return self.head(cls).squeeze(-1)   # [B] logits


# ══════════════════════════════════════════════════════════════════════════════
#  Training helper
# ══════════════════════════════════════════════════════════════════════════════

def train_ftt(Xtr, ytr, Xva, yva, n_features, pos_weight_val,
              max_epochs=200, patience=20, batch_size=256, lr=1e-4, wd=1e-5):
    """Train FT-Transformer; return best model + best val AUC."""
    model = FTTransformer(n_features).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    pw    = torch.tensor([pos_weight_val], device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=DEVICE)

    loader = DataLoader(TensorDataset(Xtr_t, ytr_t),
                        batch_size=batch_size, shuffle=True)

    best_auc, best_epoch, best_state = 0.0, 0, None
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(Xva_t).cpu().numpy()
        proba = torch.sigmoid(torch.tensor(logits)).numpy()
        try:
            auc = roc_auc_score(yva, proba)
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
    return model, best_auc, best_epoch


def predict_ftt(model, Xte):
    model.eval()
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        logits = model(Xte_t).cpu()
    return torch.sigmoid(logits).numpy()


# ══════════════════════════════════════════════════════════════════════════════
#  Data
# ══════════════════════════════════════════════════════════════════════════════

df      = pc.load_lab_df(exclude_heme_mode='leukemia_only_adj')
chemo_df = pc.load_chemo_filtered()
cm_info  = eac.build_cm_info(pd.read_csv(pc.CANCER_MAP_PATH))

pats = df['Newid'].unique()
tr_idx, te_idx = next(
    GroupShuffleSplit(test_size=0.2, random_state=SEED).split(pats, groups=pats))
train_pat, test_pat = set(pats[tr_idx]), set(pats[te_idx])
markers  = sorted(df[df['Newid'].isin(train_pat)]['NAME'].dropna().unique().tolist())
top_drugs = pc.select_top_k_drugs(chemo_df, train_pat, pc.TOP_K_DRUGS)

data = eac.build_anchor_dataset(df, chemo_df, cm_info, top_drugs, markers,
                                horizon=HORIZON, gap=10, regimen_days=0,
                                max_obs_days=MAX_OBS_DAYS,
                                regimen_mode='block')
fc   = [c for c in data.columns if c not in ('Newid', 'min_win_anc')]
tr   = data['Newid'].isin(train_pat)
te   = data['Newid'].isin(test_pat)
Xtr_raw, Xte_raw = data.loc[tr, fc].values, data.loc[te, fc].values
anc_tr = data.loc[tr, 'min_win_anc'].values
anc_te = data.loc[te, 'min_win_anc'].values

sc = StandardScaler().fit(Xtr_raw)
Xtr = sc.transform(Xtr_raw).astype(np.float32)
Xte = sc.transform(Xte_raw).astype(np.float32)
N_FEAT = Xtr.shape[1]

print(f"Train: {len(Xtr)} | Test: {len(Xte)} | Features: {N_FEAT}")


# ══════════════════════════════════════════════════════════════════════════════
#  Bootstrap CI helper
# ══════════════════════════════════════════════════════════════════════════════

def boot_ci(y_true, y_score, metric_fn):
    if N_BOOT < 1:
        return None, None
    rng = np.random.RandomState(SEED)
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

CUT = int(0.85 * len(Xtr))          # same 85/15 val split used by LGB/XGB

for thr in THRESHOLDS:
    ytr = (anc_tr < thr).astype(np.float32)
    yte = (anc_te < thr).astype(int)
    n_pos = int(ytr.sum()); n_neg = len(ytr) - n_pos
    if n_pos < 10:
        continue

    pos_weight = float(n_neg) / n_pos   # mirrors scale_pos_weight in LGB/XGB

    print(f"\n=== {LABELS[thr]} | test n_pos={int(yte.sum())} / n={len(yte)} "
          f"(prev={yte.mean():.1%}) ===")
    print(f"  {'Model':<16}  {'AUROC (95% CI)':<24}  {'PR-AUC (95% CI)':<24}")
    print(f"  {'-'*16}  {'-'*24}  {'-'*24}")

    # ── FT-Transformer ────────────────────────────────────────────────────────
    model, val_auc, best_ep = train_ftt(
        Xtr[:CUT], ytr[:CUT], Xtr[CUT:], ytr[CUT:].astype(int),
        N_FEAT, pos_weight,
    )
    p_ftt = predict_ftt(model, Xte)
    print(f"  {'FT-Transformer':<16}  {fmt(yte, p_ftt, roc_auc_score):<24}  "
          f"{fmt(yte, p_ftt, average_precision_score):<24}"
          f"  [best_epoch={best_ep}, val_auc={val_auc:.4f}]")

    # ── MLP (sklearn, reference) ──────────────────────────────────────────────
    Xtr_s = sc.transform(Xtr_raw)
    Xte_s = sc.transform(Xte_raw)
    pos_idx = np.where(ytr.astype(int) == 1)[0]
    reps = max(int(round(n_neg / n_pos)) - 1, 0)
    idx_os = (np.concatenate([np.arange(len(ytr))] + [pos_idx] * reps)
              if reps > 0 else np.arange(len(ytr)))
    rng = np.random.RandomState(SEED); rng.shuffle(idx_os)
    m_mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', alpha=1e-3,
                          early_stopping=True, n_iter_no_change=15, max_iter=300,
                          random_state=SEED)
    m_mlp.fit(Xtr_s[idx_os], ytr.astype(int)[idx_os])
    p_mlp = m_mlp.predict_proba(Xte_s)[:, 1]
    print(f"  {'MLP':<16}  {fmt(yte, p_mlp, roc_auc_score):<24}  "
          f"{fmt(yte, p_mlp, average_precision_score):<24}")

print(f"""
FT-Transformer config (paper defaults):
  d_token=192, n_blocks=3, n_heads=8, ffn_d_hidden=256
  attn_dropout=0.2, ffn_dropout=0.1, residual_dropout=0.0
  lr=1e-4, weight_decay=1e-5, batch_size=256, patience=20
  pos_weight = n_neg/n_pos  (mirrors LGB/XGB scale_pos_weight)
  device={DEVICE}

Reference: Gorishniy et al., NeurIPS 2021
  "Revisiting Deep Learning Models for Tabular Data"
  https://arxiv.org/abs/2106.11959
""")
