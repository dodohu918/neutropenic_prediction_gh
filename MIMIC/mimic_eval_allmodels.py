"""
mimic_eval_allmodels.py — 台灣訓練 → MIMIC-IV 外部驗證，全 6 個模型

模型: LightGBM, XGBoost, LR, MLP, FT-Transformer, TabR
支援 --max-obs-days (將所有 lab feature 限制在 T 前 N 天內)

用法 (從 MIMIC/ 執行):
  python mimic_eval_allmodels.py --max-obs-days 14
  python mimic_eval_allmodels.py --max-obs-days 14 --horizons 28

注意:
  - 台灣資料訓練 → 直接套用到 MIMIC (frozen model, 不 retrain)
  - DL 模型 (FT-Transformer, TabR) 在 tree 模型後才 import torch
    (Apple Silicon PyTorch/LightGBM OpenMP 衝突 workaround)
  - MIMIC dose 特徵全為 0 (emar 無欄位)
  - TabR 在 MIMIC inference 時從台灣訓練集做 k-NN retrieval
"""
import os, sys, math, argparse, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_HERE = os.path.dirname(os.path.abspath(__file__))
_TW   = os.path.join(os.path.dirname(_HERE), 'CCH')
sys.path.insert(0, _HERE)
sys.path.insert(0, _TW)

import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import pipeline_common as pc
import eventanchor_common as eac
import mimic_gcsf as mg

# ── device (for DL models) ────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--horizons',     type=int, nargs='+', default=[28])
parser.add_argument('--max-obs-days', type=int, default=None,
                    help='Restrict all lab features to within N days before T')
args = parser.parse_args()
HORIZONS      = args.horizons
MAX_OBS_DAYS  = args.max_obs_days

GAP, REGIMEN_MODE, BLOCK_GAP, MAX_BLOCK = 10, 'block', 36.0, 7
HEME_MODE = 'leukemia_only_adj'
SEEDS      = [42, 43, 44]
THRESHOLDS = [1.5, 1.0, 0.5]
LABELS     = {1.5: '<1500 (mild)', 1.0: '<1000 (moderate)', 0.5: '<500 (severe)'}
MIMIC_DIR  = os.path.join(_HERE, 'mimic_data')
MODELS     = ['LightGBM', 'XGBoost', 'LR', 'MLP', 'FT-Transformer', 'TabR']

print(f"All-model MIMIC eval | horizons={HORIZONS} | max_obs_days={MAX_OBS_DAYS} | device={DEVICE}")

# ══════════════════════════════════════════════════════════════════════════════
#  FT-Transformer model definition (Gorishniy et al., NeurIPS 2021)
# ══════════════════════════════════════════════════════════════════════════════

class _FeatureTokenizer(nn.Module):
    def __init__(self, n_features, d_token):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias   = nn.Parameter(torch.zeros(n_features, d_token))
        self.cls    = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.weight + self.bias
        cls = self.cls.expand(x.size(0), -1, -1)
        return torch.cat([cls, tokens], dim=1)


class _FTBlock(nn.Module):
    def __init__(self, d_token, n_heads, ffn_d, attn_drop, ffn_drop):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        self.attn  = nn.MultiheadAttention(d_token, n_heads, dropout=attn_drop, batch_first=True)
        self.norm2 = nn.LayerNorm(d_token)
        self.ffn   = nn.Sequential(
            nn.Linear(d_token, ffn_d), nn.GELU(), nn.Dropout(ffn_drop),
            nn.Linear(ffn_d, d_token), nn.Dropout(ffn_drop))

    def forward(self, x):
        h = self.norm1(x); h, _ = self.attn(h, h, h); x = x + h
        h = self.norm2(x); h = self.ffn(h);            x = x + h
        return x


class FTTransformer(nn.Module):
    def __init__(self, n_features, d_token=192, n_blocks=3, n_heads=8,
                 ffn_d=256, attn_drop=0.2, ffn_drop=0.1):
        super().__init__()
        self.tok    = _FeatureTokenizer(n_features, d_token)
        self.blocks = nn.ModuleList([_FTBlock(d_token, n_heads, ffn_d, attn_drop, ffn_drop)
                                     for _ in range(n_blocks)])
        self.norm   = nn.LayerNorm(d_token)
        self.head   = nn.Linear(d_token, 1)

    def forward(self, x):
        t = self.tok(x)
        for b in self.blocks: t = b(t)
        return self.head(self.norm(t[:, 0])).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
#  TabR model definition (Gorishniy et al., ICLR 2024)
# ══════════════════════════════════════════════════════════════════════════════

class TabR(nn.Module):
    def __init__(self, n_features, d_embed=128, d_label=8, K=96):
        super().__init__()
        self.d_embed = d_embed
        self.K = K
        self.encoder   = nn.Sequential(nn.Linear(n_features, d_embed), nn.ReLU(),
                                       nn.Linear(d_embed, d_embed))
        self.label_emb = nn.Embedding(2, d_label)
        self.head      = nn.Sequential(
            nn.LayerNorm(d_embed + d_embed + d_label),
            nn.Linear(d_embed + d_embed + d_label, d_embed),
            nn.ReLU(), nn.Linear(d_embed, 1))

    def encode(self, x):
        e = self.encoder(x)
        return e / (e.norm(dim=-1, keepdim=True) + 1e-8)

    def forward(self, x, cand_emb, cand_labels):
        q   = self.encode(x)
        sim = q @ cand_emb.T
        K   = min(self.K, cand_emb.size(0))
        _, top = sim.topk(K, dim=1)
        c_emb  = cand_emb[top]
        c_kv   = torch.cat([c_emb, self.label_emb(cand_labels[top])], dim=-1)
        w = torch.softmax((q.unsqueeze(1) * c_emb).sum(-1) / math.sqrt(self.d_embed), dim=-1)
        ctx = (w.unsqueeze(-1) * c_kv).sum(1)
        return self.head(torch.cat([q, ctx], dim=-1)).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
#  Training helpers
# ══════════════════════════════════════════════════════════════════════════════

def _train_ftt(Xtr, ytr, Xva, yva, n_feat, pos_w, seed,
               max_ep=200, patience=20, bs=256, lr=1e-4, wd=1e-5):
    torch.manual_seed(seed)
    model   = FTTransformer(n_feat).to(DEVICE)
    opt     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=DEVICE))
    loader  = DataLoader(TensorDataset(torch.tensor(Xtr, dtype=torch.float32),
                                       torch.tensor(ytr, dtype=torch.float32)),
                         batch_size=bs, shuffle=True)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=DEVICE)
    best, ep0, state, no_imp = 0., 0, None, 0
    for ep in range(1, max_ep + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb.to(DEVICE)), yb.to(DEVICE)).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(Xva_t)).cpu().numpy()
        try:    auc = roc_auc_score(yva, p)
        except: auc = 0.
        if auc > best:
            best, ep0 = auc, ep
            state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience: break
    model.load_state_dict(state)
    return model


def _pred_ftt(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X, dtype=torch.float32, device=DEVICE))).cpu().numpy()


def _train_tabr(Xtr, ytr, Xva, yva, n_feat, pos_w, seed,
                max_ep=200, patience=20, bs=256, lr=1e-4):
    torch.manual_seed(seed)
    model   = TabR(n_feat).to(DEVICE)
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=DEVICE))
    Xtr_t   = torch.tensor(Xtr, dtype=torch.float32)
    ytr_i   = torch.tensor(ytr, dtype=torch.long, device=DEVICE)
    Xva_t   = torch.tensor(Xva, dtype=torch.float32, device=DEVICE)
    loader  = DataLoader(TensorDataset(Xtr_t, torch.tensor(ytr, dtype=torch.float32)),
                         batch_size=bs, shuffle=True)
    best, ep0, state, no_imp = 0., 0, None, 0
    for ep in range(1, max_ep + 1):
        model.eval()
        with torch.no_grad():
            chunks = [model.encode(Xtr_t[i:i+512].to(DEVICE)) for i in range(0, len(Xtr_t), 512)]
            all_emb = torch.cat(chunks)
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            loss_fn(model(xb.to(DEVICE), all_emb, ytr_i), yb.to(DEVICE)).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(Xva_t, all_emb, ytr_i)).cpu().numpy()
        try:    auc = roc_auc_score(yva, p)
        except: auc = 0.
        if auc > best:
            best, ep0 = auc, ep
            state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience: break
    model.load_state_dict(state)
    # Recompute final training embeddings for inference retrieval
    model.eval()
    with torch.no_grad():
        chunks = [model.encode(Xtr_t[i:i+512].to(DEVICE)) for i in range(0, len(Xtr_t), 512)]
        final_emb = torch.cat(chunks)
    return model, final_emb, ytr_i


def _pred_tabr(model, X, train_emb, train_lbl):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(
            model(torch.tensor(X, dtype=torch.float32, device=DEVICE), train_emb, train_lbl)
        ).cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
#  Load data
# ══════════════════════════════════════════════════════════════════════════════

print("\n載入 MIMIC 資料...")
mimic_lab = pd.read_csv(os.path.join(MIMIC_DIR, 'mimic_lab.csv'))
mimic_lab['SIGNIN'] = pd.to_datetime(mimic_lab['SIGNIN'], errors='coerce')
mimic_lab['RESULT'] = pd.to_numeric(mimic_lab['RESULT'], errors='coerce')
mimic_lab = mimic_lab.sort_values(['Newid', 'SIGNIN']).copy()
gcsf = mg.load_gcsf(MIMIC_DIR)

_mc = pd.read_csv(os.path.join(MIMIC_DIR, 'mimic_chemo.csv'))
_mc['time'] = pd.to_datetime(_mc['time'], errors='coerce')
_mc = _mc.dropna(subset=['Newid', 'time']).copy()
_mc['canon'] = _mc['item'].apply(pc.canonicalize_drug)
_conv = _mc.apply(lambda r: pc.convert_dose_to_mg(r['item'], r['dose'], r['unit']),
                  axis=1, result_type='expand')
_mc['dose_mg'] = _conv[0].astype(float); _mc['admin_cnt'] = _conv[1].astype(int)
mimic_chemo  = (_mc[_mc['canon'] != 'GCSF']
                [['Newid', 'time', 'canon', 'dose_mg', 'admin_cnt']]
                .sort_values(['Newid', 'time']).reset_index(drop=True))
mimic_cm_info = eac.build_cm_info(pd.read_csv(os.path.join(MIMIC_DIR, 'mimic_patients.csv')))
print(f"  MIMIC 病人: {mimic_lab['Newid'].nunique()} | 化療: {len(mimic_chemo):,}")

print("載入台灣資料...")
tw_df    = pc.load_lab_df(exclude_heme_mode=HEME_MODE)
tw_chemo = pc.load_chemo_filtered()
tw_cm    = eac.build_cm_info(pd.read_csv(pc.CANCER_MAP_PATH))
print(f"  台灣病人: {tw_df['Newid'].nunique()}")

LGB_PARAMS = {'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05,
              'num_leaves': 31, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
              'bagging_freq': 5, 'verbose': -1}

# ══════════════════════════════════════════════════════════════════════════════
#  Results storage
# ══════════════════════════════════════════════════════════════════════════════
# all_res[horizon][thr][model]['tw'|'mimic'] = list of AUCs across seeds
all_res = {h: {thr: {m: {'tw': [], 'mimic': []} for m in MODELS}
               for thr in THRESHOLDS}
           for h in HORIZONS}

# ══════════════════════════════════════════════════════════════════════════════
#  Main loop
# ══════════════════════════════════════════════════════════════════════════════

for HORIZON in HORIZONS:
    print(f"\n{'#'*70}\n  HORIZON = {HORIZON}d  |  max_obs_days={MAX_OBS_DAYS}\n{'#'*70}")

    for seed in SEEDS:
        print(f"\n{'='*70}\n  Seed {seed}\n{'='*70}")
        torch.manual_seed(seed); np.random.seed(seed)

        # ── Split ─────────────────────────────────────────────────────────────
        uniq = tw_df['Newid'].unique()
        tr_i, te_i = next(GroupShuffleSplit(test_size=0.2, random_state=seed).split(uniq, groups=uniq))
        train_pts, test_pts = set(uniq[tr_i]), set(uniq[te_i])

        markers  = sorted(tw_df[tw_df['Newid'].isin(train_pts)]['NAME'].dropna().unique().tolist())
        top_drugs = pc.select_top_k_drugs(tw_chemo, train_pts, pc.TOP_K_DRUGS)

        # ── Build Taiwan anchors ──────────────────────────────────────────────
        tw_data = eac.build_anchor_dataset(
            tw_df, tw_chemo, tw_cm, top_drugs, markers,
            horizon=HORIZON, gap=GAP, regimen_days=0,
            regimen_mode=REGIMEN_MODE, block_gap_hours=BLOCK_GAP,
            max_block_days=MAX_BLOCK, max_obs_days=MAX_OBS_DAYS, verbose=False)
        fc      = [c for c in tw_data.columns if c not in ('Newid', 'min_win_anc')]
        tw_tr   = tw_data[tw_data['Newid'].isin(train_pts)]
        tw_te   = tw_data[tw_data['Newid'].isin(test_pts)]
        Xtr     = tw_tr[fc].values;  anc_tr = tw_tr['min_win_anc'].values
        Xte     = tw_te[fc].values;  anc_te = tw_te['min_win_anc'].values
        print(f"  TW train: {len(Xtr):,} | test: {len(Xte):,}")

        # ── Build MIMIC anchors ───────────────────────────────────────────────
        mimic_data = eac.build_anchor_dataset(
            mimic_lab, mimic_chemo, mimic_cm_info, top_drugs, markers,
            horizon=HORIZON, gap=GAP, regimen_days=0,
            regimen_mode=REGIMEN_MODE, block_gap_hours=BLOCK_GAP,
            max_block_days=MAX_BLOCK, max_obs_days=MAX_OBS_DAYS, verbose=False, include_window=True)
        for c in fc:
            if c not in mimic_data.columns: mimic_data[c] = 0.0
        X_mi    = mimic_data[fc].values
        anc_mi  = mimic_data['min_win_anc'].values
        print(f"  MIMIC anchors: {len(X_mi):,}")
        gk = mg.keep_masks(mimic_data, mimic_lab, gcsf, THRESHOLDS); mg.report(gk)

        # ── StandardScaler (fit on TW train, apply to TW test + MIMIC) ───────
        sc      = StandardScaler().fit(Xtr)
        Xtr_s   = sc.transform(Xtr).astype(np.float32)
        Xte_s   = sc.transform(Xte).astype(np.float32)
        X_mi_s  = sc.transform(X_mi).astype(np.float32)
        X_mi_all, X_mi_s_all, anc_mi_all = X_mi, X_mi_s, anc_mi

        cut = int(0.85 * len(Xtr))

        for thr in THRESHOLDS:
            k = gk[thr]
            X_mi, X_mi_s, anc_mi = X_mi_all[k], X_mi_s_all[k], anc_mi_all[k]
            ytr   = (anc_tr < thr).astype(int)
            n_pos = int(ytr.sum())
            if n_pos < 10: continue
            n_neg = len(ytr) - n_pos
            spw   = float(n_neg) / n_pos
            yte   = (anc_te < thr).astype(int)
            ymi   = (anc_mi < thr).astype(int)

            print(f"\n  --- {LABELS[thr]} | n_pos_train={n_pos} ---")
            print(f"  {'Model':<16} {'TW AUC':>8}  {'MIMIC AUC':>10}")
            print(f"  {'-'*40}")

            def _rec(model_name, prob_tw, prob_mi):
                def _auc(y, p):
                    try: return roc_auc_score(y, p)
                    except: return float('nan')
                a_tw = _auc(yte, prob_tw); a_mi = _auc(ymi, prob_mi)
                all_res[HORIZON][thr][model_name]['tw'].append(a_tw)
                all_res[HORIZON][thr][model_name]['mimic'].append(a_mi)
                print(f"  {model_name:<16} {a_tw:>8.3f}  {a_mi:>10.3f}")

            # ── LightGBM ─────────────────────────────────────────────────────
            p = dict(LGB_PARAMS); p['scale_pos_weight'] = spw
            dtr = lgb.Dataset(Xtr[:cut], label=ytr[:cut])
            dva = lgb.Dataset(Xtr[cut:], label=ytr[cut:], reference=dtr)
            m_lgb = lgb.train(p, dtr, num_boost_round=1000, valid_sets=[dva],
                              callbacks=[lgb.early_stopping(30, verbose=False)])
            _rec('LightGBM',
                 m_lgb.predict(Xte, num_iteration=m_lgb.best_iteration),
                 m_lgb.predict(X_mi, num_iteration=m_lgb.best_iteration))

            # ── XGBoost ──────────────────────────────────────────────────────
            m_xgb = xgb.XGBClassifier(
                n_estimators=1000, learning_rate=0.05, max_depth=6,
                scale_pos_weight=spw, eval_metric='auc',
                early_stopping_rounds=30, random_state=seed, verbosity=0)
            m_xgb.fit(Xtr[:cut], ytr[:cut], eval_set=[(Xtr[cut:], ytr[cut:])], verbose=False)
            _rec('XGBoost',
                 m_xgb.predict_proba(Xte)[:, 1],
                 m_xgb.predict_proba(X_mi)[:, 1])

            # ── LR ───────────────────────────────────────────────────────────
            m_lr = LogisticRegression(C=1.0, class_weight='balanced',
                                      max_iter=1000, solver='lbfgs', random_state=seed)
            m_lr.fit(Xtr_s, ytr)
            _rec('LR',
                 m_lr.predict_proba(Xte_s)[:, 1],
                 m_lr.predict_proba(X_mi_s)[:, 1])

            # ── MLP ──────────────────────────────────────────────────────────
            pos_idx = np.where(ytr == 1)[0]
            reps    = max(int(round(n_neg / n_pos)) - 1, 0)
            idx_os  = (np.concatenate([np.arange(len(ytr))] + [pos_idx] * reps)
                       if reps > 0 else np.arange(len(ytr)))
            rng = np.random.RandomState(seed); rng.shuffle(idx_os)
            m_mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu',
                                  alpha=1e-3, early_stopping=True, n_iter_no_change=15,
                                  max_iter=300, random_state=seed)
            m_mlp.fit(Xtr_s[idx_os], ytr[idx_os])
            _rec('MLP',
                 m_mlp.predict_proba(Xte_s)[:, 1],
                 m_mlp.predict_proba(X_mi_s)[:, 1])

            # ── FT-Transformer ───────────────────────────────────────────────
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*pin_memory.*')
                m_ftt = _train_ftt(Xtr_s[:cut], ytr[:cut], Xtr_s[cut:], ytr[cut:],
                                   Xtr_s.shape[1], spw, seed)
            _rec('FT-Transformer', _pred_ftt(m_ftt, Xte_s), _pred_ftt(m_ftt, X_mi_s))

            # ── TabR ─────────────────────────────────────────────────────────
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*pin_memory.*')
                m_tabr, tr_emb, tr_lbl = _train_tabr(
                    Xtr_s[:cut], ytr[:cut], Xtr_s[cut:], ytr[cut:],
                    Xtr_s.shape[1], spw, seed)
            _rec('TabR', _pred_tabr(m_tabr, Xte_s, tr_emb, tr_lbl),
                         _pred_tabr(m_tabr, X_mi_s, tr_emb, tr_lbl))


# ══════════════════════════════════════════════════════════════════════════════
#  Summary table: 3-seed mean ± SD for each model
# ══════════════════════════════════════════════════════════════════════════════

for HORIZON in HORIZONS:
    print(f"\n{'='*80}")
    print(f"=== HORIZON={HORIZON}d  max_obs_days={MAX_OBS_DAYS}  |  3-seed mean ± SD ===")
    print(f"{'='*80}")
    for thr in THRESHOLDS:
        print(f"\n{LABELS[thr]}")
        print(f"  {'Model':<16}  {'TW test AUROC':>16}  {'MIMIC AUROC':>16}  {'ΔAUROC':>8}")
        print(f"  {'-'*60}")
        for m in MODELS:
            tw_v  = all_res[HORIZON][thr][m]['tw']
            mi_v  = all_res[HORIZON][thr][m]['mimic']
            tw_s  = f"{np.mean(tw_v):.3f} ± {np.std(tw_v):.3f}" if tw_v else "N/A"
            mi_s  = f"{np.mean(mi_v):.3f} ± {np.std(mi_v):.3f}" if mi_v else "N/A"
            delta = (f"{np.mean(mi_v) - np.mean(tw_v):+.3f}"
                     if (tw_v and mi_v) else "N/A")
            print(f"  {m:<16}  {tw_s:>16}  {mi_s:>16}  {delta:>8}")

print(f"""
注意:
  - max_obs_days={MAX_OBS_DAYS}: 所有 lab features 僅用 T 前 {MAX_OBS_DAYS} 天內資料
  - MIMIC 的 init_{{drug}}_dose / cum_dose 全為 0 (emar 無欄位)
  - ΔAUROC = MIMIC mean − TW test mean (負值 = domain shift 衰退)
  - TabR 在 MIMIC inference 時從台灣訓練集做 k-NN retrieval
  - device={DEVICE}
""")
