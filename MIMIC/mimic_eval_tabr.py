"""
mimic_eval_tabr.py — TabR 台灣訓練 → MIMIC-IV 外部驗證（vanilla: seed=42）

用法 (從 MIMIC/ 執行):
  python mimic_eval_tabr.py --max-obs-days 28
  python mimic_eval_tabr.py --horizons 7 14 21 28 --only-severe --max-obs-days 28
"""
import os, sys, math, argparse, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_HERE = os.path.dirname(os.path.abspath(__file__))
_TW   = os.path.join(os.path.dirname(_HERE), 'CCH')
sys.path.insert(0, _HERE); sys.path.insert(0, _TW)

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import pipeline_common as pc
import eventanchor_common as eac
import mimic_gcsf as mg

# CPU + deterministic: MPS/CUDA 非決定性會讓同一 seed 重跑 AUROC 漂移 (~0.005)。
# 本腳本是 TabR 唯一來源 (CCH test NPZ + MIMIC), 報告的 Table 8/10/17 都讀這次的結果。
DEVICE = torch.device('cpu')
torch.use_deterministic_algorithms(True)

parser = argparse.ArgumentParser()
parser.add_argument('--seed',         type=int, default=42)
parser.add_argument('--horizon',      type=int, default=28)
parser.add_argument('--horizons',     type=int, nargs='+', default=None,
                    help='Multiple horizons; overrides --horizon')
parser.add_argument('--max-obs-days', type=int, default=28)
parser.add_argument('--only-severe',  action='store_true',
                    help='Only evaluate <500 threshold')
parser.add_argument('--fixed-cohort', action='store_true',
                    help='用 mimic_cohort_fixed.csv 修正後 cohort (排除無惡性腫瘤碼病人, 修正癌別)')
args = parser.parse_args()
SEED         = args.seed
HORIZONS     = args.horizons if args.horizons else [args.horizon]
MAX_OBS_DAYS = args.max_obs_days

torch.manual_seed(SEED); np.random.seed(SEED)

GAP, REGIMEN_MODE, BLOCK_GAP, MAX_BLOCK = 10, 'block', 36.0, 7
HEME_MODE  = 'leukemia_only_adj'
THRESHOLDS = [0.5] if args.only_severe else [1.5, 1.0, 0.5]
LABELS     = {1.5: '<1500 (mild)', 1.0: '<1000 (moderate)', 0.5: '<500 (severe)'}
MIMIC_DIR  = os.path.join(_HERE, 'mimic_data')

print(f"TabR MIMIC eval (vanilla) | seed={SEED} | horizons={HORIZONS} | "
      f"max_obs_days={MAX_OBS_DAYS} | only_severe={args.only_severe} | "
      f"fixed_cohort={args.fixed_cohort} | device={DEVICE}")


# ── TabR ──────────────────────────────────────────────────────────────────────
class TabR(nn.Module):
    def __init__(self, n_features, d_embed=128, d_label=8, K=96):
        super().__init__()
        self.d_embed = d_embed; self.K = K
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
        c_emb = cand_emb[top]
        c_kv  = torch.cat([c_emb, self.label_emb(cand_labels[top])], dim=-1)
        w = torch.softmax((q.unsqueeze(1) * c_emb).sum(-1) / math.sqrt(self.d_embed), dim=-1)
        ctx = (w.unsqueeze(-1) * c_kv).sum(1)
        return self.head(torch.cat([q, ctx], dim=-1)).squeeze(-1)


def train_tabr(Xtr, ytr, Xva, yva, n_feat, pos_w,
               max_ep=200, patience=20, bs=256, lr=1e-4):
    model   = TabR(n_feat).to(DEVICE)
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=DEVICE))
    Xtr_t   = torch.tensor(Xtr, dtype=torch.float32)
    ytr_i   = torch.tensor(ytr, dtype=torch.long, device=DEVICE)
    Xva_t   = torch.tensor(Xva, dtype=torch.float32, device=DEVICE)
    loader  = DataLoader(TensorDataset(Xtr_t, torch.tensor(ytr, dtype=torch.float32)),
                         batch_size=bs, shuffle=True,
                         generator=torch.Generator().manual_seed(SEED))
    best, state, no_imp = 0., None, 0
    for ep in range(1, max_ep + 1):
        model.eval()
        with torch.no_grad():
            all_emb = torch.cat([model.encode(Xtr_t[i:i+512].to(DEVICE))
                                 for i in range(0, len(Xtr_t), 512)])
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb.to(DEVICE), all_emb, ytr_i), yb.to(DEVICE)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # 須在 backward 之後才有作用
            opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(Xva_t, all_emb, ytr_i)).cpu().numpy()
        try:    auc = roc_auc_score(yva, p)
        except: auc = 0.
        if auc > best:
            best, state, no_imp = auc, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            no_imp += 1
            if no_imp >= patience: break
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        final_emb = torch.cat([model.encode(Xtr_t[i:i+512].to(DEVICE))
                               for i in range(0, len(Xtr_t), 512)])
    return model, final_emb, ytr_i


def predict_tabr(model, X, train_emb, train_lbl):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(
            model(torch.tensor(X, dtype=torch.float32, device=DEVICE), train_emb, train_lbl)
        ).cpu().numpy()


def metrics(y_true, y_prob):
    y = np.asarray(y_true); p = np.asarray(y_prob)
    n_pos = int(y.sum()); n = len(y)
    if n_pos < 2 or (n - n_pos) < 2: return None
    prev   = n_pos / n
    auroc  = roc_auc_score(y, p)
    prauc  = average_precision_score(y, p)
    lift   = prauc / prev
    fpr, tpr, _ = roc_curve(y, p)
    valid  = (1 - fpr) >= 0.8
    sens80 = float(tpr[valid].max()) if valid.any() else float('nan')
    return {'n': n, 'n_pos': n_pos, 'prev': prev,
            'auroc': auroc, 'prauc': prauc, 'lift': lift, 'sens_at_spec80': sens80}


# ── Load data ─────────────────────────────────────────────────────────────────
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
mimic_chemo = (_mc[_mc['canon'] != 'GCSF'][['Newid', 'time', 'canon', 'dose_mg', 'admin_cnt']]
               .sort_values(['Newid', 'time']).reset_index(drop=True))
_pts = pd.read_csv(os.path.join(MIMIC_DIR, 'mimic_patients.csv'))
if args.fixed_cohort:
    # 與 mimic_eval_lgb_fn.py 相同: 只留 keep=1, 癌別改用修正版
    _fix  = pd.read_csv(os.path.join(MIMIC_DIR, 'mimic_cohort_fixed.csv'))
    _keep = set(_fix.loc[_fix['keep'] == 1, 'Newid'])
    _pts  = (_pts.drop(columns=['cancer_category', 'primary_site'])
             .merge(_fix[['Newid', 'cancer_category', 'primary_site']], on='Newid'))
    _pts  = _pts[_pts['Newid'].isin(_keep)]
    mimic_lab   = mimic_lab[mimic_lab['Newid'].isin(_keep)]
    mimic_chemo = mimic_chemo[mimic_chemo['Newid'].isin(_keep)]
mimic_cm_info = eac.build_cm_info(_pts)
print(f"  MIMIC 病人: {mimic_lab['Newid'].nunique()} | 化療: {len(mimic_chemo):,}")

print("載入台灣資料...")
tw_df    = pc.load_lab_df(exclude_heme_mode=HEME_MODE)
tw_chemo = pc.load_chemo_filtered()
tw_cm    = eac.build_cm_info(pd.read_csv(pc.CANCER_MAP_PATH))
print(f"  台灣病人: {tw_df['Newid'].nunique()}")

# ── Split ─────────────────────────────────────────────────────────────────────
uniq = tw_df['Newid'].unique()
tr_i, te_i = next(GroupShuffleSplit(test_size=0.2, random_state=SEED).split(uniq, groups=uniq))
train_pts, test_pts = set(uniq[tr_i]), set(uniq[te_i])
markers   = sorted(tw_df[tw_df['Newid'].isin(train_pts)]['NAME'].dropna().unique().tolist())
top_drugs = pc.select_top_k_drugs(tw_chemo, train_pts, pc.TOP_K_DRUGS)

# ── Main loop: horizon × threshold ────────────────────────────────────────────
all_res = {}  # all_res[horizon][thr] = {'tw': r, 'mimic': r}

for HORIZON in HORIZONS:
    print(f"\n{'#'*70}\n  HORIZON = {HORIZON}d\n{'#'*70}")
    all_res[HORIZON] = {}

    print(f"  建台灣 anchor dataset (h{HORIZON})...", flush=True)
    tw_data = eac.build_anchor_dataset(
        tw_df, tw_chemo, tw_cm, top_drugs, markers,
        horizon=HORIZON, gap=GAP, regimen_days=0,
        regimen_mode=REGIMEN_MODE, block_gap_hours=BLOCK_GAP,
        max_block_days=MAX_BLOCK, max_obs_days=MAX_OBS_DAYS, verbose=False)
    fc    = [c for c in tw_data.columns if c not in ('Newid', 'min_win_anc')]
    tw_tr = tw_data[tw_data['Newid'].isin(train_pts)]
    tw_te = tw_data[tw_data['Newid'].isin(test_pts)]
    Xtr_raw = tw_tr[fc].values; anc_tr = tw_tr['min_win_anc'].values
    Xte_raw = tw_te[fc].values; anc_te = tw_te['min_win_anc'].values
    print(f"  TW train: {len(Xtr_raw):,} | test: {len(Xte_raw):,}")

    print(f"  建 MIMIC anchor dataset (h{HORIZON})...", flush=True)
    mimic_data = eac.build_anchor_dataset(
        mimic_lab, mimic_chemo, mimic_cm_info, top_drugs, markers,
        horizon=HORIZON, gap=GAP, regimen_days=0,
        regimen_mode=REGIMEN_MODE, block_gap_hours=BLOCK_GAP,
        max_block_days=MAX_BLOCK, max_obs_days=MAX_OBS_DAYS, verbose=False, include_window=True)
    for c in fc:
        if c not in mimic_data.columns: mimic_data[c] = 0.0
    X_mi_raw = mimic_data[fc].values; anc_mi = mimic_data['min_win_anc'].values
    print(f"  MIMIC: {len(X_mi_raw):,}")
    gk = mg.keep_masks(mimic_data, mimic_lab, gcsf, THRESHOLDS); mg.report(gk)

    sc   = StandardScaler().fit(Xtr_raw)
    Xtr  = sc.transform(Xtr_raw).astype(np.float32)
    Xte  = sc.transform(Xte_raw).astype(np.float32)
    X_mi = sc.transform(X_mi_raw).astype(np.float32)
    CUT  = int(0.85 * len(Xtr))
    npz  = {}
    X_mi_all, anc_mi_all = X_mi, anc_mi

    for thr in THRESHOLDS:
        X_mi, anc_mi = X_mi_all[gk[thr]], anc_mi_all[gk[thr]]
        ytr   = (anc_tr < thr).astype(int)
        n_pos = int(ytr.sum())
        if n_pos < 10:
            continue
        pos_w = float(len(ytr) - n_pos) / n_pos

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*pin_memory.*')
            torch.manual_seed(SEED); np.random.seed(SEED)   # 每個 (h, thr) 獨立可重現
            model, train_emb, train_lbl = train_tabr(
                Xtr[:CUT], ytr[:CUT], Xtr[CUT:], ytr[CUT:], Xtr.shape[1], pos_w)

        prob_tw = predict_tabr(model, Xte, train_emb, train_lbl)
        prob_mi = predict_tabr(model, X_mi, train_emb, train_lbl)

        npz[f'thr{thr}_pid']   = tw_te['Newid'].values
        npz[f'thr{thr}_pred']  = prob_tw
        npz[f'thr{thr}_truth'] = anc_te          # raw ANC; 下游再以 < thr 二值化
        r_tw = metrics((anc_te < thr).astype(int), prob_tw)
        r_mi = metrics((anc_mi < thr).astype(int), prob_mi)
        all_res[HORIZON][thr] = {'tw': r_tw, 'mimic': r_mi}

        for label, r in [('TW', r_tw), ('MIMIC', r_mi)]:
            if r:
                delta = (f"  Δ={r_mi['auroc']-r_tw['auroc']:+.3f}"
                         if label == 'MIMIC' and r_tw else '')
                print(f"  {LABELS[thr]} | {label}: AUROC={r['auroc']:.3f} "
                      f"PR-AUC={r['prauc']:.3f} Lift={r['lift']:.2f}x "
                      f"Prev={r['prev']*100:.1f}%{delta}")

    # CCH test 預測存檔 (與 tabr_eventanchored.py 同名同格式, 供 Table 8/10/17 使用)
    npz['_thresholds'] = np.array(THRESHOLDS); npz['_horizon'] = np.array([HORIZON])
    npz['_seed'] = np.array([SEED])
    _out = os.path.join(_TW, f'predictions_tabr_h{HORIZON}_noleukadj_seed{SEED}.npz')
    np.savez_compressed(_out, **npz)
    print(f"  ✓ CCH test 預測 → {_out}")

# ── Summary table ──────────────────────────────────────────────────────────────
print(f"\n\n{'='*100}")
print(f"=== TabR MIMIC 外部驗證摘要 | seed={SEED} | max_obs_days={MAX_OBS_DAYS} ===")
print(f"{'='*100}")

for thr in THRESHOLDS:
    print(f"\n── {LABELS[thr]} ──")
    print(f"  {'H':>4}  {'Set':>5}  {'AUROC':>7}  {'PR-AUC':>7}  {'Lift':>6}  "
          f"{'Prev':>7}  {'n_pos':>6}  {'Sens@Sp80':>10}  {'ΔAUROC':>8}")
    print(f"  {'-'*80}")
    for h in sorted(HORIZONS):
        if thr not in all_res.get(h, {}): continue
        r_tw = all_res[h][thr]['tw']
        r_mi = all_res[h][thr]['mimic']
        delta = (f"{r_mi['auroc']-r_tw['auroc']:+.3f}"
                 if (r_tw and r_mi) else 'N/A')
        for label, r in [('TW', r_tw), ('MIMIC', r_mi)]:
            if r:
                d_s = delta if label == 'MIMIC' else ''
                print(f"  {h:>3}d  {label:>5}  {r['auroc']:>7.3f}  {r['prauc']:>7.3f}  "
                      f"{r['lift']:>6.2f}  {r['prev']*100:>6.1f}%  {r['n_pos']:>6}  "
                      f"{r['sens_at_spec80']:>10.3f}  {d_s:>8}")
        print()

print(f"""
TabR config (paper defaults):
  d_embed=128, K=96 (retrieved candidates), d_label=8
  lr=1e-4, weight_decay=0, batch_size=256, patience=20
  pos_weight = n_neg/n_pos
  seed={SEED}, horizons={HORIZONS}, max_obs_days={MAX_OBS_DAYS}
  device={DEVICE}
  MIMIC inference: k-NN retrieval from TW training embeddings (cross-domain)
""")
