"""
tabnet_eventanchored.py — TabNet vs MLP baseline, event-anchored tabular

同一份 event-anchored 資料 (leukemia_only_adj + block regimen = final primary)。
TabNet (AAAI 2021) 是專為 tabular 設計的 attention-based DL，與 MLP 對比。
使用 MPS (Apple Silicon) 或 CUDA 自動加速。

用法 (從 CCH/ 執行):
  python tabnet_eventanchored.py --seed 42
  python tabnet_eventanchored.py --seed 42 --n-boot 500

注意: 與 mlp_vs_lgb_eventanchored.py 分開執行是因為
  PyTorch 和 LightGBM 在 Apple Silicon 上共用 OpenMP runtime，
  同一 process 載入會 segfault。分開腳本完全避免衝突。
"""
import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import pipeline_common as pc
import eventanchor_common as eac

# device
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--horizon', type=int, default=28)
parser.add_argument('--n-boot', type=int, default=500,
                    help='Bootstrap resamples for 95%% CI (0 = skip CI)')
args = parser.parse_args()
SEED, HORIZON, N_BOOT = args.seed, args.horizon, args.n_boot
THRESHOLDS = [1.5, 1.0, 0.5]
LABELS = {1.5: '<1500 (mild)', 1.0: '<1000 (moderate)', 0.5: '<500 (severe/FN)'}

print(f"TabNet | event-anchored h{HORIZON} block primary seed{SEED} | device={DEVICE}")

# --- data ---
df = pc.load_lab_df(exclude_heme_mode='leukemia_only_adj')
chemo_df = pc.load_chemo_filtered()
cm_info = eac.build_cm_info(pd.read_csv(pc.CANCER_MAP_PATH))

pats = df['Newid'].unique()
tr_idx, te_idx = next(GroupShuffleSplit(test_size=0.2, random_state=SEED).split(pats, groups=pats))
train_pat, test_pat = set(pats[tr_idx]), set(pats[te_idx])
markers = sorted(df[df['Newid'].isin(train_pat)]['NAME'].dropna().unique().tolist())
top_drugs = pc.select_top_k_drugs(chemo_df, train_pat, pc.TOP_K_DRUGS)

data = eac.build_anchor_dataset(df, chemo_df, cm_info, top_drugs, markers,
                                horizon=HORIZON, gap=10, regimen_days=0,
                                regimen_mode='block')
fc = [c for c in data.columns if c not in ('Newid', 'min_win_anc')]
tr = data['Newid'].isin(train_pat)
te = data['Newid'].isin(test_pat)
Xtr, Xte = data.loc[tr, fc].values, data.loc[te, fc].values
anc_tr = data.loc[tr, 'min_win_anc'].values
anc_te = data.loc[te, 'min_win_anc'].values

sc = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

print(f"Train anchors: {len(Xtr)} | Test anchors: {len(Xte)} | Features: {len(fc)}")


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


def fmt_metric(y_true, y_score, metric_fn):
    pt = metric_fn(y_true, y_score)
    lo, hi = boot_ci(y_true, y_score, metric_fn)
    if lo is None:
        return f"{pt:.3f}"
    return f"{pt:.3f} [{lo:.3f}-{hi:.3f}]"


for thr in THRESHOLDS:
    ytr = (anc_tr < thr).astype(int)
    yte = (anc_te < thr).astype(int)
    n_pos_tr = int(ytr.sum())
    n_neg_tr = len(ytr) - n_pos_tr
    if n_pos_tr < 10:
        continue

    print(f"\n=== {LABELS[thr]} | test n_pos={int(yte.sum())} / n={len(yte)} "
          f"(prev={yte.mean():.1%}) ===")
    print(f"  {'Model':<12}  {'AUROC (95% CI)':<22}  {'PR-AUC (95% CI)':<22}")
    print(f"  {'-'*12}  {'-'*22}  {'-'*22}")

    cut = int(0.85 * len(Xtr))

    # TabNet
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*pin_memory.*')
        m_tabnet = TabNetClassifier(
            n_d=8, n_a=8, n_steps=3, gamma=1.3,
            n_independent=2, n_shared=2, momentum=0.02,
            optimizer_fn=torch.optim.Adam,
            optimizer_params={'lr': 2e-2, 'weight_decay': 1e-5},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            scheduler_params={'step_size': 10, 'gamma': 0.9},
            mask_type='sparsemax',
            device_name=DEVICE,
            verbose=0,
            seed=SEED,
        )
        m_tabnet.fit(
            Xtr[:cut].astype(np.float32), ytr[:cut].astype(np.int64),
            eval_set=[(Xtr[cut:].astype(np.float32), ytr[cut:].astype(np.int64))],
            eval_metric=['auc'],
            max_epochs=200,
            patience=30,
            weights={0: 1.0, 1: float(n_neg_tr) / n_pos_tr},
            batch_size=1024,
            virtual_batch_size=128,
        )
    p_tabnet = m_tabnet.predict_proba(Xte.astype(np.float32))[:, 1]
    print(f"  {'TabNet':<12}  {fmt_metric(yte, p_tabnet, roc_auc_score):<22}  "
          f"{fmt_metric(yte, p_tabnet, average_precision_score):<22}  [best_epoch={m_tabnet.best_epoch}]")

    # MLP (reference baseline from same process — no OpenMP conflict here)
    pos_idx = np.where(ytr == 1)[0]
    reps = max(int(round(n_neg_tr / n_pos_tr)) - 1, 0)
    idx = (np.concatenate([np.arange(len(ytr))] + [pos_idx] * reps)
           if reps > 0 else np.arange(len(ytr)))
    rng = np.random.RandomState(SEED)
    rng.shuffle(idx)
    m_mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', alpha=1e-3,
                          early_stopping=True, n_iter_no_change=15, max_iter=300,
                          random_state=SEED)
    m_mlp.fit(Xtr_s[idx], ytr[idx])
    p_mlp = m_mlp.predict_proba(Xte_s)[:, 1]
    print(f"  {'MLP':<12}  {fmt_metric(yte, p_mlp, roc_auc_score):<22}  "
          f"{fmt_metric(yte, p_mlp, average_precision_score):<22}")

print(f"""
註:
  - TabNet: n_d=n_a=8, n_steps=3, class weights {{0:1, 1:ratio}}, patience=30, device={DEVICE}
  - MLP: (64,32) ReLU, minority oversampling (reference baseline)
  - Pipeline: leukemia_only_adj + block regimen mode (final primary)
  - CI: bootstrap 95% percentile interval (N_BOOT={N_BOOT} resamples)
  - 與 mlp_vs_lgb_eventanchored.py 分開執行避免 PyTorch/LightGBM OpenMP 衝突
""")
