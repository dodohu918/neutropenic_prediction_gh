"""
mlp_vs_lgb_eventanchored.py — LightGBM vs XGBoost vs LR vs MLP, event-anchored tabular

同一份 event-anchored 資料 (leukemia_only_adj + block regimen = final primary),
四個 baseline 全用 library default 超參數, 公平對比 AUROC / PR-AUC (bootstrap 95% CI)。

用法 (從 CCH/ 執行):
  python mlp_vs_lgb_eventanchored.py --seed 42
  python mlp_vs_lgb_eventanchored.py --seed 42 --n-boot 1000

TabNet 比較請用獨立腳本: tabnet_eventanchored.py
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import pipeline_common as pc
import eventanchor_common as eac

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--horizon', type=int, default=28)
parser.add_argument('--n-boot', type=int, default=500,
                    help='Bootstrap resamples for 95%% CI (0 = skip CI)')
parser.add_argument('--max-obs-days', type=int, default=28,
                    help='Only use lab features within this many days before T (None = unlimited)')
args = parser.parse_args()
SEED, HORIZON, N_BOOT, MAX_OBS_DAYS = args.seed, args.horizon, args.n_boot, args.max_obs_days
THRESHOLDS = [1.5, 1.0, 0.5]
LABELS = {1.5: '<1500 (mild)', 1.0: '<1000 (moderate)', 0.5: '<500 (severe/FN)'}

print(f"LGB / XGB / LR / MLP | event-anchored h{HORIZON} block primary seed{SEED} max_obs_days={MAX_OBS_DAYS}")

# --- final primary cohort: leukemia_only_adj + block regimen ---
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
                                max_obs_days=MAX_OBS_DAYS,
                                regimen_mode='block')
fc = [c for c in data.columns if c not in ('Newid', 'min_win_anc')]
tr = data['Newid'].isin(train_pat)
te = data['Newid'].isin(test_pat)
Xtr, Xte = data.loc[tr, fc].values, data.loc[te, fc].values
anc_tr = data.loc[tr, 'min_win_anc'].values
anc_te = data.loc[te, 'min_win_anc'].values

sc = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)


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


LGB_PARAMS = {
    'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05,
    'num_leaves': 31, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
    'bagging_freq': 5, 'verbose': -1,
}

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

    # LightGBM
    p = dict(LGB_PARAMS)
    p['scale_pos_weight'] = n_neg_tr / n_pos_tr
    dtr = lgb.Dataset(Xtr[:cut], label=ytr[:cut])
    dva = lgb.Dataset(Xtr[cut:], label=ytr[cut:], reference=dtr)
    m_lgb = lgb.train(p, dtr, num_boost_round=1000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
    p_lgb = m_lgb.predict(Xte, num_iteration=m_lgb.best_iteration)
    print(f"  {'LightGBM':<12}  {fmt_metric(yte, p_lgb, roc_auc_score):<22}  "
          f"{fmt_metric(yte, p_lgb, average_precision_score):<22}")

    # XGBoost
    m_xgb = xgb.XGBClassifier(
        n_estimators=1000, learning_rate=0.05, max_depth=6,
        scale_pos_weight=n_neg_tr / n_pos_tr,
        eval_metric='auc', early_stopping_rounds=30,
        random_state=SEED, verbosity=0,
    )
    m_xgb.fit(Xtr[:cut], ytr[:cut],
              eval_set=[(Xtr[cut:], ytr[cut:])],
              verbose=False)
    p_xgb = m_xgb.predict_proba(Xte)[:, 1]
    print(f"  {'XGBoost':<12}  {fmt_metric(yte, p_xgb, roc_auc_score):<22}  "
          f"{fmt_metric(yte, p_xgb, average_precision_score):<22}")

    # Logistic Regression
    m_lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000,
                              solver='lbfgs', random_state=SEED)
    m_lr.fit(Xtr_s, ytr)
    p_lr = m_lr.predict_proba(Xte_s)[:, 1]
    print(f"  {'LR':<12}  {fmt_metric(yte, p_lr, roc_auc_score):<22}  "
          f"{fmt_metric(yte, p_lr, average_precision_score):<22}")

    # MLP (minority oversampling since MLPClassifier lacks class_weight)
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

print("""
註:
  - 所有模型 library default 超參數 (無額外 tuning)
  - LightGBM / XGBoost: scale_pos_weight + early stopping (15% val split)
  - LR: C=1.0 (L2), class_weight='balanced', scaled features
  - MLP: (64,32) ReLU, minority oversampling, scaled features
  - Pipeline: leukemia_only_adj + block regimen mode (final primary)
  - CI: bootstrap 95% percentile interval (N_BOOT resamples)
  - TabNet 比較請用: tabnet_eventanchored.py
""")
