"""
uq_selective_risk_coverage.py — Conformal selective risk-coverage analysis on the frozen UQ pipeline.

沿用 uq_eventanchored.py 完全相同的 patient-level 60/20/20 train/calib/test 切分、
LightGBM 訓練參數、isotonic calibration。差異:
  1. 只跑 severe (ANC<500) 閾值 (Table 8 / 主 risk-coverage 圖只關心這個閾值)
  2. conformal alpha 從固定 0.10 換成一組 grid,對每個 alpha 重新用 calibration set
     算 Mondrian class-conditional quantile (q_+/q_-),不重新訓練模型、不重新 fit isotonic
  3. 每個 alpha 額外輸出:
       overall / positive-class / negative-class coverage
       acceptance(singleton) rate, abstention rate
       selective error rate among singleton predictions
       confident-low PPV
       {low} / {not-low} / {low,not-low} / {} 比例
  4. 同時保留原本 |p-0.5| 的 AUROC-vs-coverage 曲線 (改名為 complementary
     probability-margin analysis,輸出另一份 CSV 給 Supplementary 圖用)

用法:
  python uq_selective_risk_coverage.py --seed 42 --horizon 28 \
      --regimen-mode block --heme-mode leukemia_only_adj
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
import pipeline_common as pc
import eventanchor_common as eac

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--horizon', type=int, default=28)
parser.add_argument('--regimen-days', type=int, default=0)
parser.add_argument('--regimen-mode', choices=['day', 'block'], default='block')
parser.add_argument('--block-gap-hours', type=float, default=36.0)
parser.add_argument('--max-block-days', type=int, default=7)
parser.add_argument('--max-obs-days', type=int, default=28,
                    help='觀察窗 (天); 預設 28, 對齊 uq_eventanchored.py')
parser.add_argument('--heme-mode', default='leukemia_only',
                    choices=['none', 'all', 'leukemia_only', 'leukemia_only_adj'])
args = parser.parse_args()

SEED = args.seed
HORIZON = args.horizon
REGIMEN_MODE = args.regimen_mode
HEME_MODE = args.heme_mode
REGTAG = f'reg{args.regimen_days}' if REGIMEN_MODE == 'day' else f'block{args.block_gap_hours:g}h'
if HEME_MODE != 'leukemia_only':
    REGTAG = f'{REGTAG}_{pc._TAG_MAP[HEME_MODE]}'

ALPHA_GRID = [0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30]
THR = 0.5  # severe (ANC<500) only
DEFAULT_PARAMS = {
    'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05,
    'num_leaves': 31, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
    'bagging_freq': 5, 'verbose': -1,
}

print(f"=== risk-coverage sweep | horizon={HORIZON} regtag={REGTAG} seed={SEED} ===")

# ---------------------------------------------------------------------------
# 1. load + patient-level 三分 (與 uq_eventanchored.py 一致)
# ---------------------------------------------------------------------------
df = pc.load_lab_df(exclude_heme_mode=HEME_MODE)
chemo_df = pc.load_chemo_filtered()
cm_info = eac.build_cm_info(pd.read_csv(pc.CANCER_MAP_PATH))

patients = df['Newid'].unique()
gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
rest_idx, test_idx = next(gss1.split(patients, groups=patients))
rest_pat = patients[rest_idx]
test_pat = set(patients[test_idx])
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
tr_idx, ca_idx = next(gss2.split(rest_pat, groups=rest_pat))
train_pat = set(rest_pat[tr_idx])
calib_pat = set(rest_pat[ca_idx])
print(f"Patients — train: {len(train_pat)} | calib: {len(calib_pat)} | TEST: {len(test_pat)}")

EXPECTED_MARKERS = sorted(df[df['Newid'].isin(train_pat)]['NAME'].dropna().unique().tolist())
top_drugs = pc.select_top_k_drugs(chemo_df, train_pat, pc.TOP_K_DRUGS)

data = eac.build_anchor_dataset(df, chemo_df, cm_info, top_drugs, EXPECTED_MARKERS,
                                horizon=HORIZON, gap=10, regimen_days=args.regimen_days,
                                regimen_mode=REGIMEN_MODE,
                                block_gap_hours=args.block_gap_hours,
                                max_block_days=args.max_block_days,
                                max_obs_days=args.max_obs_days)
feat_cols = [c for c in data.columns if c not in ('Newid', 'min_win_anc')]


def subset(pat_set):
    m = data['Newid'].isin(pat_set)
    return (data.loc[m, feat_cols].values,
            data.loc[m, 'min_win_anc'].values,
            data.loc[m, 'Newid'].values)


Xtr, anc_tr, _ = subset(train_pat)
Xca, anc_ca, pid_ca = subset(calib_pat)
Xte, anc_te, pid_te = subset(test_pat)
print(f"Anchors — train: {len(Xtr)} | calib: {len(Xca)} | TEST: {len(Xte)}")

ytr = (anc_tr < THR).astype(int)
yca = (anc_ca < THR).astype(int)
yte = (anc_te < THR).astype(int)
n_pos = int(ytr.sum())
n_neg = len(ytr) - n_pos

# ---------------------------------------------------------------------------
# 2. train model + isotonic calibration (與 uq_eventanchored.py 一致,只訓練 1 次)
# ---------------------------------------------------------------------------
params = dict(DEFAULT_PARAMS)
params['scale_pos_weight'] = n_neg / n_pos
cut = int(0.85 * len(Xtr))
dtr = lgb.Dataset(Xtr[:cut], label=ytr[:cut])
dva = lgb.Dataset(Xtr[cut:], label=ytr[cut:], reference=dtr)
model = lgb.train(params, dtr, num_boost_round=1000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
p_ca_raw = model.predict(Xca, num_iteration=model.best_iteration)
p_te_raw = model.predict(Xte, num_iteration=model.best_iteration)

iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(p_ca_raw, yca)
p_ca = iso.predict(p_ca_raw)
p_te = iso.predict(p_te_raw)

brier_raw = brier_score_loss(yte, p_te_raw)
brier_cal = brier_score_loss(yte, p_te)
auc_te = roc_auc_score(yte, p_te)
prev = yte.mean()
print(f"[Model]  AUC={auc_te:.3f} | Brier {brier_raw:.4f} -> {brier_cal:.4f} | "
      f"n_pos={int(yte.sum())} prev={prev*100:.1f}%")


def conformal_q(scores, alpha):
    n = len(scores)
    if n == 0:
        return 1.0
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level, method='higher'))


def first_anchor_mask(pids):
    seen = set()
    keep = np.zeros(len(pids), dtype=bool)
    for i, p in enumerate(pids):
        if p not in seen:
            seen.add(p)
            keep[i] = True
    return keep


fam = first_anchor_mask(pid_te)
s_pos = 1 - p_ca[yca == 1]  # nonconformity for true positives (label=low)
s_neg = p_ca[yca == 0]      # nonconformity for true negatives (label=not-low)

# ---------------------------------------------------------------------------
# 3. alpha grid sweep (純後處理,不重新訓練/不重新 fit isotonic)
# ---------------------------------------------------------------------------
rows = []
for alpha in ALPHA_GRID:
    q_pos = conformal_q(s_pos, alpha)
    q_neg = conformal_q(s_neg, alpha)
    in_low = (1 - p_te) <= q_pos
    in_notlow = p_te <= q_neg

    covered = np.where(yte == 1, in_low, in_notlow)
    cov_marg = covered.mean()
    cov_pos = covered[yte == 1].mean()
    cov_neg = covered[yte == 0].mean()
    cov_patient = covered[fam].mean()

    singleton = in_low ^ in_notlow
    both = in_low & in_notlow
    empty = ~in_low & ~in_notlow
    abstain = both | empty

    pred_low_conf = in_low & ~in_notlow
    pred_notlow_conf = in_notlow & ~in_low
    ppv_conf = yte[pred_low_conf].mean() if pred_low_conf.sum() > 0 else float('nan')

    # selective error rate among singleton predictions
    err_low = pred_low_conf & (yte == 0)      # 預測 low, 實際 not-low
    err_notlow = pred_notlow_conf & (yte == 1)  # 預測 not-low, 實際 low
    n_singleton = int(singleton.sum())
    sel_err = (int(err_low.sum()) + int(err_notlow.sum())) / n_singleton if n_singleton > 0 else float('nan')

    rows.append({
        'alpha': alpha, 'target_coverage': 1 - alpha,
        'q_pos': q_pos, 'q_neg': q_neg,
        'overall_coverage': cov_marg, 'positive_class_coverage': cov_pos,
        'negative_class_coverage': cov_neg,
        'patient_level_coverage': cov_patient,
        'singleton_acceptance_rate': singleton.mean(), 'abstention_rate': abstain.mean(),
        'selective_error_rate_among_singletons': sel_err,
        'confident_low_ppv': ppv_conf,
        'set_pct_low': pred_low_conf.mean(), 'set_pct_notlow': pred_notlow_conf.mean(),
        'set_pct_low_and_notlow': both.mean(), 'set_pct_empty': empty.mean(),
    })
    print(f"  alpha={alpha:<6} target_cov={1-alpha:.0%}  marg_cov={cov_marg*100:5.1f}% "
          f"(pos={cov_pos*100:5.1f}%, neg={cov_neg*100:5.1f}%)  "
          f"accept={singleton.mean()*100:5.1f}%  abstain={abstain.mean()*100:5.1f}%  "
          f"sel_err={sel_err*100:5.1f}%  ppv_low={ppv_conf*100:5.1f}%")

out = pd.DataFrame(rows)
out['seed'] = SEED
out['horizon'] = HORIZON
out['regtag'] = REGTAG
out['auc'] = auc_te
out['brier_raw'] = brier_raw
out['brier_cal'] = brier_cal
out['n_test'] = len(yte)
out['n_pos'] = int(yte.sum())
out['prevalence'] = prev

csv_path = os.path.join(_HERE, f'riskcov_h{HORIZON}_{REGTAG}_seed{SEED}.csv')
out.to_csv(csv_path, index=False)
print(f"\n✓ risk-coverage CSV → {csv_path}")

# ---------------------------------------------------------------------------
# 4. complementary probability-margin analysis (原 |p-0.5| AUROC-vs-coverage, 改名保留)
# ---------------------------------------------------------------------------
order = np.argsort(-np.abs(p_te - 0.5))
cov_grid = np.linspace(0.2, 1.0, 9)
auc_curve = []
for c in cov_grid:
    k = max(int(c * len(p_te)), 10)
    idx = order[:k]
    yy = yte[idx]
    if yy.sum() >= 2 and (1 - yy).sum() >= 2:
        auc_curve.append(roc_auc_score(yy, p_te[idx]))
    else:
        auc_curve.append(np.nan)

margin_df = pd.DataFrame({'coverage_pct': cov_grid * 100, 'auroc': auc_curve})
margin_df['seed'] = SEED
margin_df['horizon'] = HORIZON
margin_df['regtag'] = REGTAG
margin_csv = os.path.join(_HERE, f'margin_h{HORIZON}_{REGTAG}_seed{SEED}.csv')
margin_df.to_csv(margin_csv, index=False)
print(f"✓ probability-margin CSV → {margin_csv}")
