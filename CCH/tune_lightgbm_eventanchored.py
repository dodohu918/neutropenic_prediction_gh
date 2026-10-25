"""
tune_lightgbm_eventanchored.py — LightGBM 超參數調校 on event-anchored pipeline

方法學與 MIMIC/tune_lightgbm.py 完全一致，但資料換成 event-anchored tabular
(leukemia_only_adj + block regimen = final primary)。

方法學紀律:
  1. GroupShuffleSplit(seed) 切 train/TEST → 與 mlp_vs_lgb_eventanchored.py 同一份 split。
     TEST 全程不碰，只在最後用最佳參數評估一次。
  2. 在 TRAIN 內用 GroupKFold(on Newid) 做 CV → Optuna 用 CV mean AUC 找參數。
  3. 是否採用 tuned 參數只依 TRAIN 內的 CV 判斷: 用「同一組 folds」算 Default CV AUC,
     與 CV-best AUC 比較 (CV gain)。TEST 不參與任何決策。
  4. 另外報 Tuned / Default TEST AUC 僅供紀錄 (selection gap = CV-best − Tuned TEST)。

用法 (從 CCH/ 執行):
  python tune_lightgbm_eventanchored.py
  python tune_lightgbm_eventanchored.py --threshold 0.5 --n-trials 150

輸出: best_params_ea_<tag>_thr<thr>_seed<seed>.json
"""
import os
import sys
import json
import argparse
import warnings
import numpy as np
import optuna
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import pipeline_common as pc
import eventanchor_common as eac
import pandas as pd

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--seed',        type=int,   default=42)
parser.add_argument('--horizon',     type=int,   default=28)
parser.add_argument('--threshold',   type=float, default=0.5,
                    help='調校目標 ANC threshold: 0.5=<500 (default), 1.0=<1000, 1.5=<1500')
parser.add_argument('--n-trials',    type=int,   default=150)
parser.add_argument('--n-folds',     type=int,   default=5)
parser.add_argument('--optuna-seed', type=int,   default=2025)
parser.add_argument('--max-obs-days', type=int,  default=28,
                    help='與主 pipeline 一致 (obs28); 舊版未設定 = 不限觀察窗')
args = parser.parse_args()

SEED, HORIZON, THR = args.seed, args.horizon, args.threshold
THR_LABEL = {0.5: '<500 (severe)', 1.0: '<1000 (moderate)', 1.5: '<1500 (mild)'}
OUT_JSON = os.path.join(_HERE,
    f'best_params_ea_noleukadj_obs{args.max_obs_days}_thr{THR}_seed{SEED}.json')

print("=" * 72)
print(f"LightGBM tuning (event-anchored) | threshold {THR_LABEL.get(THR, THR)}")
print(f"seed={SEED} | h{HORIZON} block noleukadj | "
      f"n_trials={args.n_trials} | folds={args.n_folds}")
print("=" * 72)

# ---------------------------------------------------------------------------
# 1. Load data + identical split to mlp_vs_lgb_eventanchored.py
# ---------------------------------------------------------------------------
df       = pc.load_lab_df(exclude_heme_mode='leukemia_only_adj')
chemo_df = pc.load_chemo_filtered()
cm_info  = eac.build_cm_info(pd.read_csv(pc.CANCER_MAP_PATH))

pats = df['Newid'].unique()
tr_idx, te_idx = next(
    GroupShuffleSplit(test_size=0.2, random_state=SEED).split(pats, groups=pats)
)
train_pat = set(pats[tr_idx])
test_pat  = set(pats[te_idx])
print(f"Train patients: {len(train_pat)} | TEST patients (held-out): {len(test_pat)}")

markers   = sorted(df[df['Newid'].isin(train_pat)]['NAME'].dropna().unique().tolist())
top_drugs = pc.select_top_k_drugs(chemo_df, train_pat, pc.TOP_K_DRUGS)

data = eac.build_anchor_dataset(df, chemo_df, cm_info, top_drugs, markers,
                                horizon=HORIZON, gap=10, regimen_days=0,
                                regimen_mode='block', max_obs_days=args.max_obs_days)
fc  = [c for c in data.columns if c not in ('Newid', 'min_win_anc')]
tr  = data['Newid'].isin(train_pat)
te  = data['Newid'].isin(test_pat)

Xtr  = data.loc[tr, fc].values
Xte  = data.loc[te, fc].values
ytr  = (data.loc[tr, 'min_win_anc'].values < THR).astype(int)
yte  = (data.loc[te, 'min_win_anc'].values < THR).astype(int)
gtr  = data.loc[tr, 'Newid'].values   # groups for GroupKFold

n_pos = int(ytr.sum()); n_neg = len(ytr) - n_pos
print(f"Train anchors: {len(ytr)} | pos={n_pos} ({n_pos/len(ytr):.1%}) | "
      f"TEST anchors: {len(yte)} | pos={int(yte.sum())}")

# ---------------------------------------------------------------------------
# 2. Optuna objective: GroupKFold CV mean AUC (train set only)
# ---------------------------------------------------------------------------
BASE_PARAMS = {
    'objective': 'binary', 'metric': 'auc',
    'verbose': -1, 'bagging_freq': 5,
}
DEFAULT_PARAMS = dict(BASE_PARAMS)
DEFAULT_PARAMS.update({
    'learning_rate': 0.05, 'num_leaves': 31,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8,
})

gkf   = GroupKFold(n_splits=args.n_folds)
FOLDS = list(gkf.split(Xtr, ytr, groups=gtr))


def cv_auc(params):
    """GroupKFold CV mean AUC on TRAIN only (所有參數組共用同一組 FOLDS)。"""
    params = dict(params)
    fold_aucs = []
    for fold_tr, fold_va in FOLDS:
        yf_tr = ytr[fold_tr]
        n_p = int(yf_tr.sum()); n_n = len(yf_tr) - n_p
        if n_p < 5 or n_n < 5:
            continue
        p = dict(params)
        p['scale_pos_weight'] = n_n / n_p
        dtr_f = lgb.Dataset(Xtr[fold_tr], label=yf_tr)
        dva_f = lgb.Dataset(Xtr[fold_va], label=ytr[fold_va], reference=dtr_f)
        mdl   = lgb.train(p, dtr_f, num_boost_round=1000, valid_sets=[dva_f],
                          callbacks=[lgb.early_stopping(30, verbose=False)])
        pred  = mdl.predict(Xtr[fold_va], num_iteration=mdl.best_iteration)
        yva   = ytr[fold_va]
        if yva.sum() >= 2 and (1 - yva).sum() >= 2:
            fold_aucs.append(roc_auc_score(yva, pred))
    return float(np.mean(fold_aucs)) if fold_aucs else 0.5


def objective(trial):
    params = dict(BASE_PARAMS)
    params.update({
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.2,  log=True),
        'num_leaves':        trial.suggest_int(  'num_leaves',    15,   255),
        'max_depth':         trial.suggest_int(  'max_depth',     3,    12),
        'min_child_samples': trial.suggest_int(  'min_child_samples', 5, 200),
        'feature_fraction':  trial.suggest_float('feature_fraction',  0.4, 1.0),
        'bagging_fraction':  trial.suggest_float('bagging_fraction',  0.4, 1.0),
        'lambda_l1':         trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True),
        'lambda_l2':         trial.suggest_float('lambda_l2', 1e-3, 10.0, log=True),
        'min_split_gain':    trial.suggest_float('min_split_gain', 0.0, 0.5),
    })
    return cv_auc(params)


print(f"\nOptuna 搜尋 ({args.n_trials} trials, TPE seed={args.optuna_seed})...")
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=args.optuna_seed),
)
study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

cv_best_auc = study.best_value
best_params  = dict(BASE_PARAMS)
best_params.update(study.best_params)

default_cv_auc = cv_auc(DEFAULT_PARAMS)
print(f"\nCV-best AUC (tuned): {cv_best_auc:.4f}  (帶 selection optimism)")
print(f"Default CV AUC     : {default_cv_auc:.4f}  (同一組 folds)")
print(f"CV gain (tuned − default) = {cv_best_auc - default_cv_auc:+.4f}   ← 決策依據 (TRAIN only)")
print(f"Best params: {json.dumps(study.best_params, indent=2)}")

# ---------------------------------------------------------------------------
# 3. 一次性 held-out TEST 評估 (train on full train set, evaluate on TEST)
# ---------------------------------------------------------------------------
def eval_test(params):
    p = dict(params)
    p['scale_pos_weight'] = n_neg / n_pos
    cut = int(0.85 * len(Xtr))
    dtr = lgb.Dataset(Xtr[:cut], label=ytr[:cut])
    dva = lgb.Dataset(Xtr[cut:], label=ytr[cut:], reference=dtr)
    mdl = lgb.train(p, dtr, num_boost_round=1000, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(30, verbose=False)])
    pred = mdl.predict(Xte, num_iteration=mdl.best_iteration)
    return roc_auc_score(yte, pred)


tuned_test_auc   = eval_test(best_params)
default_test_auc = eval_test(DEFAULT_PARAMS)

print(f"\n{'='*72}")
print(f"=== 誠實對比 | {THR_LABEL.get(THR, THR)} | "
      f"TEST n={len(yte)} pos={int(yte.sum())} ===")
print(f"{'='*72}")
print(f"  Default params  → TEST AUC = {default_test_auc:.4f}   (現有 frozen 設定)")
print(f"  Tuned   params  → TEST AUC = {tuned_test_auc:.4f}")
print(f"  Tuning net gain            = {tuned_test_auc - default_test_auc:+.4f}")
print(f"  CV-best AUC                = {cv_best_auc:.4f}")
print(f"  Default CV AUC             = {default_cv_auc:.4f}")
print(f"  CV gain (decision basis)   = {cv_best_auc - default_cv_auc:+.4f}")
print(f"  Selection gap (CV−TEST)    = {cv_best_auc - tuned_test_auc:+.4f}  "
      f"(>0 = CV 比 test 樂觀)")

# ---------------------------------------------------------------------------
# 4. Save results
# ---------------------------------------------------------------------------
result = {
    'pipeline':          'event_anchored_block_noleukadj',
    'threshold':         THR,
    'split_seed':        SEED,
    'horizon':           HORIZON,
    'n_trials':          args.n_trials,
    'n_folds':           args.n_folds,
    'max_obs_days':      args.max_obs_days,
    'cv_best_auc':       cv_best_auc,
    'cv_default_auc':    default_cv_auc,
    'cv_gain':           cv_best_auc - default_cv_auc,
    'test_auc_tuned':    tuned_test_auc,
    'test_auc_default':  default_test_auc,
    'tuning_net_gain':   tuned_test_auc - default_test_auc,
    'selection_gap':     cv_best_auc - tuned_test_auc,
    'best_params':       best_params,
}
with open(OUT_JSON, 'w') as f:
    json.dump(result, f, indent=2)
print(f"\n✓ 結果存到 {OUT_JSON}")
