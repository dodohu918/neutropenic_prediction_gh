"""
tune_all_eventanchored.py — 統一調校 LightGBM / XGBoost / LR，輸出 before/after 對照表

方法學:
  - 同一份 GroupShuffleSplit(seed) train/TEST split (與 mlp_vs_lgb_eventanchored.py 一致)
  - Optuna TPE (= Bayesian optimization via Tree-structured Parzen Estimator)
  - GroupKFold(5) on Newid within train — 病人不跨 fold 洩漏
  - TEST 全程不碰，只在最後對每個 model 評估一次
  - LR 在每個 fold 內獨立 fit StandardScaler (防止 train→val leakage)
  - 報: Default TEST AUC / Tuned TEST AUC / Net gain / CV-best / Selection gap

用法 (從 CCH/ 執行):
  python tune_all_eventanchored.py
  python tune_all_eventanchored.py --n-trials 150 --threshold 0.5
"""
import os
import sys
import json
import argparse
import warnings
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import pipeline_common as pc
import eventanchor_common as eac

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--seed',        type=int,   default=42)
parser.add_argument('--horizon',     type=int,   default=28)
parser.add_argument('--threshold',   type=float, default=0.5,
                    help='ANC threshold: 0.5=<500 severe (default), 1.0=<1000, 1.5=<1500')
parser.add_argument('--n-trials',    type=int,   default=150,
                    help='Optuna trials per model (LR uses half since search space is smaller)')
parser.add_argument('--n-folds',     type=int,   default=5)
parser.add_argument('--optuna-seed', type=int,   default=2025)
args = parser.parse_args()

SEED, HORIZON, THR = args.seed, args.horizon, args.threshold
N_FOLDS = args.n_folds
THR_LABEL = {0.5: '<500 (severe)', 1.0: '<1000 (moderate)', 1.5: '<1500 (mild)'}
thr_str = THR_LABEL.get(THR, f'<{THR*1000:.0f}')

print("=" * 72)
print(f"Tuning LGB / XGB / LR | {thr_str} | h{HORIZON} block noleukadj seed{SEED}")
print(f"n_trials={args.n_trials} | folds={N_FOLDS} | optuna_seed={args.optuna_seed}")
print("=" * 72)

# ---------------------------------------------------------------------------
# 1. Data (identical split to mlp_vs_lgb_eventanchored.py)
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

markers   = sorted(df[df['Newid'].isin(train_pat)]['NAME'].dropna().unique().tolist())
top_drugs = pc.select_top_k_drugs(chemo_df, train_pat, pc.TOP_K_DRUGS)

data = eac.build_anchor_dataset(df, chemo_df, cm_info, top_drugs, markers,
                                horizon=HORIZON, gap=10, regimen_days=0,
                                regimen_mode='block')
fc  = [c for c in data.columns if c not in ('Newid', 'min_win_anc')]
tr  = data['Newid'].isin(train_pat)
te  = data['Newid'].isin(test_pat)

Xtr  = data.loc[tr, fc].values
Xte  = data.loc[te, fc].values
ytr  = (data.loc[tr, 'min_win_anc'].values < THR).astype(int)
yte  = (data.loc[te, 'min_win_anc'].values < THR).astype(int)
gtr  = data.loc[tr, 'Newid'].values

n_pos = int(ytr.sum()); n_neg = len(ytr) - n_pos; spw = n_neg / n_pos

# Scaled features for LR (fit on full train; per-fold scaling in CV objective)
sc    = StandardScaler().fit(Xtr)
Xtr_s = sc.transform(Xtr)
Xte_s = sc.transform(Xte)

print(f"Train: {len(ytr)} anchors, pos={n_pos} ({n_pos/len(ytr):.1%})")
print(f"TEST:  {len(yte)} anchors, pos={int(yte.sum())} ({yte.mean():.1%})\n")

FOLDS = list(GroupKFold(n_splits=N_FOLDS).split(Xtr, ytr, groups=gtr))

# ---------------------------------------------------------------------------
# 2. Default TEST AUC for all three (identical to mlp_vs_lgb_eventanchored.py)
# ---------------------------------------------------------------------------
def train_lgb(params, Xtr_, ytr_, Xval, yval):
    p = dict(params); p['scale_pos_weight'] = (len(ytr_) - ytr_.sum()) / ytr_.sum()
    dtr = lgb.Dataset(Xtr_, label=ytr_)
    dva = lgb.Dataset(Xval, label=yval, reference=dtr)
    m = lgb.train(p, dtr, num_boost_round=1000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
    return m

def train_xgb(params, Xtr_, ytr_, Xval, yval):
    spw_ = (len(ytr_) - ytr_.sum()) / ytr_.sum()
    m = xgb.XGBClassifier(**params, scale_pos_weight=spw_,
                           eval_metric='auc', early_stopping_rounds=30,
                           random_state=SEED, verbosity=0)
    m.fit(Xtr_, ytr_, eval_set=[(Xval, yval)], verbose=False)
    return m

cut = int(0.85 * len(Xtr))

LGB_DEFAULT = {'objective': 'binary', 'metric': 'auc', 'verbose': -1,
               'bagging_freq': 5, 'learning_rate': 0.05, 'num_leaves': 31,
               'feature_fraction': 0.8, 'bagging_fraction': 0.8}
XGB_DEFAULT = {'n_estimators': 1000, 'learning_rate': 0.05, 'max_depth': 6}
LR_DEFAULT  = {'C': 1.0, 'class_weight': 'balanced', 'max_iter': 1000,
               'solver': 'lbfgs', 'random_state': SEED}

lgb_def = train_lgb(LGB_DEFAULT, Xtr[:cut], ytr[:cut], Xtr[cut:], ytr[cut:])
xgb_def = train_xgb(XGB_DEFAULT, Xtr[:cut], ytr[:cut], Xtr[cut:], ytr[cut:])
lr_def  = LogisticRegression(**LR_DEFAULT).fit(Xtr_s, ytr)

default_aucs = {
    'LightGBM': roc_auc_score(yte, lgb_def.predict(Xte)),
    'XGBoost':  roc_auc_score(yte, xgb_def.predict_proba(Xte)[:, 1]),
    'LR':       roc_auc_score(yte, lr_def.predict_proba(Xte_s)[:, 1]),
}
print("Default TEST AUC:", {k: f"{v:.4f}" for k, v in default_aucs.items()})

# ---------------------------------------------------------------------------
# 3. Optuna tuning — LightGBM
# ---------------------------------------------------------------------------
print(f"\n[1/3] Tuning LightGBM ({args.n_trials} trials)...")
LGB_BASE = {'objective': 'binary', 'metric': 'auc', 'verbose': -1, 'bagging_freq': 5}

def lgb_objective(trial):
    p = dict(LGB_BASE)
    p.update({
        'learning_rate':     trial.suggest_float('learning_rate',     0.01, 0.2,  log=True),
        'num_leaves':        trial.suggest_int(  'num_leaves',        15,   255),
        'max_depth':         trial.suggest_int(  'max_depth',         3,    12),
        'min_child_samples': trial.suggest_int(  'min_child_samples', 5,    200),
        'feature_fraction':  trial.suggest_float('feature_fraction',  0.4,  1.0),
        'bagging_fraction':  trial.suggest_float('bagging_fraction',  0.4,  1.0),
        'lambda_l1':         trial.suggest_float('lambda_l1',         1e-3, 10.0, log=True),
        'lambda_l2':         trial.suggest_float('lambda_l2',         1e-3, 10.0, log=True),
        'min_split_gain':    trial.suggest_float('min_split_gain',    0.0,  0.5),
    })
    aucs = []
    for ftr, fva in FOLDS:
        yf = ytr[ftr]
        p['scale_pos_weight'] = (len(yf) - yf.sum()) / max(yf.sum(), 1)
        dtr_ = lgb.Dataset(Xtr[ftr], label=yf)
        dva_ = lgb.Dataset(Xtr[fva], label=ytr[fva], reference=dtr_)
        m = lgb.train(p, dtr_, num_boost_round=1000, valid_sets=[dva_],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
        pred = m.predict(Xtr[fva], num_iteration=m.best_iteration)
        yva = ytr[fva]
        if yva.sum() >= 2 and (1 - yva).sum() >= 2:
            aucs.append(roc_auc_score(yva, pred))
    return float(np.mean(aucs)) if aucs else 0.5

study_lgb = optuna.create_study(direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=args.optuna_seed))
study_lgb.optimize(lgb_objective, n_trials=args.n_trials, show_progress_bar=True)

lgb_best = dict(LGB_BASE); lgb_best.update(study_lgb.best_params)
lgb_tuned = train_lgb(lgb_best, Xtr[:cut], ytr[:cut], Xtr[cut:], ytr[cut:])
lgb_tuned_auc = roc_auc_score(yte, lgb_tuned.predict(Xte))

# ---------------------------------------------------------------------------
# 4. Optuna tuning — XGBoost
# ---------------------------------------------------------------------------
print(f"\n[2/3] Tuning XGBoost ({args.n_trials} trials)...")

def xgb_objective(trial):
    params = {
        'n_estimators':    1000,
        'learning_rate':   trial.suggest_float('learning_rate',   0.01, 0.2,  log=True),
        'max_depth':       trial.suggest_int(  'max_depth',       3,    12),
        'min_child_weight':trial.suggest_int(  'min_child_weight',1,    20),
        'subsample':       trial.suggest_float('subsample',       0.4,  1.0),
        'colsample_bytree':trial.suggest_float('colsample_bytree',0.4,  1.0),
        'reg_lambda':      trial.suggest_float('reg_lambda',      1e-3, 10.0, log=True),
        'reg_alpha':       trial.suggest_float('reg_alpha',       1e-3, 10.0, log=True),
        'gamma':           trial.suggest_float('gamma',           0.0,  0.5),
    }
    aucs = []
    for ftr, fva in FOLDS:
        yf  = ytr[ftr]
        spw_ = (len(yf) - yf.sum()) / max(yf.sum(), 1)
        m = xgb.XGBClassifier(**params, scale_pos_weight=spw_,
                               eval_metric='auc', early_stopping_rounds=30,
                               random_state=SEED, verbosity=0)
        m.fit(Xtr[ftr], yf, eval_set=[(Xtr[fva], ytr[fva])], verbose=False)
        pred = m.predict_proba(Xtr[fva])[:, 1]
        yva = ytr[fva]
        if yva.sum() >= 2 and (1 - yva).sum() >= 2:
            aucs.append(roc_auc_score(yva, pred))
    return float(np.mean(aucs)) if aucs else 0.5

study_xgb = optuna.create_study(direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=args.optuna_seed))
study_xgb.optimize(xgb_objective, n_trials=args.n_trials, show_progress_bar=True)

xgb_best_params = dict(study_xgb.best_params)
xgb_best_params['n_estimators'] = 1000
xgb_tuned = train_xgb(xgb_best_params, Xtr[:cut], ytr[:cut], Xtr[cut:], ytr[cut:])
xgb_tuned_auc = roc_auc_score(yte, xgb_tuned.predict_proba(Xte)[:, 1])

# ---------------------------------------------------------------------------
# 5. Optuna tuning — LR
# ---------------------------------------------------------------------------
# LR has only 1 key hyperparameter (C); use half the trials
lr_trials = max(args.n_trials // 2, 50)
print(f"\n[3/3] Tuning LR ({lr_trials} trials — smaller search space)...")

def lr_objective(trial):
    C_ = trial.suggest_float('C', 1e-3, 100.0, log=True)
    aucs = []
    for ftr, fva in FOLDS:
        # Scale within fold to avoid leakage
        sc_ = StandardScaler().fit(Xtr[ftr])
        m   = LogisticRegression(C=C_, class_weight='balanced', max_iter=1000,
                                 solver='lbfgs', random_state=SEED)
        m.fit(sc_.transform(Xtr[ftr]), ytr[ftr])
        pred = m.predict_proba(sc_.transform(Xtr[fva]))[:, 1]
        yva = ytr[fva]
        if yva.sum() >= 2 and (1 - yva).sum() >= 2:
            aucs.append(roc_auc_score(yva, pred))
    return float(np.mean(aucs)) if aucs else 0.5

study_lr = optuna.create_study(direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=args.optuna_seed))
study_lr.optimize(lr_objective, n_trials=lr_trials, show_progress_bar=True)

best_C    = study_lr.best_params['C']
lr_tuned  = LogisticRegression(C=best_C, class_weight='balanced', max_iter=1000,
                                solver='lbfgs', random_state=SEED).fit(Xtr_s, ytr)
lr_tuned_auc = roc_auc_score(yte, lr_tuned.predict_proba(Xte_s)[:, 1])

# ---------------------------------------------------------------------------
# 6. Combined results table
# ---------------------------------------------------------------------------
results = {
    'LightGBM': {
        'default': default_aucs['LightGBM'],
        'tuned':   lgb_tuned_auc,
        'cv_best': study_lgb.best_value,
        'best_params': study_lgb.best_params,
    },
    'XGBoost': {
        'default': default_aucs['XGBoost'],
        'tuned':   xgb_tuned_auc,
        'cv_best': study_xgb.best_value,
        'best_params': study_xgb.best_params,
    },
    'LR': {
        'default': default_aucs['LR'],
        'tuned':   lr_tuned_auc,
        'cv_best': study_lr.best_value,
        'best_params': study_lr.best_params,
    },
}

print(f"\n{'='*72}")
print(f"TUNING RESULTS — {thr_str} | h{HORIZON} block noleukadj seed{SEED}")
print(f"TEST n={len(yte)}, pos={int(yte.sum())} ({yte.mean():.1%})")
print(f"{'='*72}")
print(f"{'Model':<12} {'Default AUC':>13} {'Tuned AUC':>11} {'Net Gain':>10} "
      f"{'CV-best':>9} {'Sel. Gap':>10}")
print(f"{'-'*12} {'-'*13} {'-'*11} {'-'*10} {'-'*9} {'-'*10}")
for model, r in results.items():
    gain = r['tuned'] - r['default']
    gap  = r['cv_best'] - r['tuned']
    print(f"{model:<12} {r['default']:>13.4f} {r['tuned']:>11.4f} {gain:>+10.4f} "
          f"{r['cv_best']:>9.4f} {gap:>+10.4f}")

print(f"\nBest hyperparameters found:")
for model, r in results.items():
    print(f"  {model}: {json.dumps(r['best_params'], indent=None)}")

# Save JSON
out = {
    'pipeline': 'event_anchored_block_noleukadj',
    'threshold': THR, 'seed': SEED, 'horizon': HORIZON,
    'n_trials_lgb_xgb': args.n_trials, 'n_trials_lr': lr_trials,
    'results': {k: {kk: vv for kk, vv in v.items()} for k, v in results.items()},
}
# make JSON serializable
for m in out['results']:
    for kk in out['results'][m]:
        v = out['results'][m][kk]
        if isinstance(v, (np.floating, np.integer)):
            out['results'][m][kk] = float(v)

out_path = os.path.join(_HERE, f'tune_all_ea_thr{THR}_seed{SEED}.json')
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\n✓ Results saved to {os.path.basename(out_path)}")
print("\n註: Selection gap < 0 意味 CV 比 test 保守 (無 overfitting); > 0 表示 CV 過於樂觀。")
