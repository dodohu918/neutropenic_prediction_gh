"""
shap_analysis.py — SHAP feature importance for three models on the final primary pipeline

  A. LightGBM  (main model)    — beeswarm + bar plot
  B. XGBoost   (cross-check)   — bar plot
  C. LR        (linear ref)    — bar plot
  D. Comparison                — top-N from LightGBM shown across all three, normalised

Pipeline: leukemia_only_adj + block regimen mode (final primary, same as mlp_vs_lgb_eventanchored.py)

Usage (from CCH/):
  python shap_analysis.py
  python shap_analysis.py --threshold 1.0 --top-n 15

Output files (same folder as this script):
  fig_shap_lgb_beeswarm.png  — LGB beeswarm (direction + magnitude per sample)
  fig_shap_lgb_bar.png       — LGB mean |SHAP| bar
  fig_shap_xgb_bar.png       — XGB mean |SHAP| bar
  fig_shap_lr_bar.png        — LR  mean |SHAP| bar
  fig_shap_comparison.png    — top-N normalised comparison across all three
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import pipeline_common as pc
import eventanchor_common as eac

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--seed',      type=int,   default=42)
parser.add_argument('--horizon',   type=int,   default=28)
parser.add_argument('--threshold', type=float, default=0.5,
                    help='ANC threshold: 0.5=<500 severe (default), 1.0=<1000, 1.5=<1500')
parser.add_argument('--top-n',     type=int,   default=20,
                    help='Number of features to display in plots')
parser.add_argument('--max-obs-days', type=int, default=28,
                    help='觀察窗 (天); 預設 28, 與主分析一致')
args = parser.parse_args()
SEED, HORIZON, THR, TOP_N = args.seed, args.horizon, args.threshold, args.top_n
MAX_OBS_DAYS = args.max_obs_days

THR_LABEL = {0.5: '<500 (severe)', 1.0: '<1000 (moderate)', 1.5: '<1500 (mild)'}
thr_str = THR_LABEL.get(THR, f'<{THR*1000:.0f}')

# Human-readable labels for the most common features
LABEL_MAP = {
    'hist_min_anc':          'Min prior ANC',
    'anc_last':              'Most recent ANC',
    'days_since_prev_cycle': 'Days since prev cycle',
    'n_prior_lt500':         '# prior ANC <500',
    'n_prior_lt1000':        '# prior ANC <1000',
    'n_prior_lt1500':        '# prior ANC <1500',
    'anc_slope_per_day':     'ANC slope (per day)',
    'anc_last2':             '2nd most recent ANC',
    'anc_last_daysago':      'Days since last ANC',
    'anc_last_mask':         'ANC available (mask)',
    'n_prior_anc':           '# prior ANC measurements',
    'cycle_number':          'Cycle number',
    'cum_n_admin':           'Cumulative admin count',
    'cum_dose':              'Cumulative dose (mg)',
    'init_n_admin':          'Init admin count (block)',
    'init_n_distinct_drugs': 'Init distinct drugs',
    'days_since_dx':         'Days since diagnosis',
    'age_at_dx':             'Age at diagnosis',
    'sex':                   'Sex',
    'cancer_cat':            'Cancer category',
}

print(f"SHAP analysis | {thr_str} | h{HORIZON} block noleukadj seed{SEED} | top-{TOP_N}")

# ---------------------------------------------------------------------------
# Data + split (identical to mlp_vs_lgb_eventanchored.py)
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
                                regimen_mode='block', max_obs_days=MAX_OBS_DAYS)
fc  = [c for c in data.columns if c not in ('Newid', 'min_win_anc')]
tr  = data['Newid'].isin(train_pat)
te  = data['Newid'].isin(test_pat)

Xtr = data.loc[tr, fc].values
Xte = data.loc[te, fc].values
ytr = (data.loc[tr, 'min_win_anc'].values < THR).astype(int)
yte = (data.loc[te, 'min_win_anc'].values < THR).astype(int)

n_pos = int(ytr.sum())
n_neg = len(ytr) - n_pos
spw   = n_neg / n_pos

sc     = StandardScaler().fit(Xtr)
Xtr_s  = sc.transform(Xtr)
Xte_s  = sc.transform(Xte)

feat_labels = [LABEL_MAP.get(f, f) for f in fc]

print(f"  train n={len(ytr)} pos={n_pos}  |  test n={len(yte)} pos={int(yte.sum())}")

# ---------------------------------------------------------------------------
# Train all three models
# ---------------------------------------------------------------------------
print("Training models...")
cut = int(0.85 * len(Xtr))

# LightGBM
LGB_PARAMS = {
    'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05,
    'num_leaves': 31, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
    'bagging_freq': 5, 'verbose': -1, 'scale_pos_weight': spw,
}
dtr     = lgb.Dataset(Xtr[:cut], label=ytr[:cut])
dva     = lgb.Dataset(Xtr[cut:], label=ytr[cut:], reference=dtr)
lgb_mdl = lgb.train(LGB_PARAMS, dtr, num_boost_round=1000, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(30, verbose=False)])

# XGBoost
xgb_mdl = xgb.XGBClassifier(
    n_estimators=1000, learning_rate=0.05, max_depth=6,
    scale_pos_weight=spw, eval_metric='auc', early_stopping_rounds=30,
    random_state=SEED, verbosity=0,
)
xgb_mdl.fit(Xtr[:cut], ytr[:cut],
             eval_set=[(Xtr[cut:], ytr[cut:])], verbose=False)

# LR (scaled features)
lr_mdl = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000,
                             solver='lbfgs', random_state=SEED)
lr_mdl.fit(Xtr_s, ytr)

print("Done. Computing SHAP values...\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _positive_class(sv):
    """Return SHAP values for the positive class (handles list or array)."""
    return sv[1] if isinstance(sv, list) else sv


def bar_fig(mean_abs, title, fname, color, top_n=TOP_N):
    """Horizontal bar chart of mean |SHAP|, top-N features."""
    idx = np.argsort(mean_abs)[::-1][:top_n][::-1]   # ascending for barh
    fig, ax = plt.subplots(figsize=(8, top_n * 0.35 + 1.5))
    ax.barh([feat_labels[i] for i in idx], mean_abs[idx], color=color)
    ax.set_xlabel('Mean |SHAP value|  (impact on log-odds output)')
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    path = os.path.join(_HERE, fname)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}")
    return mean_abs


def console_table(mean_abs, model_name, top_n=TOP_N):
    top_idx = np.argsort(mean_abs)[::-1][:top_n]
    print(f"  Top {top_n} features — {model_name}:")
    for rank, i in enumerate(top_idx, 1):
        print(f"    {rank:>2}. {feat_labels[i]:<40} {mean_abs[i]:.4f}")


# ---------------------------------------------------------------------------
# A. LightGBM  (main model)
# ---------------------------------------------------------------------------
print("=" * 60)
print("A. LightGBM  (main)")
print("=" * 60)

expl_lgb   = shap.TreeExplainer(lgb_mdl)
sv_lgb     = _positive_class(expl_lgb.shap_values(Xte))
ma_lgb     = np.abs(sv_lgb).mean(axis=0)
top_idx_lgb = np.argsort(ma_lgb)[::-1][:TOP_N]

# Beeswarm — shows direction (positive/negative) and spread across all test samples
Xte_top     = pd.DataFrame(Xte[:, top_idx_lgb],
                            columns=[feat_labels[i] for i in top_idx_lgb])
shap.summary_plot(sv_lgb[:, top_idx_lgb], Xte_top, show=False)
fig_bee = plt.gcf()
fig_bee.suptitle(f'LightGBM SHAP  —  {thr_str}  (seed {SEED})', y=1.01, fontsize=10)
plt.tight_layout()
bee_path = os.path.join(_HERE, 'fig_shap_lgb_beeswarm.png')
fig_bee.savefig(bee_path, dpi=150, bbox_inches='tight')
plt.close(fig_bee)
print("  Saved fig_shap_lgb_beeswarm.png")

# Bar
bar_fig(ma_lgb,
        f'LightGBM — mean |SHAP|  ({thr_str})',
        'fig_shap_lgb_bar.png', color='steelblue')

console_table(ma_lgb, 'LightGBM')

# ---------------------------------------------------------------------------
# B. XGBoost  (cross-check)
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("B. XGBoost  (cross-check)")
print("=" * 60)

expl_xgb = shap.TreeExplainer(xgb_mdl)
sv_xgb   = _positive_class(expl_xgb.shap_values(Xte))
ma_xgb   = np.abs(sv_xgb).mean(axis=0)

bar_fig(ma_xgb,
        f'XGBoost — mean |SHAP|  ({thr_str})',
        'fig_shap_xgb_bar.png', color='darkorange')

console_table(ma_xgb, 'XGBoost')

# ---------------------------------------------------------------------------
# C. LR  (linear reference)
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("C. LR  (linear reference)")
print("=" * 60)

# interventional perturbation — fastest and standard for linear models
expl_lr = shap.LinearExplainer(lr_mdl, Xtr_s)
sv_lr   = _positive_class(expl_lr.shap_values(Xte_s))
ma_lr   = np.abs(sv_lr).mean(axis=0)

bar_fig(ma_lr,
        f'LR — mean |SHAP|  ({thr_str})',
        'fig_shap_lr_bar.png', color='seagreen')

console_table(ma_lr, 'LR')

# ---------------------------------------------------------------------------
# D. Comparison — top-N from LightGBM, normalised, across all three
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("D. Comparison plot (top features from LightGBM)")
print("=" * 60)

TOP_COMP = min(15, TOP_N)
comp_idx = np.argsort(ma_lgb)[::-1][:TOP_COMP]

# Normalise each model's mean |SHAP| to sum-to-1 so bar heights are
# comparable across models (they live in different log-odds scales)
def normalise(arr):
    s = arr.sum()
    return arr / s if s > 0 else arr

n_lgb = normalise(ma_lgb)
n_xgb = normalise(ma_xgb)
n_lr  = normalise(ma_lr)

labels_comp = [feat_labels[i] for i in comp_idx[::-1]]   # ascending for barh
y = np.arange(len(labels_comp))
w = 0.25

fig, ax = plt.subplots(figsize=(9, TOP_COMP * 0.5 + 2.5))
ax.barh(y - w, [n_lgb[i] for i in comp_idx[::-1]], w,
        label='LightGBM', color='steelblue')
ax.barh(y,     [n_xgb[i] for i in comp_idx[::-1]], w,
        label='XGBoost',  color='darkorange')
ax.barh(y + w, [n_lr[i]  for i in comp_idx[::-1]], w,
        label='LR',       color='seagreen')
ax.set_yticks(y)
ax.set_yticklabels(labels_comp, fontsize=9)
ax.set_xlabel('Normalised mean |SHAP|  (relative importance within each model)')
ax.set_title(f'Feature importance comparison — {thr_str}  (top {TOP_COMP} by LightGBM, seed {SEED})',
             fontsize=10)
ax.legend(fontsize=9)
plt.tight_layout()
comp_path = os.path.join(_HERE, 'fig_shap_comparison.png')
fig.savefig(comp_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved fig_shap_comparison.png")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print("Output files:")
for fname in ['fig_shap_lgb_beeswarm.png', 'fig_shap_lgb_bar.png',
              'fig_shap_xgb_bar.png', 'fig_shap_lr_bar.png',
              'fig_shap_comparison.png']:
    path = os.path.join(_HERE, fname)
    kb   = os.path.getsize(path) // 1024 if os.path.exists(path) else 0
    print(f"  {fname:<35} {kb} KB")
