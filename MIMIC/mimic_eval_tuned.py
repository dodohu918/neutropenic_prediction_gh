"""
mimic_eval_tuned.py — 比較 Default LightGBM vs Tuned LightGBM vs Tuned XGBoost
                       在台灣內部測試集 + MIMIC 外部驗證集的表現

設計：
  - 同一份 TW train/test split (GroupShuffleSplit, 3 seeds)
  - 三個模型使用完全相同的 feature / cohort / split
  - Tuned params 從 tune_all_ea_thr0.5_seed42.json 載入
    （以 seed=42 threshold=0.5 調出的參數，套用到所有 seeds/thresholds）
  - 聚焦 h=28 (primary)；threshold 全報 (1.5/1.0/0.5)

用法 (從 MIMIC/ 執行):
  python mimic_eval_tuned.py
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import pipeline_common as pc
import eventanchor_common as eac
import mimic_gcsf as mg

# ── 設定 ──────────────────────────────────────────────────────────────────────
HORIZON      = 28
GAP          = 10
REGIMEN_MODE = 'block'
BLOCK_GAP    = 36.0
MAX_BLOCK    = 7
HEME_MODE    = 'leukemia_only_adj'
SEEDS        = [42, 43, 44]
THRESHOLDS   = [1.5, 1.0, 0.5]
LABELS       = {1.5: '<1500 (mild)', 1.0: '<1000 (moderate)', 0.5: '<500 (severe)'}
MIMIC_DIR    = os.path.join(_HERE, 'mimic_data')

# ── Tuned params（從 tune_all_ea_thr0.5_seed42.json 載入）─────────────────────
TUNE_JSON = os.path.join(
    os.path.dirname(_HERE), 'CCH',
    'tune_all_ea_thr0.5_seed42.json'
)
with open(TUNE_JSON) as f:
    tune_data = json.load(f)

LGB_TUNED_PARAMS_EXTRA = tune_data['results']['LightGBM']['best_params']
XGB_TUNED_PARAMS       = tune_data['results']['XGBoost']['best_params']
XGB_TUNED_PARAMS['n_estimators'] = 1000

LGB_BASE = {'objective': 'binary', 'metric': 'auc', 'verbose': -1, 'bagging_freq': 5}
LGB_DEFAULT_EXTRA = {
    'learning_rate': 0.05, 'num_leaves': 31,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8,
}

print("=" * 72)
print(f"Default LGB vs Tuned LGB vs Tuned XGB | MIMIC external validation")
print(f"h={HORIZON} | block | leukemia_only_adj | seeds={SEEDS}")
print(f"Tuned params loaded from: {os.path.basename(TUNE_JSON)}")
print("=" * 72)

# ── 載入 MIMIC 資料 ────────────────────────────────────────────────────────────
print("\n載入 MIMIC 資料...")
mimic_lab = pd.read_csv(os.path.join(MIMIC_DIR, 'mimic_lab.csv'))
mimic_lab['SIGNIN'] = pd.to_datetime(mimic_lab['SIGNIN'], errors='coerce')
mimic_lab['RESULT'] = pd.to_numeric(mimic_lab['RESULT'], errors='coerce')
mimic_lab = mimic_lab.sort_values(['Newid', 'SIGNIN']).copy()
gcsf = mg.load_gcsf(MIMIC_DIR)

_mc = pd.read_csv(os.path.join(MIMIC_DIR, 'mimic_chemo.csv'))
_mc['time']  = pd.to_datetime(_mc['time'], errors='coerce')
_mc = _mc.dropna(subset=['Newid', 'time']).copy()
_mc['canon'] = _mc['item'].apply(pc.canonicalize_drug)
_conv = _mc.apply(
    lambda r: pc.convert_dose_to_mg(r['item'], r['dose'], r['unit']),
    axis=1, result_type='expand')
_mc['dose_mg'] = _conv[0].astype(float)
_mc['admin_cnt'] = _conv[1].astype(int)
mimic_chemo = (_mc[_mc['canon'] != 'GCSF']
               [['Newid', 'time', 'canon', 'dose_mg', 'admin_cnt']]
               .sort_values(['Newid', 'time']).reset_index(drop=True))
mimic_cm_info = eac.build_cm_info(pd.read_csv(os.path.join(MIMIC_DIR, 'mimic_patients.csv')))
print(f"  MIMIC 病人: {mimic_lab['Newid'].nunique()} | 化療筆: {len(mimic_chemo):,}")

# ── 載入台灣資料 ───────────────────────────────────────────────────────────────
print("載入台灣資料...")
tw_df      = pc.load_lab_df(exclude_heme_mode=HEME_MODE)
tw_chemo   = pc.load_chemo_filtered(drop_gcsf=True)
tw_cm_info = eac.build_cm_info(pd.read_csv(pc.CANCER_MAP_PATH))
print(f"  台灣病人: {tw_df['Newid'].nunique()}")


# ── 輔助：訓練 LightGBM ────────────────────────────────────────────────────────
def train_lgb(extra_params, Xtr, ytr, cut):
    p = dict(LGB_BASE); p.update(extra_params)
    n_pos = int(ytr.sum()); n_neg = len(ytr) - n_pos
    p['scale_pos_weight'] = n_neg / n_pos
    dtr = lgb.Dataset(Xtr[:cut], label=ytr[:cut])
    dva = lgb.Dataset(Xtr[cut:], label=ytr[cut:], reference=dtr)
    m = lgb.train(p, dtr, num_boost_round=1000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
    return m


# ── 主迴圈 ─────────────────────────────────────────────────────────────────────
# results[thr][model] = {'tw': [], 'mimic': []}
MODELS = ['Default LGB', 'Tuned LGB', 'Tuned XGB']
results = {
    thr: {m: {'tw': [], 'mimic': []} for m in MODELS}
    for thr in THRESHOLDS
}

for seed in SEEDS:
    print(f"\n{'='*72}")
    print(f"Seed {seed}")
    print(f"{'='*72}")

    # Train/test split
    uniq = tw_df['Newid'].unique()
    tr_idx, te_idx = next(
        GroupShuffleSplit(test_size=0.2, random_state=seed).split(uniq, groups=uniq)
    )
    train_pts = set(uniq[tr_idx]); test_pts = set(uniq[te_idx])

    # Feature schema from training patients
    expected_markers = sorted(
        tw_df[tw_df['Newid'].isin(train_pts)]['NAME'].dropna().unique().tolist())
    top_drugs = pc.select_top_k_drugs(tw_chemo, train_pts, pc.TOP_K_DRUGS)

    # Build TW anchor dataset
    print("  建台灣 anchor dataset...", flush=True)
    tw_data = eac.build_anchor_dataset(
        tw_df, tw_chemo, tw_cm_info, top_drugs, expected_markers,
        horizon=HORIZON, gap=GAP, regimen_days=0,
        regimen_mode=REGIMEN_MODE, block_gap_hours=BLOCK_GAP,
        max_block_days=MAX_BLOCK, verbose=False)
    feat_cols = [c for c in tw_data.columns if c not in ('Newid', 'min_win_anc')]
    tw_tr  = tw_data[tw_data['Newid'].isin(train_pts)]
    tw_te  = tw_data[tw_data['Newid'].isin(test_pts)]
    Xtr    = tw_tr[feat_cols].values
    Xte    = tw_te[feat_cols].values
    anc_tr = tw_tr['min_win_anc'].values
    anc_te = tw_te['min_win_anc'].values
    cut = int(0.85 * len(Xtr))

    # Build MIMIC anchor dataset
    print("  建 MIMIC anchor dataset...", flush=True)
    mimic_data = eac.build_anchor_dataset(
        mimic_lab, mimic_chemo, mimic_cm_info, top_drugs, expected_markers,
        horizon=HORIZON, gap=GAP, regimen_days=0,
        regimen_mode=REGIMEN_MODE, block_gap_hours=BLOCK_GAP,
        max_block_days=MAX_BLOCK, verbose=False, include_window=True)
    for c in feat_cols:
        if c not in mimic_data.columns:
            mimic_data[c] = 0.0
    X_mimic   = mimic_data[feat_cols].values
    anc_mimic = mimic_data['min_win_anc'].values
    gk = mg.keep_masks(mimic_data, mimic_lab, gcsf, THRESHOLDS); mg.report(gk)
    X_mimic_all, anc_mimic_all = X_mimic, anc_mimic

    print(f"  TW train: {len(Xtr):,} | TW test: {len(Xte):,} | MIMIC: {len(X_mimic):,}")
    print(f"\n  {'Threshold':<22} {'Model':<14} {'TW test AUC':>12} {'MIMIC AUC':>11}")
    print(f"  {'-'*62}")

    for thr in THRESHOLDS:
        X_mimic, anc_mimic = X_mimic_all[gk[thr]], anc_mimic_all[gk[thr]]
        ytr_bin = (anc_tr < thr).astype(int)
        yte_bin = (anc_te < thr).astype(int)
        ymi_bin = (anc_mimic < thr).astype(int)
        if ytr_bin.sum() < 10 or yte_bin.sum() < 2 or ymi_bin.sum() < 2:
            continue
        n_pos = int(ytr_bin.sum()); n_neg = len(ytr_bin) - n_pos

        # Default LightGBM
        m_lgb_def = train_lgb(LGB_DEFAULT_EXTRA, Xtr, ytr_bin, cut)
        p_def_tw  = m_lgb_def.predict(Xte,     num_iteration=m_lgb_def.best_iteration)
        p_def_mi  = m_lgb_def.predict(X_mimic, num_iteration=m_lgb_def.best_iteration)
        auc_def_tw = roc_auc_score(yte_bin, p_def_tw)
        auc_def_mi = roc_auc_score(ymi_bin, p_def_mi)
        results[thr]['Default LGB']['tw'].append(auc_def_tw)
        results[thr]['Default LGB']['mimic'].append(auc_def_mi)

        # Tuned LightGBM
        m_lgb_tun = train_lgb(LGB_TUNED_PARAMS_EXTRA, Xtr, ytr_bin, cut)
        p_tun_tw  = m_lgb_tun.predict(Xte,     num_iteration=m_lgb_tun.best_iteration)
        p_tun_mi  = m_lgb_tun.predict(X_mimic, num_iteration=m_lgb_tun.best_iteration)
        auc_tun_tw = roc_auc_score(yte_bin, p_tun_tw)
        auc_tun_mi = roc_auc_score(ymi_bin, p_tun_mi)
        results[thr]['Tuned LGB']['tw'].append(auc_tun_tw)
        results[thr]['Tuned LGB']['mimic'].append(auc_tun_mi)

        # Tuned XGBoost
        m_xgb = xgb.XGBClassifier(
            **XGB_TUNED_PARAMS,
            scale_pos_weight=n_neg / n_pos,
            eval_metric='auc', early_stopping_rounds=30,
            random_state=seed, verbosity=0,
        )
        m_xgb.fit(Xtr[:cut], ytr_bin[:cut],
                  eval_set=[(Xtr[cut:], ytr_bin[cut:])], verbose=False)
        p_xgb_tw = m_xgb.predict_proba(Xte)[:, 1]
        p_xgb_mi = m_xgb.predict_proba(X_mimic)[:, 1]
        auc_xgb_tw = roc_auc_score(yte_bin, p_xgb_tw)
        auc_xgb_mi = roc_auc_score(ymi_bin, p_xgb_mi)
        results[thr]['Tuned XGB']['tw'].append(auc_xgb_tw)
        results[thr]['Tuned XGB']['mimic'].append(auc_xgb_mi)

        lbl = LABELS[thr]
        print(f"  {lbl:<22} {'Default LGB':<14} {auc_def_tw:>12.3f} {auc_def_mi:>11.3f}")
        print(f"  {'':<22} {'Tuned LGB':<14} {auc_tun_tw:>12.3f} {auc_tun_mi:>11.3f}")
        print(f"  {'':<22} {'Tuned XGB':<14} {auc_xgb_tw:>12.3f} {auc_xgb_mi:>11.3f}")
        print(f"  {'':<22} {'-'*38}")


# ── 摘要表 (3-seed mean ± SD) ──────────────────────────────────────────────────
print(f"\n{'='*72}")
print("=== 3-seed 摘要 (mean ± SD) ===")
print(f"{'='*72}")
print(f"\n{'Threshold':<22} {'Model':<14} {'TW test AUC':>20} {'MIMIC AUC':>20} {'ΔAUROC':>8}")
print(f"{'-'*87}")
for thr in THRESHOLDS:
    lbl = LABELS[thr]
    for model in MODELS:
        tw_v  = results[thr][model]['tw']
        mi_v  = results[thr][model]['mimic']
        tw_s  = f"{np.mean(tw_v):.3f} ± {np.std(tw_v):.3f}" if tw_v else "N/A"
        mi_s  = f"{np.mean(mi_v):.3f} ± {np.std(mi_v):.3f}" if mi_v else "N/A"
        delta = f"{np.mean(mi_v) - np.mean(tw_v):+.3f}" if (tw_v and mi_v) else "N/A"
        row_lbl = lbl if model == MODELS[0] else ''
        print(f"{row_lbl:<22} {model:<14} {tw_s:>20} {mi_s:>20} {delta:>8}")
    print(f"{'-'*87}")

print("""
Notes:
  - Tuned params: optimised on seed=42 / threshold=<500; applied to all seeds/thresholds
  - MIMIC dose features (init_{{drug}}_dose, cum_dose) all zero — unavailable in emar
  - ΔAUROC = MIMIC − TW test (negative = generalisation drop)
""")
