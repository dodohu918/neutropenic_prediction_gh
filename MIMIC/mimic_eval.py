"""
mimic_eval.py — MIMIC-IV 外部驗證（支援多 horizon）

設計：
  台灣資料 (noleukadj, block) 訓練 LightGBM → 直接套用到 MIMIC-IV
  並排輸出台灣內部測試集 AUROC (reference) vs MIMIC AUROC (external validation)
  3 seeds × 3 thresholds × 多 horizon

用法 (從 MIMIC/ 執行):
  python mimic_eval.py                    # 只跑 h28（預設）
  python mimic_eval.py --horizons 7 14 21 28   # 跑全部

注意：MIMIC dose 資料不可用 (emar 無欄位)，所有 init_{drug}_dose / cum_dose = 0。
      init_n_admin / init_n_distinct_drugs 仍正確，模型靠這些計數特徵辨識方案。
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import pipeline_common as pc
import eventanchor_common as eac
import mimic_gcsf as mg

# ─── 設定 ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--horizons', type=int, nargs='+', default=[28],
                    help='要跑的 horizon 列表，例如 --horizons 7 14 21 28')
args = parser.parse_args()
HORIZONS     = args.horizons

GAP          = 10
REGIMEN_MODE = 'block'
BLOCK_GAP    = 36.0
MAX_BLOCK    = 7
HEME_MODE    = 'leukemia_only_adj'
SEEDS        = [42, 43, 44]
THRESHOLDS   = [1.5, 1.0, 0.5]
LABELS       = {1.5: '<1500 (mild)', 1.0: '<1000 (moderate)', 0.5: '<500 (severe)'}

MIMIC_DIR    = os.path.join(_HERE, 'mimic_data')

DEFAULT_PARAMS = {
    'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05,
    'num_leaves': 31, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
    'bagging_freq': 5, 'verbose': -1,
}


# ─── 載入 MIMIC（一次，共用所有 seed）─────────────────────────────────────────
print("=" * 70)
print("載入 MIMIC 資料...")

mimic_lab = pd.read_csv(os.path.join(MIMIC_DIR, 'mimic_lab.csv'))
mimic_lab['SIGNIN'] = pd.to_datetime(mimic_lab['SIGNIN'], errors='coerce')
mimic_lab['RESULT'] = pd.to_numeric(mimic_lab['RESULT'], errors='coerce')
mimic_lab = mimic_lab.sort_values(['Newid', 'SIGNIN']).copy()
gcsf = mg.load_gcsf(MIMIC_DIR)

_mchemo = pd.read_csv(os.path.join(MIMIC_DIR, 'mimic_chemo.csv'))
_mchemo['time'] = pd.to_datetime(_mchemo['time'], errors='coerce')
_mchemo = _mchemo.dropna(subset=['Newid', 'time']).copy()
_mchemo['canon'] = _mchemo['item'].apply(pc.canonicalize_drug)
_conv = _mchemo.apply(
    lambda r: pc.convert_dose_to_mg(r['item'], r['dose'], r['unit']),
    axis=1, result_type='expand')
_mchemo['dose_mg']   = _conv[0].astype(float)
_mchemo['admin_cnt'] = _conv[1].astype(int)
mimic_chemo = (_mchemo[_mchemo['canon'] != 'GCSF']
               [['Newid', 'time', 'canon', 'dose_mg', 'admin_cnt']]
               .sort_values(['Newid', 'time']).reset_index(drop=True))

mimic_cm_raw  = pd.read_csv(os.path.join(MIMIC_DIR, 'mimic_patients.csv'))
mimic_cm_info = eac.build_cm_info(mimic_cm_raw)

print(f"  病人: {mimic_lab['Newid'].nunique()} | "
      f"化療筆: {len(mimic_chemo):,} | "
      f"demographics: {len(mimic_cm_info)}")

# ─── 載入台灣資料 ──────────────────────────────────────────────────────────────
print("\n載入台灣資料...")
tw_df    = pc.load_lab_df(exclude_heme_mode=HEME_MODE)
tw_chemo = pc.load_chemo_filtered(drop_gcsf=True)
tw_cm    = pd.read_csv(pc.CANCER_MAP_PATH)
tw_cm_info = eac.build_cm_info(tw_cm)
print(f"  台灣病人: {tw_df['Newid'].nunique()}")


# ─── 指標函式 ─────────────────────────────────────────────────────────────────
def binary_metrics(min_anc, prob, thr):
    y = (np.asarray(min_anc) < thr).astype(int)
    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    if n_pos < 2 or n_neg < 2:
        return None
    auc  = roc_auc_score(y, prob)
    prauc = average_precision_score(y, prob)
    fpr, tpr, _ = roc_curve(y, prob)
    valid = (1 - fpr) >= 0.8
    sens80 = float(tpr[valid].max()) if valid.any() else float('nan')
    return {'n': len(y), 'n_pos': n_pos, 'prev': n_pos/len(y),
            'auc': auc, 'pr_auc': prauc, 'sens_at_spec80': sens80}


# ─── 主迴圈：horizon × seed ───────────────────────────────────────────────────
# results[horizon][thr]['tw'|'mimic'] = list of AUCs across seeds
all_results = {
    h: {thr: {'tw': [], 'mimic': [], 'tw_pr': [], 'mimic_pr': [],
              'tw_prev': [], 'mimic_prev': []}
        for thr in THRESHOLDS}
    for h in HORIZONS
}

for HORIZON in HORIZONS:
    print(f"\n{'#'*70}")
    print(f"  HORIZON = {HORIZON} 天")
    print(f"{'#'*70}")

    for seed in SEEDS:
        print(f"\n{'='*70}")
        print(f"  Seed {seed}  |  horizon={HORIZON}d")
        print(f"{'='*70}")

        # 1. 台灣 train/test split
        uniq = tw_df['Newid'].unique()
        gss  = GroupShuffleSplit(test_size=0.2, random_state=seed)
        tr_idx, te_idx = next(gss.split(uniq, groups=uniq))
        train_pts = set(uniq[tr_idx])
        test_pts  = set(uniq[te_idx])

        # 2. 從台灣 training 決定 feature schema
        expected_markers = sorted(
            tw_df[tw_df['Newid'].isin(train_pts)]['NAME'].dropna().unique().tolist())
        top_drugs = pc.select_top_k_drugs(tw_chemo, train_pts, pc.TOP_K_DRUGS)

        # 3. 建台灣 anchor dataset
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
        print(f"  TW train anchors: {len(Xtr):,} | test anchors: {len(Xte):,}")

        # 4. 建 MIMIC anchor dataset（同 feature schema）
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
        print(f"  MIMIC anchors:    {len(X_mimic):,}")
        gk = mg.keep_masks(mimic_data, mimic_lab, gcsf, THRESHOLDS); mg.report(gk)
        X_mimic_all, anc_mimic_all = X_mimic, anc_mimic

        # 5. 每個 threshold 訓練 + 評估
        cut = int(0.85 * len(Xtr))
        print(f"\n  {'Threshold':<22} {'TW test AUC':>11}  {'MIMIC AUC':>10}  "
              f"{'MIMIC n_pos':>11}  {'MIMIC prev':>10}")
        print(f"  {'-'*68}")

        for thr in THRESHOLDS:
            X_mimic, anc_mimic = X_mimic_all[gk[thr]], anc_mimic_all[gk[thr]]
            ytr   = (anc_tr < thr).astype(int)
            n_pos = int(ytr.sum())
            if n_pos < 10:
                print(f"  {LABELS[thr]:<22} 跳過 (n_pos={n_pos})")
                continue
            spw    = float(len(ytr) - n_pos) / float(n_pos)
            params = dict(DEFAULT_PARAMS); params['scale_pos_weight'] = spw

            dtr   = lgb.Dataset(Xtr[:cut], label=ytr[:cut])
            dva   = lgb.Dataset(Xtr[cut:], label=ytr[cut:], reference=dtr)
            model = lgb.train(params, dtr, num_boost_round=1000, valid_sets=[dva],
                              callbacks=[lgb.early_stopping(30, verbose=False)])

            prob_tw    = model.predict(Xte,     num_iteration=model.best_iteration)
            prob_mimic = model.predict(X_mimic, num_iteration=model.best_iteration)

            r_tw    = binary_metrics(anc_te,    prob_tw,    thr)
            r_mimic = binary_metrics(anc_mimic, prob_mimic, thr)

            if r_tw:
                all_results[HORIZON][thr]['tw'].append(r_tw['auc'])
                all_results[HORIZON][thr]['tw_pr'].append(r_tw['pr_auc'])
                all_results[HORIZON][thr]['tw_prev'].append(r_tw['prev'])
            if r_mimic:
                all_results[HORIZON][thr]['mimic'].append(r_mimic['auc'])
                all_results[HORIZON][thr]['mimic_pr'].append(r_mimic['pr_auc'])
                all_results[HORIZON][thr]['mimic_prev'].append(r_mimic['prev'])

            tw_s   = f"{r_tw['auc']:.3f}"       if r_tw    else "N/A"
            mi_s   = f"{r_mimic['auc']:.3f}"    if r_mimic else "N/A"
            npos_s = str(r_mimic['n_pos'])       if r_mimic else "N/A"
            prev_s = f"{r_mimic['prev']*100:.1f}%" if r_mimic else "N/A"
            print(f"  {LABELS[thr]:<22} {tw_s:>11}  {mi_s:>10}  {npos_s:>11}  {prev_s:>10}")


# ─── 全 horizon 摘要表 ────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("=== Horizon Robustness 摘要 (3-seed mean ± SD) ===")
print(f"{'='*70}")
for thr in THRESHOLDS:
    print(f"\n{LABELS[thr]}")
    print(f"  {'Horizon':<10} {'TW test (internal)':>20}  {'MIMIC (external)':>20}  {'MIMIC n_pos':>12}")
    print(f"  {'-'*66}")
    for h in sorted(HORIZONS):
        tw_v  = all_results[h][thr]['tw']
        mi_v  = all_results[h][thr]['mimic']
        tw_s  = f"{np.mean(tw_v):.3f} ± {np.std(tw_v):.3f}" if tw_v else "N/A"
        mi_s  = f"{np.mean(mi_v):.3f} ± {np.std(mi_v):.3f}" if mi_v else "N/A"
        # n_pos 從 scoping 已知，這裡用最後一個 seed 的值近似
        npos  = all_results[h][thr]['mimic']
        print(f"  {h}d{'':<7} {tw_s:>20}  {mi_s:>20}")

print(f"\n{'='*70}")
print("=== PR-AUC + Lift 摘要 (3-seed mean ± SD) ===")
print("    Lift = PR-AUC / prevalence (跨 cohort 公平比較)")
print(f"{'='*70}")
for thr in THRESHOLDS:
    print(f"\n{LABELS[thr]}")
    print(f"  {'Horizon':<8} {'TW PR-AUC':>12}  {'TW lift':>9}  "
          f"{'MIMIC PR-AUC':>14}  {'MIMIC lift':>11}  {'MIMIC prev':>11}")
    print(f"  {'-'*72}")
    for h in sorted(HORIZONS):
        tw_pr   = all_results[h][thr]['tw_pr']
        mi_pr   = all_results[h][thr]['mimic_pr']
        tw_prev = all_results[h][thr]['tw_prev']
        mi_prev = all_results[h][thr]['mimic_prev']
        tw_pr_s = f"{np.mean(tw_pr):.3f}±{np.std(tw_pr):.3f}" if tw_pr else "N/A"
        mi_pr_s = f"{np.mean(mi_pr):.3f}±{np.std(mi_pr):.3f}" if mi_pr else "N/A"
        tw_lift = f"{np.mean(tw_pr)/np.mean(tw_prev):.1f}×"    if (tw_pr and tw_prev) else "N/A"
        mi_lift = f"{np.mean(mi_pr)/np.mean(mi_prev):.1f}×"    if (mi_pr and mi_prev) else "N/A"
        prev_s  = f"{np.mean(mi_prev)*100:.1f}%"               if mi_prev else "N/A"
        print(f"  {h}d{'':<5} {tw_pr_s:>12}  {tw_lift:>9}  {mi_pr_s:>14}  {mi_lift:>11}  {prev_s:>11}")

# backward-compat dummy
results = all_results[HORIZONS[-1]]
for thr in THRESHOLDS:
    tw_v = results[thr]['tw']
    mi_v = results[thr]['mimic']
    tw_s = f"{np.mean(tw_v):.3f} ± {np.std(tw_v):.3f}" if tw_v else "N/A"
    mi_s = f"{np.mean(mi_v):.3f} ± {np.std(mi_v):.3f}" if mi_v else "N/A"
    print(f"{LABELS[thr]:<22} {tw_s:>20}  {mi_s:>18}")

print(f"""
⚠️  解讀注意:
  - MIMIC prevalence ~27% vs 台灣 ~8.7% (住院 vs 門診化療，domain shift)
  - MIMIC dose 資料不可用，init_{{drug}}_dose / cum_dose 全為 0
    → 方案特徵依賴 init_n_admin / init_n_distinct_drugs
  - AUROC 是 rank-based，不受 prevalence 影響，可直接比較
  - Calibration 不會 transfer，需在 MIMIC 上重新校準才能比較
""")
