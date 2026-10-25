"""
mimic_eval_lgb.py — LightGBM 台灣訓練 → MIMIC-IV 外部驗證，多 horizon

用法 (從 MIMIC/ 執行):
  python mimic_eval_lgb.py --seeds 42 48 49 --horizons 7 14 21 28 --max-obs-days 28
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

_HERE = os.path.dirname(os.path.abspath(__file__))
_TW   = os.path.join(os.path.dirname(_HERE), 'CCH')
sys.path.insert(0, _HERE); sys.path.insert(0, _TW)

import pipeline_common as pc
import eventanchor_common as eac
import mimic_gcsf as mg

parser = argparse.ArgumentParser()
parser.add_argument('--seeds',        type=int, nargs='+', default=[42, 48, 49])
parser.add_argument('--horizons',     type=int, nargs='+', default=[7, 14, 21, 28])
parser.add_argument('--max-obs-days', type=int, default=28)
args = parser.parse_args()
SEEDS, HORIZONS, MAX_OBS_DAYS = args.seeds, args.horizons, args.max_obs_days

GAP, REGIMEN_MODE, BLOCK_GAP, MAX_BLOCK = 10, 'block', 36.0, 7
HEME_MODE  = 'leukemia_only_adj'
THRESHOLDS = [1.5, 1.0, 0.5]
LABELS     = {1.5: '<1500 (mild)', 1.0: '<1000 (moderate)', 0.5: '<500 (severe)'}
MIMIC_DIR  = os.path.join(_HERE, 'mimic_data')

LGB_PARAMS = {'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05,
              'num_leaves': 31, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
              'bagging_freq': 5, 'verbose': -1}

print(f"LightGBM MIMIC eval | seeds={SEEDS} | horizons={HORIZONS} | max_obs_days={MAX_OBS_DAYS}")

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
mimic_cm_info = eac.build_cm_info(pd.read_csv(os.path.join(MIMIC_DIR, 'mimic_patients.csv')))
print(f"  MIMIC 病人: {mimic_lab['Newid'].nunique()} | 化療: {len(mimic_chemo):,}")

print("載入台灣資料...")
tw_df    = pc.load_lab_df(exclude_heme_mode=HEME_MODE)
tw_chemo = pc.load_chemo_filtered()
tw_cm    = eac.build_cm_info(pd.read_csv(pc.CANCER_MAP_PATH))
print(f"  台灣病人: {tw_df['Newid'].nunique()}")


def metrics(y_true, y_prob):
    y = np.asarray(y_true); p = np.asarray(y_prob)
    n_pos = int(y.sum()); n = len(y); n_neg = n - n_pos
    if n_pos < 2 or n_neg < 2:
        return None
    prev   = n_pos / n
    auroc  = roc_auc_score(y, p)
    prauc  = average_precision_score(y, p)
    lift   = prauc / prev
    fpr, tpr, _ = roc_curve(y, p)
    valid  = (1 - fpr) >= 0.8
    sens80 = float(tpr[valid].max()) if valid.any() else float('nan')
    return {'n': n, 'n_pos': n_pos, 'prev': prev,
            'auroc': auroc, 'prauc': prauc, 'lift': lift, 'sens_at_spec80': sens80}


# all_res[horizon][thr]['tw'|'mimic'][seed] = metrics dict
all_res = {h: {thr: {'tw': {}, 'mimic': {}} for thr in THRESHOLDS} for h in HORIZONS}

for HORIZON in HORIZONS:
    print(f"\n{'#'*70}\n  HORIZON = {HORIZON}d\n{'#'*70}")

    for seed in SEEDS:
        print(f"\n{'='*60}\n  Seed {seed}  |  horizon={HORIZON}d\n{'='*60}")
        np.random.seed(seed)

        uniq = tw_df['Newid'].unique()
        tr_i, te_i = next(GroupShuffleSplit(test_size=0.2, random_state=seed).split(uniq, groups=uniq))
        train_pts, test_pts = set(uniq[tr_i]), set(uniq[te_i])

        markers   = sorted(tw_df[tw_df['Newid'].isin(train_pts)]['NAME'].dropna().unique().tolist())
        top_drugs = pc.select_top_k_drugs(tw_chemo, train_pts, pc.TOP_K_DRUGS)

        tw_data = eac.build_anchor_dataset(
            tw_df, tw_chemo, tw_cm, top_drugs, markers,
            horizon=HORIZON, gap=GAP, regimen_days=0,
            regimen_mode=REGIMEN_MODE, block_gap_hours=BLOCK_GAP,
            max_block_days=MAX_BLOCK, max_obs_days=MAX_OBS_DAYS, verbose=False)
        fc    = [c for c in tw_data.columns if c not in ('Newid', 'min_win_anc')]
        tw_tr = tw_data[tw_data['Newid'].isin(train_pts)]
        tw_te = tw_data[tw_data['Newid'].isin(test_pts)]
        Xtr = tw_tr[fc].values; anc_tr = tw_tr['min_win_anc'].values
        Xte = tw_te[fc].values; anc_te = tw_te['min_win_anc'].values
        print(f"  TW train: {len(Xtr):,} | test: {len(Xte):,}")

        mimic_data = eac.build_anchor_dataset(
            mimic_lab, mimic_chemo, mimic_cm_info, top_drugs, markers,
            horizon=HORIZON, gap=GAP, regimen_days=0,
            regimen_mode=REGIMEN_MODE, block_gap_hours=BLOCK_GAP,
            max_block_days=MAX_BLOCK, max_obs_days=MAX_OBS_DAYS, verbose=False, include_window=True)
        for c in fc:
            if c not in mimic_data.columns: mimic_data[c] = 0.0
        X_mi  = mimic_data[fc].values; anc_mi = mimic_data['min_win_anc'].values
        print(f"  MIMIC anchors: {len(X_mi):,}")
        gk = mg.keep_masks(mimic_data, mimic_lab, gcsf, THRESHOLDS); mg.report(gk)
        X_mi_all, anc_mi_all = X_mi, anc_mi

        cut = int(0.85 * len(Xtr))

        for thr in THRESHOLDS:
            X_mi, anc_mi = X_mi_all[gk[thr]], anc_mi_all[gk[thr]]
            ytr   = (anc_tr < thr).astype(int)
            n_pos = int(ytr.sum())
            if n_pos < 10: continue
            spw   = float(len(ytr) - n_pos) / n_pos

            p = dict(LGB_PARAMS); p['scale_pos_weight'] = spw
            dtr   = lgb.Dataset(Xtr[:cut], label=ytr[:cut])
            dva   = lgb.Dataset(Xtr[cut:], label=ytr[cut:], reference=dtr)
            model = lgb.train(p, dtr, num_boost_round=1000, valid_sets=[dva],
                              callbacks=[lgb.early_stopping(30, verbose=False)])

            prob_tw = model.predict(Xte, num_iteration=model.best_iteration)
            prob_mi = model.predict(X_mi, num_iteration=model.best_iteration)

            r_tw = metrics((anc_te < thr).astype(int), prob_tw)
            r_mi = metrics((anc_mi < thr).astype(int), prob_mi)
            if r_tw: all_res[HORIZON][thr]['tw'][seed]    = r_tw
            if r_mi: all_res[HORIZON][thr]['mimic'][seed] = r_mi

            tw_s = f"{r_tw['auroc']:.3f}" if r_tw else "N/A"
            mi_s = f"{r_mi['auroc']:.3f}" if r_mi else "N/A"
            print(f"  {LABELS[thr]}: TW={tw_s}  MIMIC={mi_s}")


# ── Summary ───────────────────────────────────────────────────────────────────
def _mv(dicts, key):
    return [d[key] for d in dicts if d is not None and key in d]

def _ms(dicts, key, fmt='.3f'):
    v = _mv(dicts, key)
    if not v: return 'N/A'
    if len(v) == 1: return f"{v[0]:{fmt}}"
    return f"{np.mean(v):{fmt}} ± {np.std(v, ddof=1):{fmt}}"

print(f"\n\n{'='*110}")
print(f"=== LightGBM MIMIC 外部驗證摘要 | seeds={SEEDS} | max_obs_days={MAX_OBS_DAYS} ===")
print(f"{'='*110}")

for thr in THRESHOLDS:
    print(f"\n── {LABELS[thr]} ──")
    print(f"  {'H':>4}  {'Set':>5}  {'AUROC':>18}  {'PR-AUC':>18}  {'Lift':>8}  {'Prev':>7}  {'Sens@Sp80':>10}  {'ΔAUROC':>8}")
    print(f"  {'-'*90}")
    for h in sorted(HORIZONS):
        tw_d = list(all_res[h][thr]['tw'].values())
        mi_d = list(all_res[h][thr]['mimic'].values())

        tw_auroc = _mv(tw_d, 'auroc'); mi_auroc = _mv(mi_d, 'auroc')
        delta = (f"{np.mean(mi_auroc) - np.mean(tw_auroc):+.3f}"
                 if (tw_auroc and mi_auroc) else 'N/A')

        for label, dicts in [('TW', tw_d), ('MIMIC', mi_d)]:
            prev_v = _mv(dicts, 'prev')
            prev_s = f"{np.mean(prev_v)*100:.1f}%" if prev_v else 'N/A'
            d_s    = delta if label == 'MIMIC' else ''
            print(f"  {h:>3}d  {label:>5}  {_ms(dicts,'auroc'):>18}  "
                  f"{_ms(dicts,'prauc'):>18}  {_ms(dicts,'lift','.2f'):>8}  "
                  f"{prev_s:>7}  {_ms(dicts,'sens_at_spec80'):>10}  {d_s:>8}")
        print()

print(f"""
注意:
  - max_obs_days={MAX_OBS_DAYS}: lab features 僅用 T 前 {MAX_OBS_DAYS} 天內資料
  - MIMIC dose 特徵全為 0 (emar 無欄位)
  - Lift = PR-AUC ÷ prevalence (相對 random baseline 放大倍率)
  - Sens@Sp80 = sensitivity at specificity ≥ 80%
  - ΔAUROC = MIMIC mean − TW test mean
  - SD: sample std (ddof=1), N={len(SEEDS)} seeds
""")
