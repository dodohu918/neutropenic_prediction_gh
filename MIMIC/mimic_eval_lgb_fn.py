"""
mimic_eval_lgb_fn.py — 台灣訓練的 LightGBM (ANC<500) 在 MIMIC-IV FN 子群的表現

前置: python mimic_fn_labels.py (產生 mimic_cohort_fixed.csv / mimic_fn_events.csv / mimic_fn_icd_adm.csv)

模型與 mimic_eval_lgb.py 完全相同 (台灣 severe ANC<500 label 訓練)，只換 MIMIC 端的 outcome:
  severe_orig   : 原 cohort，min_win_anc < 0.5 (重現既有結果)
  severe        : 修正 cohort，min_win_anc < 0.5
  FN_proxy      : 標籤窗 (T, win_end] 內某次 ANC<500，其 ±48h 內同時有 blood culture + CRP 或 Procalcitonin
                  (與 CCH/cch_fn_proxy.py 定義相同)
                  (proxy 事件須在 T 之後，避免預測時點前已發生)
  FN_vs_afebG4  : 只在 severe anchors 中比較 FN_proxy vs 無 FN 的 grade 4
  FN_BC         : 同上但只要求 blood culture
  FN_ICD        : ANC<500 落在 ICD 編碼 FN 的住院期間
  FN_definite   : FN_proxy 且 (ICU 體溫 ≥38.3°C 或 ICD FN)

用法 (從 MIMIC/ 執行):
  python mimic_eval_lgb_fn.py --seeds 40 41 42 --horizons 7 14 21 28 --max-obs-days 28
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
parser.add_argument('--seeds',        type=int, nargs='+', default=[40, 41, 42])
parser.add_argument('--horizons',     type=int, nargs='+', default=[7, 14, 21, 28])
parser.add_argument('--max-obs-days', type=int, default=28)
args = parser.parse_args()
SEEDS, HORIZONS, MAX_OBS_DAYS = args.seeds, args.horizons, args.max_obs_days

GAP, REGIMEN_MODE, BLOCK_GAP, MAX_BLOCK = 10, 'block', 36.0, 7
HEME_MODE  = 'leukemia_only_adj'
THR        = 0.5
PROXY_WIN  = pd.Timedelta('48h')
MIMIC_DIR  = os.path.join(_HERE, 'mimic_data')
OUTCOMES   = ['severe_orig', 'severe', 'FN_proxy', 'FN_vs_afebG4', 'FN_BC', 'FN_ICD', 'FN_definite']

LGB_PARAMS = {'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05,
              'num_leaves': 31, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
              'bagging_freq': 5, 'verbose': -1}

print(f"LightGBM MIMIC FN eval | seeds={SEEDS} | horizons={HORIZONS} | max_obs_days={MAX_OBS_DAYS}")

# ── Load MIMIC ────────────────────────────────────────────────────────────────
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

pts_orig  = pd.read_csv(os.path.join(MIMIC_DIR, 'mimic_patients.csv'))
cohort    = pd.read_csv(os.path.join(MIMIC_DIR, 'mimic_cohort_fixed.csv'))
keep_ids  = set(cohort.loc[cohort['keep'] == 1, 'Newid'])
pts_fixed = (pts_orig.drop(columns=['cancer_category', 'primary_site'])
             .merge(cohort[['Newid', 'cancer_category', 'primary_site']], on='Newid'))
cm_orig  = eac.build_cm_info(pts_orig)
cm_fixed = eac.build_cm_info(pts_fixed[pts_fixed['Newid'].isin(keep_ids)])
lab_fixed   = mimic_lab[mimic_lab['Newid'].isin(keep_ids)]
chemo_fixed = mimic_chemo[mimic_chemo['Newid'].isin(keep_ids)]
print(f"  原 cohort: {mimic_lab['Newid'].nunique()} 位 | 修正後: {lab_fixed['Newid'].nunique()} 位")

fn_ev = pd.read_csv(os.path.join(MIMIC_DIR, 'mimic_fn_events.csv'), parse_dates=['time'])
icd_adm = pd.read_csv(os.path.join(MIMIC_DIR, 'mimic_fn_icd_adm.csv'),
                      parse_dates=['admittime', 'dischtime'])
sev_anc = (mimic_lab[(mimic_lab['NAME'] == eac.ANC_NAME) & (mimic_lab['RESULT'] < THR)]
           [['Newid', 'SIGNIN']].rename(columns={'SIGNIN': 't'}).drop_duplicates())

print("載入台灣資料...")
tw_df    = pc.load_lab_df(exclude_heme_mode=HEME_MODE)
tw_chemo = pc.load_chemo_filtered()
tw_cm    = eac.build_cm_info(pd.read_csv(pc.CANCER_MAP_PATH))


def fn_labels(data):
    """data: build_anchor_dataset(include_window=True) 輸出; 回傳各 FN label (index 對齊 data)"""
    a = data[['Newid', 'anchor_time', 'win_end']].reset_index().rename(columns={'index': 'aid'})
    s = a.merge(sev_anc, on='Newid')
    s = s[(s['t'] > s['anchor_time']) & (s['t'] <= s['win_end'])]          # 窗內每一次 ANC<500

    def near(kind):
        e = fn_ev.loc[fn_ev['kind'] == kind, ['Newid', 'time']]
        m = s.merge(e, on='Newid')
        m = m[((m['time'] - m['t']).abs() <= PROXY_WIN) & (m['time'] > m['anchor_time'])]
        return set(zip(m['aid'], m['t']))

    bc, inflab, fever = near('BC'), near('INFLAB'), near('FEVER')
    m = s.merge(icd_adm, on='Newid')
    m = m[(m['t'] >= m['admittime']) & (m['t'] <= m['dischtime'])]
    icd = set(zip(m['aid'], m['t']))

    proxy = bc & inflab
    lab = pd.DataFrame(index=data.index)
    aid_of = lambda pairs: data.index.isin({p[0] for p in pairs})
    lab['FN_proxy']    = aid_of(proxy)
    lab['FN_BC']       = aid_of(bc)
    lab['FN_ICD']      = aid_of(icd)
    lab['FN_definite'] = aid_of(proxy & (fever | icd))
    return lab.astype(int)


def metrics(y_true, y_prob):
    y = np.asarray(y_true); p = np.asarray(y_prob)
    n_pos = int(y.sum()); n = len(y); n_neg = n - n_pos
    if n_pos < 5 or n_neg < 5:
        return None
    prev  = n_pos / n
    auroc = roc_auc_score(y, p)
    prauc = average_precision_score(y, p)
    fpr, tpr, _ = roc_curve(y, p)
    valid = (1 - fpr) >= 0.8
    sens80 = float(tpr[valid].max()) if valid.any() else float('nan')
    return {'n': n, 'n_pos': n_pos, 'prev': prev, 'auroc': auroc, 'prauc': prauc,
            'lift': prauc / prev, 'sens_at_spec80': sens80}


N_BOOT = 1000


def boot_ci(y, p, pids, n_boot=N_BOOT, seed=0):
    """以病人為單位 bootstrap 的 AUROC 95% CI (同一病人多個 anchors 一起抽)"""
    y, p = np.asarray(y), np.asarray(p)
    groups = pd.Series(np.arange(len(y))).groupby(np.asarray(pids)).apply(np.array).values
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = np.concatenate(groups[rng.integers(0, len(groups), len(groups))])
        if 0 < y[idx].sum() < len(idx):
            aucs.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def build(df, chemo, cm, top_drugs, markers, horizon, **kw):
    return eac.build_anchor_dataset(
        df, chemo, cm, top_drugs, markers, horizon=horizon, gap=GAP, regimen_days=0,
        regimen_mode=REGIMEN_MODE, block_gap_hours=BLOCK_GAP, max_block_days=MAX_BLOCK,
        max_obs_days=MAX_OBS_DAYS, verbose=False, **kw)


# res[h][outcome][seed] = metrics dict ; tw_res[h][seed]
res    = {h: {o: {} for o in OUTCOMES} for h in HORIZONS}
tw_res = {h: {} for h in HORIZONS}

for HORIZON in HORIZONS:
    print(f"\n{'#'*70}\n  HORIZON = {HORIZON}d\n{'#'*70}")
    for seed in SEEDS:
        np.random.seed(seed)
        uniq = tw_df['Newid'].unique()
        tr_i, te_i = next(GroupShuffleSplit(test_size=0.2, random_state=seed).split(uniq, groups=uniq))
        train_pts, test_pts = set(uniq[tr_i]), set(uniq[te_i])
        markers   = sorted(tw_df[tw_df['Newid'].isin(train_pts)]['NAME'].dropna().unique().tolist())
        top_drugs = pc.select_top_k_drugs(tw_chemo, train_pts, pc.TOP_K_DRUGS)

        tw_data = build(tw_df, tw_chemo, tw_cm, top_drugs, markers, HORIZON)
        fc    = [c for c in tw_data.columns if c not in ('Newid', 'min_win_anc')]
        tw_tr = tw_data[tw_data['Newid'].isin(train_pts)]
        tw_te = tw_data[tw_data['Newid'].isin(test_pts)]
        Xtr, ytr = tw_tr[fc].values, (tw_tr['min_win_anc'].values < THR).astype(int)

        spw = float(len(ytr) - ytr.sum()) / ytr.sum()
        p = dict(LGB_PARAMS); p['scale_pos_weight'] = spw
        cut = int(0.85 * len(Xtr))
        dtr = lgb.Dataset(Xtr[:cut], label=ytr[:cut])
        dva = lgb.Dataset(Xtr[cut:], label=ytr[cut:], reference=dtr)
        model = lgb.train(p, dtr, num_boost_round=1000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(30, verbose=False)])
        predict = lambda X: model.predict(X, num_iteration=model.best_iteration)

        tw_res[HORIZON][seed] = metrics((tw_te['min_win_anc'].values < THR).astype(int),
                                        predict(tw_te[fc].values))

        def mimic_X(data):
            for c in fc:
                if c not in data.columns: data[c] = 0.0
            return data[fc].values

        # 原 cohort (重現)
        mo = build(mimic_lab, mimic_chemo, cm_orig, top_drugs, markers, HORIZON, include_window=True)
        gk = mg.keep_masks(mo, mimic_lab, gcsf, [THR]); mg.report(gk, 'MIMIC 原 cohort')
        mo = mo[gk[THR]].copy()
        res[HORIZON]['severe_orig'][seed] = metrics((mo['min_win_anc'].values < THR).astype(int),
                                                    predict(mimic_X(mo)))
        # 修正 cohort + FN labels
        mf  = build(lab_fixed, chemo_fixed, cm_fixed, top_drugs, markers, HORIZON, include_window=True)
        gk = mg.keep_masks(mf, lab_fixed, gcsf, [THR]); mg.report(gk, 'MIMIC 修正 cohort')
        mf  = mf[gk[THR]].copy()
        lab = fn_labels(mf)
        prob = predict(mimic_X(mf))
        sev  = (mf['min_win_anc'].values < THR).astype(int)
        res[HORIZON]['severe'][seed] = metrics(sev, prob)
        for o in ['FN_proxy', 'FN_BC', 'FN_ICD', 'FN_definite']:
            res[HORIZON][o][seed] = metrics(lab[o].values, prob)
        in_sev = sev == 1
        res[HORIZON]['FN_vs_afebG4'][seed] = metrics(lab['FN_proxy'].values[in_sev], prob[in_sev])
        # 主要 outcome 的病人層級 bootstrap CI
        pids = mf['Newid'].values
        for o, y in [('severe', sev), ('FN_proxy', lab['FN_proxy'].values)]:
            if res[HORIZON][o][seed]:
                lo, hi = boot_ci(y, prob, pids, seed=seed)
                res[HORIZON][o][seed].update(ci_lo=lo, ci_hi=hi)

        if seed == SEEDS[0]:
            out = mf[['Newid', 'anchor_time', 'win_end', 'min_win_anc']].join(lab).assign(prob=prob)
            out.to_csv(os.path.join(_HERE, f'mimic_fn_predictions_h{HORIZON}_seed{seed}.csv'), index=False)
            print(f"  MIMIC anchors 原/修正: {len(mo):,}/{len(mf):,} | severe {sev.sum()} | "
                  + " | ".join(f"{o} {lab[o].sum()}" for o in lab.columns))

        s = lambda r: f"{r['auroc']:.3f}" if r else 'N/A'
        print(f"  seed {seed}: TW={s(tw_res[HORIZON][seed])} | "
              + " | ".join(f"{o}={s(res[HORIZON][o][seed])}" for o in OUTCOMES))


# ── 機器可讀結果 ───────────────────────────────────────────────────────────────
rows = []
for h in HORIZONS:
    for seed in SEEDS:
        if tw_res[h].get(seed):
            rows.append({'horizon': h, 'seed': seed, 'outcome': 'TW_internal_severe', **tw_res[h][seed]})
        for o in OUTCOMES:
            if res[h][o].get(seed):
                rows.append({'horizon': h, 'seed': seed, 'outcome': o, **res[h][o][seed]})
out_csv = os.path.join(_HERE, 'mimic_fn_results.csv')
pd.DataFrame(rows).to_csv(out_csv, index=False)
print(f"\n✓ 結果 CSV → {out_csv}")


# ── Summary ───────────────────────────────────────────────────────────────────
def _ms(dicts, key, fmt='.3f'):
    v = [d[key] for d in dicts if d is not None]
    if not v: return 'N/A'
    if len(v) == 1: return f"{v[0]:{fmt}}"
    return f"{np.mean(v):{fmt}} ± {np.std(v, ddof=1):{fmt}}"

DESC = {'severe_orig': 'ANC<500, 原 cohort', 'severe': 'ANC<500, 修正 cohort',
        'FN_proxy': 'FN proxy (BC+CRP/PCT)', 'FN_vs_afebG4': 'FN vs 無 FN grade 4 (severe 內)',
        'FN_BC': 'FN (只要 BC)', 'FN_ICD': 'FN (ICD)', 'FN_definite': 'FN definite (+體溫/ICD)'}

print(f"\n\n{'='*112}")
print(f"=== 台灣 LightGBM (ANC<500) → MIMIC FN 子群 | seeds={SEEDS} | max_obs_days={MAX_OBS_DAYS} ===")
print(f"{'='*112}")
for h in sorted(HORIZONS):
    print(f"\n── horizon {h}d ── (TW internal test AUROC {_ms(list(tw_res[h].values()), 'auroc')})")
    print(f"  {'Outcome':<34}{'n':>7}{'pos':>6}{'Prev':>7}  {'AUROC':>15}  {'PR-AUC':>15}  {'Lift':>6}  {'Sens@Sp80':>15}")
    print(f"  {'-'*108}")
    for o in OUTCOMES:
        d = list(res[h][o].values())
        r0 = next((x for x in d if x), None)
        if r0 is None:
            print(f"  {DESC[o]:<34}{'N/A':>7}"); continue
        print(f"  {DESC[o]:<34}{r0['n']:>7,}{r0['n_pos']:>6}{r0['prev']*100:>6.1f}%  "
              f"{_ms(d,'auroc'):>15}  {_ms(d,'prauc'):>15}  {_ms(d,'lift','.2f'):>6}  {_ms(d,'sens_at_spec80'):>15}")

print(f"""
注意:
  - 模型皆為台灣 ANC<500 label 訓練的 LightGBM，MIMIC 端只換 outcome，不重新訓練
  - n / pos / Prev 取第一個 seed (MIMIC anchors 各 seed 相同)
  - FN proxy = ANC<500 ±48h 內同時有 blood culture + CRP 或 Procalcitonin (MIMIC-IV 3.1 只有 CRP)
  - FN_vs_afebG4: 只在 severe anchors 內，看模型能否把 FN 排在無 FN 的 grade 4 前面
  - 修正 cohort: 排除無惡性腫瘤碼 (多為 MTX/CTX 的自體免疫病人) 與 ICD-9 白血病/骨髓瘤
  - SD: sample std (ddof=1), N={len(SEEDS)} seeds
""")
