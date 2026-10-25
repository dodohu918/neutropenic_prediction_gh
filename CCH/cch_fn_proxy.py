"""cch_fn_proxy.py — CCH 端 FN-proxy

FN proxy 定義 (與 MIMIC/mimic_eval_lgb_fn.py 相同):
  預測窗 (T, win_end] 內某次 ANC<500, 其 ±48h 內同時有
    (1) blood culture (抽血培養), 且
    (2) infection work-up lab: CRP 或 Procalcitonin (任何數值)
  兩者皆須在 anchor T 之後 (避免預測時點前已發生)。
計算 h=28 SN anchors 中符合 FN-proxy 的比例。cohort 與主 pipeline 完全相同
(block36h, leukemia_only_adj, obs28, 排除 biologics)。

Blood culture 不在 test_result*.csv 內, 從 BC_PATH 讀 (欄位: Newid, SIGNIN = 抽血培養時間)。

用法 (從本資料夾執行):  python cch_fn_proxy.py
"""
import os, sys
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import eventanchor_common as eac   # noqa: E402  (會把 CCH/ 加進 sys.path)
import pipeline_common as pc       # noqa: E402

_DATA_DIR = os.path.dirname(pc.FILTERED_RESULT_PATH)
RAW = [os.path.join(_DATA_DIR, f) for f in ('test_result1.csv', 'test_result2.csv')]
BC_PATH = os.path.join(_DATA_DIR, 'blood_culture.csv')
INFECTION_LABS = {'CRP', 'Procalcitonin'}
PROXY_WIN = np.timedelta64(48, 'h')

df = pc.load_lab_df(exclude_heme_mode='leukemia_only_adj', verbose=False)
chemo = pc.load_chemo_filtered(verbose=False)
cm = eac.build_cm_info(pd.read_csv(pc.CANCER_MAP_PATH))
top = pc.select_top_k_drugs(chemo, set(df['Newid']), pc.TOP_K_DRUGS)
d = eac.build_anchor_dataset(df, chemo, cm, top, sorted(df['NAME'].dropna().unique()),
                             horizon=28, gap=10, regimen_days=0, regimen_mode='block',
                             block_gap_hours=36.0, max_block_days=7, max_obs_days=28,
                             verbose=False, include_window=True)
sn = d[d['min_win_anc'] < 0.5]

raw = pd.concat([pd.read_csv(f, usecols=['Newid', 'NAME', 'SIGNIN'], low_memory=False) for f in RAW])
raw = raw[raw['NAME'].astype(str).str.strip().isin(INFECTION_LABS)].copy()
raw['SIGNIN'] = pd.to_datetime(raw['SIGNIN'], errors='coerce')
lab_by = {k: g['SIGNIN'].values for k, g in raw.dropna().groupby('Newid')}
bc = pd.read_csv(BC_PATH, usecols=['Newid', 'SIGNIN'])
bc['SIGNIN'] = pd.to_datetime(bc['SIGNIN'], errors='coerce')
bc_by = {k: g['SIGNIN'].values for k, g in bc.dropna().groupby('Newid')}
anc = df[(df['NAME'] == eac.ANC_NAME) & (df['RESULT'] < 0.5)].dropna(subset=['SIGNIN'])
anc_by = {k: pd.to_datetime(g['SIGNIN']).values for k, g in anc.groupby('Newid')}

print(f"h=28 anchors {len(d):,} | SN anchors {len(sn):,}")
w = PROXY_WIN; n = 0
for r in sn.itertuples():
    b, c, a = bc_by.get(r.Newid), lab_by.get(r.Newid), anc_by.get(r.Newid)
    if b is None or c is None or a is None:
        continue
    T = np.datetime64(r.anchor_time)
    b, c = b[b > T], c[c > T]
    a = a[(a > T) & (a <= np.datetime64(r.win_end))]
    n += any(((b >= t - w) & (b <= t + w)).any() and ((c >= t - w) & (c <= t + w)).any() for t in a)
print(f"  BC + CRP/PCT within ±48h of an in-window ANC<500: {n:,}/{len(sn):,} = {n/len(sn):.1%}")
