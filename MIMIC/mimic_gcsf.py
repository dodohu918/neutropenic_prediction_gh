"""
mimic_gcsf.py — 排除受 G-CSF 影響 (contaminated) 的 MIMIC anchors

G-CSF 會拉高 ANC, 讓原本會發生的 neutropenia 不發生 (或延後), 因此視為對 label 的污染。
anchor T 的標籤窗為 (T, win_end]; 若 G-CSF 給藥時間落在
    [T - GCSF_LOOKBACK_DAYS, t_event]
就排除該 anchor。t_event = 窗內第一次 ANC < thr 的時間; 窗內沒有則 = win_end。
  - T 之前 GCSF_LOOKBACK_DAYS 天內: 前一週期的 G-CSF (尤其 pegfilgrastim) 效果仍在
  - (T, t_event]: 本週期的預防性 G-CSF
  - t_event 之後的 G-CSF 是治療性用藥, label 已確定, 不排除 (保留這些 positives)
t_event 依 threshold 而定, 所以每個 threshold 各有一個 mask。

需要: mimic_extract.py 產生的 mimic_data/mimic_gcsf.csv,
      build_anchor_dataset(include_window=True) 的 anchor_time / win_end 欄位。
"""
import os
import numpy as np
import pandas as pd

GCSF_LOOKBACK_DAYS = 14
ANC_NAME = 'Absolute Neutrophil Count'


def load_gcsf(mimic_dir):
    """回傳 {Newid: 排序後的 G-CSF 給藥時間 (datetime64 array)}"""
    g = pd.read_csv(os.path.join(mimic_dir, 'mimic_gcsf.csv'), usecols=['Newid', 'time'])
    g['time'] = pd.to_datetime(g['time'], errors='coerce')
    g = g.dropna().sort_values(['Newid', 'time'])
    return {k: grp['time'].values for k, grp in g.groupby('Newid')}


def keep_masks(data, lab, gcsf, thresholds, lookback_days=GCSF_LOOKBACK_DAYS):
    """回傳 {thr: bool array}, True = 保留 (未受 G-CSF 影響), index 對齊 data。"""
    anc = lab[lab['NAME'] == ANC_NAME][['Newid', 'SIGNIN', 'RESULT']].copy()
    anc['SIGNIN'] = pd.to_datetime(anc['SIGNIN'], errors='coerce')
    anc['RESULT'] = pd.to_numeric(anc['RESULT'], errors='coerce')
    anc = anc.dropna().sort_values(['Newid', 'SIGNIN'])
    anc_by = {k: (grp['SIGNIN'].values, grp['RESULT'].values) for k, grp in anc.groupby('Newid')}
    lb = np.timedelta64(lookback_days, 'D')
    empty = np.array([], dtype='datetime64[ns]')

    masks = {thr: np.ones(len(data), dtype=bool) for thr in thresholds}
    for i, (pid, T, we) in enumerate(zip(data['Newid'].values,
                                         data['anchor_time'].values, data['win_end'].values)):
        g = gcsf.get(pid)
        if g is None:
            continue
        g = g[g >= T - lb]
        if len(g) == 0:
            continue
        t_anc, v_anc = anc_by.get(pid, (empty, np.array([])))
        in_win = (t_anc > T) & (t_anc <= we)
        for thr in thresholds:
            ev = t_anc[in_win & (v_anc < thr)]
            t_event = ev[0] if len(ev) else we
            if (g <= t_event).any():
                masks[thr][i] = False
    return masks


def report(masks, label='MIMIC'):
    for thr, m in masks.items():
        print(f"  [G-CSF] {label} ANC<{thr*1000:.0f}: 排除 {int((~m).sum()):,}/{len(m):,} anchors "
              f"(lookback {GCSF_LOOKBACK_DAYS}d)")
