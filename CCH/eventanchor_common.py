"""
eventanchor_common.py — event-anchored 資料集建構 (從 event_anchored_eval.py 抽出共用)

讓 event_anchored_eval.py 跟 uq_eventanchored.py 共用「完全相同」的 anchor / 特徵建構,
避免重複實作 drift。所有特徵只用 anchor「之前」的資料 + anchor 當天 (REGIMEN_DAYS=0) 的方案。

⚠️ 這是 event-anchored dataset 的單一真實來源 (single source of truth)。
"""
import os
import sys
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))                       # CCH
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import pipeline_common as pc                                              # noqa: E402

ANC_NAME = 'Absolute Neutrophil Count'
NO_PREV_CYCLE_CAP = 365.0
DAY = np.timedelta64(1, 'D')

CANCER_CATS = ['Head and Neck', 'Lung', 'Breast', 'Liver/Biliary', 'Colorectal',
               'GI (Other)', 'Gynecologic', 'GU', 'Skin', 'CNS', 'Other',
               'Hematologic', 'Unknown']
_CAT2I = {c: i for i, c in enumerate(CANCER_CATS)}


def _sex_code(s):
    s = '' if pd.isna(s) else str(s)
    if s.startswith('女') or s.upper().startswith('F'):
        return 1
    if s.startswith('男') or s.upper().startswith('M'):
        return 0
    return -1


def build_cm_info(cm):
    """cm = patient_cancer_subgroups.csv 的 DataFrame; 回傳 {Newid: {cat,sex,age,dx}}"""
    cm = cm.copy()
    cm['Newid'] = cm['Newid'].astype(int)
    cm['dx_date'] = pd.to_datetime(cm['dx_date'], errors='coerce')
    info = {}
    for _, r in cm.iterrows():
        info[int(r['Newid'])] = {
            'cat': _CAT2I.get(r['cancer_category'], _CAT2I['Unknown']),
            'sex': _sex_code(r['sex']),
            'age': pd.to_numeric(r['age_at_dx'], errors='coerce'),
            'dx': r['dx_date'],
        }
    return info


def _recent_before(times, values, anchor, max_days=None):
    """Most recent observation strictly before anchor.
    If max_days is set, returns (None, None) if the observation is older than max_days days.
    """
    idx = np.searchsorted(times, anchor, side='right') - 1
    if idx < 0:
        return None, None
    dago = float((anchor - times[idx]) / DAY)
    if max_days is not None and dago > max_days:
        return None, None
    return float(values[idx]), dago


def build_anchor_dataset(df, chemo_df, cm_info, top_drugs, expected_markers,
                         horizon, gap, regimen_days, verbose=True,
                         include_anchor_time=False,
                         regimen_mode='day', block_gap_hours=36.0, max_block_days=7,
                         max_obs_days=None, include_window=False):
    """include_anchor_time=True 時額外加一欄 'anchor_time' (給 Sundial 等外部 join 用);
    include_window=True 時加 'anchor_time' + 'win_end' (標籤窗 (T, win_end], 給 MIMIC FN label join 用);
    預設皆 False, 不影響 event_anchored_eval 的 feature schema。"""
    """回傳 anchor-level DataFrame (特徵 + Newid + min_win_anc)。

    df          : cohort 過濾後的 lab DataFrame
    chemo_df    : chemo DataFrame
    cm_info     : build_cm_info() 的輸出
    top_drugs   : 由 TRAIN 病人選出的 top-K canonical drugs
    expected_markers : 由 TRAIN 病人決定的 marker 清單
    horizon/gap/regimen_days : 見 event_anchored_eval 說明
    max_obs_days : 若設定，捨棄最近一筆 ANC 距 T 超過此天數的 anchor（預設 None = 不限制）
    """
    other_markers = [m for m in expected_markers if m != ANC_NAME]
    # block 模式參數 (decision time = leading 連續 block 結束點)
    block_gap_td = np.timedelta64(int(round(block_gap_hours * 60)), 'm')
    max_block_td = np.timedelta64(int(max_block_days), 'D')

    if verbose:
        print("預先整理每位病人的 lab time series...")
    lab_by_pid = {}
    for pid, g in df.dropna(subset=['SIGNIN', 'RESULT']).groupby('Newid'):
        d = {}
        for name, gg in g.groupby('NAME'):
            gg = gg.sort_values('SIGNIN')
            d[name] = (gg['SIGNIN'].values.astype('datetime64[ns]'),
                       gg['RESULT'].values.astype(float))
        lab_by_pid[pid] = d

    chemo_lookup = pc.build_chemo_lookup(chemo_df)
    empty_t = np.array([], dtype='datetime64[ns]')
    empty_v = np.array([], dtype=float)

    def build_rows(pid):
        rows = []
        chemo = chemo_lookup.get(pid)
        if chemo is None or len(chemo) == 0:
            return rows
        labs = lab_by_pid.get(pid, {})
        anc_times, anc_vals = labs.get(ANC_NAME, (empty_t, empty_v))

        ctimes = chemo['time'].values.astype('datetime64[ns]')
        ccanon = chemo['canon'].values
        cdose = chemo['dose_mg'].values
        ccnt = chemo['admin_cnt'].values

        cycle_start_pos = [0]
        for i in range(1, len(ctimes)):
            if (ctimes[i] - ctimes[i - 1]) / DAY >= gap:
                cycle_start_pos.append(i)
        anchors = [ctimes[p] for p in cycle_start_pos]
        n_cyc = len(cycle_start_pos)

        info = cm_info.get(int(pid), {'cat': _CAT2I['Unknown'], 'sex': -1, 'age': np.nan, 'dx': pd.NaT})

        for ci in range(n_cyc):
            p = cycle_start_pos[ci]
            anchor = ctimes[p]                              # cycle 起點 (cycle spacing / cumulative 用)
            next_p = cycle_start_pos[ci + 1] if ci + 1 < n_cyc else len(ctimes)

            # --- decision time T + regimen 採計範圍 ---
            if regimen_mode == 'block':
                # leading 連續 block: consecutive gap <= block_gap_td 且整段 <= max_block_td
                be = p
                while be + 1 < next_p:
                    if ((ctimes[be + 1] - ctimes[be]) <= block_gap_td and
                            (ctimes[be + 1] - ctimes[p]) <= max_block_td):
                        be += 1
                    else:
                        break
                T = ctimes[be]                              # block 最後一筆給藥 → 之後才預測, 無 leakage
                regimen_block_mask = np.zeros(len(ctimes), dtype=bool)
                regimen_block_mask[p:be + 1] = True
            else:                                           # 'day' (reg0/reg3, frozen 預設)
                T = anchor
                regimen_block_mask = None

            # --- 需求: T 之前至少 1 次 ANC ---
            anc_before_idx = np.searchsorted(anc_times, T, side='right')
            if anc_before_idx < 1:
                continue
            # --- 標籤窗: (T, min(T+horizon, 下一 cycle 起點)] ---
            win_end = T + np.timedelta64(horizon, 'D')
            if ci + 1 < n_cyc:
                win_end = min(win_end, anchors[ci + 1])
            in_win = (anc_times > T) & (anc_times <= win_end)
            if not in_win.any():
                continue
            min_win_anc = float(anc_vals[in_win].min())

            feat = {}

            # ── ANC last (+ max_obs_days recency gate) ────────────────────────
            # _recent_before with max_days returns (None,None) if too old → drops anchor
            v, dago = _recent_before(anc_times, anc_vals, T, max_days=max_obs_days)
            if v is None:
                continue
            feat['anc_last'] = v
            feat['anc_last_daysago'] = dago
            feat['anc_last_mask'] = 1

            # ── Other lab markers (same max_obs_days window) ──────────────────
            for m in other_markers:
                mt, mv = labs.get(m, (empty_t, empty_v))
                vv, _ = _recent_before(mt, mv, T, max_days=max_obs_days)
                feat[f'{pc._safe(m)}_last'] = vv if vv is not None else 0.0
                feat[f'{pc._safe(m)}_last_mask'] = 1 if vv is not None else 0

            # ── Historical ANC statistics (restricted to max_obs_days window) ─
            # All prior ANC values before T
            all_prior_anc_t = anc_times[:anc_before_idx]
            all_prior_anc_v = anc_vals[:anc_before_idx]
            # Apply window if max_obs_days is set
            if max_obs_days is not None:
                cutoff = T - np.timedelta64(int(max_obs_days * 24 * 60), 'm')
                win = all_prior_anc_t >= cutoff
                prior_anc_t = all_prior_anc_t[win]
                prior_anc   = all_prior_anc_v[win]
            else:
                prior_anc_t = all_prior_anc_t
                prior_anc   = all_prior_anc_v

            feat['anc_last2'] = float(prior_anc[-2]) if len(prior_anc) >= 2 else feat['anc_last']
            if len(prior_anc) >= 2:
                dt = float((prior_anc_t[-1] - prior_anc_t[-2]) / DAY)
                feat['anc_slope_per_day'] = (prior_anc[-1] - prior_anc[-2]) / dt if dt > 0 else 0.0
            else:
                feat['anc_slope_per_day'] = 0.0

            feat['n_prior_anc']   = float(len(prior_anc))
            feat['hist_min_anc']  = float(prior_anc.min()) if len(prior_anc) > 0 else feat['anc_last']
            feat['n_prior_lt500'] = float((prior_anc < 0.5).sum())
            feat['n_prior_lt1000']= float((prior_anc < 1.0).sum())
            feat['n_prior_lt1500']= float((prior_anc < 1.5).sum())

            feat['cycle_number'] = float(ci + 1)
            if ci == 0:
                feat['days_since_prev_cycle'] = NO_PREV_CYCLE_CAP
            else:
                feat['days_since_prev_cycle'] = float(min((anchor - anchors[ci - 1]) / DAY, NO_PREV_CYCLE_CAP))

            # (e) 起始方案: day = anchor 當天(+regimen_days 日曆天); block = 整個 leading 連續 block
            if regimen_mode == 'block':
                init_mask = regimen_block_mask
            else:
                anchor_day = anchor.astype('datetime64[D]')
                init_hi = anchor_day + np.timedelta64(regimen_days + 1, 'D')
                init_mask = (ctimes >= anchor) & (ctimes < init_hi)
            for d in top_drugs:
                feat[f'init_{d}_dose'] = 0.0
            distinct = set()
            for c, dose, n in zip(ccanon[init_mask], cdose[init_mask], ccnt[init_mask]):
                distinct.add(c)
                if c in top_drugs:
                    feat[f'init_{c}_dose'] += float(dose)
            feat['init_n_distinct_drugs'] = float(len(distinct))
            feat['init_n_admin'] = float(ccnt[init_mask].sum())

            cum_mask = ctimes < anchor
            feat['cum_n_admin'] = float(ccnt[cum_mask].sum())
            feat['cum_dose'] = float(cdose[cum_mask].sum())

            feat['age_at_dx'] = float(info['age']) if pd.notna(info['age']) else -1.0
            feat['sex'] = float(info['sex'])
            feat['cancer_cat'] = float(info['cat'])
            if pd.notna(info['dx']):
                feat['days_since_dx'] = float(max((anchor - info['dx'].to_datetime64()) / DAY, 0.0))
            else:
                feat['days_since_dx'] = -1.0

            feat['Newid'] = pid
            feat['min_win_anc'] = min_win_anc
            if include_anchor_time or include_window:
                feat['anchor_time'] = pd.Timestamp(T)
            if include_window:
                feat['win_end'] = pd.Timestamp(win_end)
            rows.append(feat)
        return rows

    if verbose:
        print("建 anchor-level 樣本 (所有病人)...")
    all_rows = []
    for pid in df['Newid'].unique():
        all_rows.extend(build_rows(pid))
    data = pd.DataFrame(all_rows)
    if verbose:
        print(f"總 anchor 樣本數: {len(data)} (來自 {data['Newid'].nunique()} 位病人)")
    return data
