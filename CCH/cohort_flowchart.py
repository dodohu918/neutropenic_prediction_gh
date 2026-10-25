"""
cohort_flowchart.py — CCH 病人篩選流程 (cohort flowchart) 數字 + 圖

逐步重現 eventanchor_common.build_anchor_dataset 的 anchor 條件，算出每一關剩下的病人數:
  實驗室資料庫 → 排白血病/骨髓瘤 → 人工判讀排除 → 有 ANC → T 前有 ANC
  → T 前 28 天內有 ANC → 預測窗 (T, T+h] 內有 ANC = analytic cohort

與 build_anchor_dataset 相同設定 (h=28, gap=10, block36h, max_block=7, max_obs=28),
最後一關的病人/anchor 數必須與它一致 (程式結尾 assert)。

用法 (從 CCH/ 執行):
  python cohort_flowchart.py                 # 預設 exclude_mab=True (最終 pipeline)
  python cohort_flowchart.py --include-mab   # 舊版 (mAb 未排除) 數字, 供比對
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import eventanchor_common as eac   # noqa: E402  (會把 CCH/ 加進 sys.path)
import pipeline_common as pc       # noqa: E402

DAY = np.timedelta64(1, 'D')
HORIZON, GAP, BLOCK_GAP_H, MAX_BLOCK_D, MAX_OBS_D = 28, 10, 36.0, 7, 28


def flowchart_counts(exclude_mab=True, verbose=True):
    """回傳 list[(label, n_remaining, excluded_label, n_excluded)] 與 analytic anchors 數"""
    raw = pd.read_csv(pc.FILTERED_RESULT_PATH, usecols=['Newid'])
    n_db = raw['Newid'].nunique()
    n_noleuk = pc.load_lab_df(exclude_heme_mode='leukemia_only', verbose=False)['Newid'].nunique()
    df = pc.load_lab_df(exclude_heme_mode='leukemia_only_adj', verbose=False)
    chemo = pc.load_chemo_filtered(exclude_mab=exclude_mab, verbose=False)
    pids = df['Newid'].unique()

    anc = df[(df['NAME'] == eac.ANC_NAME)].dropna(subset=['SIGNIN', 'RESULT']).sort_values('SIGNIN')
    anc_by_pid = {p: g['SIGNIN'].values.astype('datetime64[ns]') for p, g in anc.groupby('Newid')}
    chemo_lookup = pc.build_chemo_lookup(chemo)

    block_gap_td = np.timedelta64(int(round(BLOCK_GAP_H * 60)), 'm')
    max_block_td = np.timedelta64(MAX_BLOCK_D, 'D')
    max_obs_td = np.timedelta64(MAX_OBS_D, 'D')

    has_anc, any_before, any_recent, any_label = set(), set(), set(), set()
    n_anchors = 0
    for pid in pids:
        at = anc_by_pid.get(pid)
        if at is None or len(at) == 0:
            continue
        has_anc.add(pid)
        ch = chemo_lookup.get(pid)
        if ch is None or len(ch) == 0:
            continue
        ct = ch['time'].values.astype('datetime64[ns]')
        starts = [0] + [i for i in range(1, len(ct)) if (ct[i] - ct[i - 1]) / DAY >= GAP]
        for ci, p in enumerate(starts):
            next_p = starts[ci + 1] if ci + 1 < len(starts) else len(ct)
            be = p
            while (be + 1 < next_p and (ct[be + 1] - ct[be]) <= block_gap_td
                   and (ct[be + 1] - ct[p]) <= max_block_td):
                be += 1
            T = ct[be]
            idx = np.searchsorted(at, T, side='right')
            if idx < 1:
                continue
            any_before.add(pid)
            if (T - at[idx - 1]) > max_obs_td:
                continue
            any_recent.add(pid)
            win_end = T + np.timedelta64(HORIZON, 'D')
            if ci + 1 < len(starts):
                win_end = min(win_end, ct[starts[ci + 1]])
            if ((at > T) & (at <= win_end)).any():
                any_label.add(pid)
                n_anchors += 1

    steps = [
        ('Laboratory database', n_db, None, None),
        ('After hematologic cancer exclusion', n_noleuk,
         'Excluded: leukemia or multiple myeloma', n_db - n_noleuk),
        ('After manual adjudication', len(pids),
         'Excluded: misclassified as lymphoma (leukemia) or\nlymphoma with bone-marrow involvement',
         n_noleuk - len(pids)),
        ('Patients with ≥1 ANC measurement', len(has_anc),
         'Excluded: no ANC measurement recorded\nduring the study period', len(pids) - len(has_anc)),
        ('Patients with ANC before decision time T', len(any_before),
         'Excluded: no ANC before decision time T in any\nchemotherapy cycle (incl. no non-mAb chemotherapy;\nfeature construction not possible)',
         len(has_anc) - len(any_before)),
        ('Patients with ≥1 ANC within 28-day\nobservation window before T', len(any_recent),
         'Excluded: ANC before T but outside the 28-day\nobservation window in all cycles',
         len(any_before) - len(any_recent)),
        ('Analytic cohort', len(any_label),
         'Excluded: no ANC within prediction window\n(T, T+28d] in any cycle (outcome labeling not possible)',
         len(any_recent) - len(any_label)),
    ]
    if verbose:
        for lab, n, ex, nex in steps:
            if ex:
                print(f"    ↳ {ex.replace(chr(10), ' ')}: {nex:,}")
            print(f"  {lab.replace(chr(10), ' ')}: {n:,}")
        print(f"  Cycle-level prediction anchors: {n_anchors:,}")
    return steps, n_anchors


def cohort_by_horizon(exclude_mab=True, horizons=(7, 14, 21, 28)):
    """各 horizon 的 cohort 大小 (Table 12) 與預測窗截斷分析 (Table 13), 全 cohort (非 test set)。
    有效窗長 = win_end − T; 下一 cycle 起點早於 T+h 時視為截斷。"""
    df = pc.load_lab_df(exclude_heme_mode='leukemia_only_adj', verbose=False)
    chemo = pc.load_chemo_filtered(exclude_mab=exclude_mab, verbose=False)
    cm = eac.build_cm_info(pd.read_csv(pc.CANCER_MAP_PATH))
    top = pc.select_top_k_drugs(chemo, set(df['Newid']), pc.TOP_K_DRUGS)
    markers = sorted(df['NAME'].dropna().unique())
    rows = []
    for h in horizons:
        d = eac.build_anchor_dataset(df, chemo, cm, top, markers, horizon=h, gap=GAP, regimen_days=0,
                                     regimen_mode='block', block_gap_hours=BLOCK_GAP_H,
                                     max_block_days=MAX_BLOCK_D, max_obs_days=MAX_OBS_D,
                                     verbose=False, include_window=True)
        eff = (d['win_end'] - d['anchor_time']).dt.total_seconds() / 86400
        rows.append({'horizon': h, 'patients': d['Newid'].nunique(), 'anchors': len(d),
                     'severe_anchors': int((d['min_win_anc'] < 0.5).sum()),
                     'severe_prev': float((d['min_win_anc'] < 0.5).mean()),
                     'truncated_frac': float((eff < h - 1e-9).mean()),
                     'eff_window_median': float(eff.median()), 'eff_window_mean': float(eff.mean())})
        print(f"  h={h}: {rows[-1]}")
    return pd.DataFrame(rows)


def cancer_distribution():
    """Source cohort 原發癌別分布 (Table 3), 未合併; 報告端再套用顯示用合併規則"""
    ids = set(pc.load_lab_df(exclude_heme_mode='leukemia_only_adj', verbose=False)['Newid'])
    cm = pd.read_csv(pc.CANCER_MAP_PATH, usecols=['Newid', 'cancer_category']).drop_duplicates('Newid')
    cat = pd.Series(sorted(ids)).map(cm.set_index('Newid')['cancer_category']).fillna('Unknown')
    return cat.value_counts().rename_axis('cancer_category').reset_index(name='n')


def drug_usage(exclude_mab=True):
    """Source cohort (排除血液腫瘤後) 各藥物使用病人數 / 給藥次數 (Table 6)。
    % cohort 分母 = source cohort 病人數; % administrations 分母 = 全部給藥次數 (已排 mAb)。"""
    df = pc.load_lab_df(exclude_heme_mode='leukemia_only_adj', verbose=False)
    ids = set(df['Newid'])
    chemo = pc.load_chemo_filtered(exclude_mab=exclude_mab, verbose=False)
    chemo = chemo[chemo['Newid'].isin(ids)]
    total = chemo['admin_cnt'].sum()
    g = (chemo.groupby('canon')
         .agg(patients=('Newid', 'nunique'), administrations=('admin_cnt', 'sum'))
         .reset_index().rename(columns={'canon': 'drug'}))
    g['pct_cohort'] = g['patients'] / len(ids) * 100
    g['pct_administrations'] = g['administrations'] / total * 100
    g['cohort_n'], g['total_administrations'] = len(ids), total
    return g.sort_values('administrations', ascending=False)


def draw_flowchart(steps, n_anchors, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(10, 11))
    ax.set_xlim(0, 10); ax.set_ylim(0, 11); ax.axis('off')
    main_x, main_w, ex_x, ex_w, box_h = 0.3, 3.9, 5.2, 4.5, 0.95
    ys = np.linspace(10.2, 0.6, len(steps))
    face, edge = '#E8E8E8', '#1F3864'

    def box(x, y, w, text, bold=False):
        ax.add_patch(FancyBboxPatch((x, y - box_h / 2), w, box_h, boxstyle='round,pad=0.02,rounding_size=0.12',
                                    fc=face, ec=edge, lw=1.4))
        ax.text(x + w / 2, y, text, ha='center', va='center', fontsize=9.5,
                fontweight='bold' if bold else 'normal', linespacing=1.3)

    cx = main_x + main_w / 2
    for i, (lab, n, ex, nex) in enumerate(steps):
        y = ys[i]
        txt = f"{lab}\nn = {n:,}"
        if i == len(steps) - 1:
            txt += f"\n({n_anchors:,} cycle-level anchors)"
        box(main_x, y, main_w, txt, bold=(i == len(steps) - 1))
        if i + 1 < len(steps):
            y2 = ys[i + 1]
            ax.annotate('', xy=(cx, y2 + box_h / 2), xytext=(cx, y - box_h / 2),
                        arrowprops=dict(arrowstyle='-|>', color=edge, lw=1.6))
            ym = (y + y2) / 2
            _, _, ex2, nex2 = steps[i + 1]
            ax.annotate('', xy=(ex_x, ym), xytext=(cx, ym),
                        arrowprops=dict(arrowstyle='-|>', color=edge, lw=1.6))
            box(ex_x, ym, ex_w, f"{ex2}\nn = {nex2:,}")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ Flowchart → {out_path}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--include-mab', action='store_true', help='不排除 mAb (舊版數字)')
    args = ap.parse_args()
    excl = not args.include_mab
    print(f"Cohort flowchart | h={HORIZON} | block36h | max_obs={MAX_OBS_D}d | exclude_mab={excl}")
    steps, n_anchors = flowchart_counts(exclude_mab=excl)

    # 一致性檢查: 必須與 build_anchor_dataset 相同
    df = pc.load_lab_df(exclude_heme_mode='leukemia_only_adj', verbose=False)
    chemo = pc.load_chemo_filtered(exclude_mab=excl, verbose=False)
    cm = eac.build_cm_info(pd.read_csv(pc.CANCER_MAP_PATH))
    d = eac.build_anchor_dataset(df, chemo, cm, pc.select_top_k_drugs(chemo, set(df['Newid']), pc.TOP_K_DRUGS),
                                 sorted(df['NAME'].dropna().unique()), horizon=HORIZON, gap=GAP,
                                 regimen_days=0, regimen_mode='block', block_gap_hours=BLOCK_GAP_H,
                                 max_block_days=MAX_BLOCK_D, max_obs_days=MAX_OBS_D, verbose=False)
    assert (d['Newid'].nunique(), len(d)) == (steps[-1][1], n_anchors), \
        f"不一致: build_anchor_dataset={d['Newid'].nunique()}/{len(d)} vs flowchart={steps[-1][1]}/{n_anchors}"
    print("  ✓ 與 build_anchor_dataset 一致")

    tag = 'nomab' if excl else 'withmab'
    draw_flowchart(steps, n_anchors, os.path.join(_HERE, f'fig_cohort_flowchart_h28_obs28_{tag}.png'))

    print("\nCohort by horizon (Table 12 / 13)...")
    by_h = cohort_by_horizon(exclude_mab=excl)
    out_csv = os.path.join(_HERE, f'cohort_by_horizon_obs28_{tag}.csv')
    by_h.to_csv(out_csv, index=False)
    print(f"  ✓ → {out_csv}")

    print("\nCancer-site distribution in source cohort (Table 3)...")
    out_csv = os.path.join(_HERE, 'cancer_distribution_sourcecohort.csv')
    cancer_distribution().to_csv(out_csv, index=False)
    print(f"  ✓ → {out_csv}")

    print("\nDrug usage in source cohort (Table 6)...")
    out_csv = os.path.join(_HERE, f'drug_usage_sourcecohort_{tag}.csv')
    drug_usage(exclude_mab=excl).to_csv(out_csv, index=False)
    print(f"  ✓ → {out_csv}")
