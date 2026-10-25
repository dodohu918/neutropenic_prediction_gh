"""
subgroup_eventanchored.py — Cancer-subtype subgroup analysis for the event-anchored model

讀已存的 block-mode 預測 (predictions_eventanchored_h28_block36h_seed*.npz),
依癌別把 TEST anchors 分群, 算每個 subgroup × threshold 的 AUROC / PR-AUC / Sens@Spec80 /
PPV@Sens0.8, 並跨 3 seeds 報 mean ± SD。不需重訓 (用存好的預測)。

⚠️ 注意: cohort ③ 已排除 leukemia/myeloma (骨髓原發), 所以這裡的 'Hematologic' 類
   = 只剩 lymphoma (淋巴結/脾/胸腺原發)。報表會標註。

用法:
  cd CCH
  python subgroup_eventanchored.py
"""
import os
import sys
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import pipeline_common as pc
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument('--tag', default='block36h',
                 help="預測檔的 regtag; adj cohort 用 'block36h_noleukadj'")
_args = _ap.parse_args()

SEEDS = [42, 43, 44]
HORIZON = 28
REGTAG = _args.tag
THRESHOLDS = [1.5, 1.0, 0.5]
LABELS = {1.5: '<1500', 1.0: '<1000', 0.5: '<500 (severe)'}
MIN_POS_PER_SEED = 8        # 每 seed 至少要這麼多陽性才算該 subgroup 的 AUROC

# cancer map: Newid -> category (cohort ③ 下 'Hematologic' = lymphoma only)
cm = pd.read_csv(pc.CANCER_MAP_PATH, usecols=['Newid', 'cancer_category'])
pid2cat = dict(zip(cm['Newid'].astype(int), cm['cancer_category'].fillna('Unknown')))


def metrics(truth, pred, thr):
    y = (np.asarray(truth) < thr).astype(int)
    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    if n_pos < MIN_POS_PER_SEED or n_neg < MIN_POS_PER_SEED:
        return None
    auc = roc_auc_score(y, pred); pr = average_precision_score(y, pred)
    fpr, tpr, _ = roc_curve(y, pred)
    valid = (1 - fpr) >= 0.8
    sens80 = float(tpr[valid].max()) if valid.any() else np.nan
    m = tpr >= 0.8
    if m.any():
        i = np.where(m)[0]; b = i[np.argmin(fpr[i])]
        tp = tpr[b] * n_pos; fp = fpr[b] * n_neg
        ppv80 = tp / (tp + fp) if tp + fp > 0 else 0.0
    else:
        ppv80 = np.nan
    return {'n': len(y), 'n_pos': n_pos, 'prev': n_pos / len(y),
            'auc': auc, 'pr': pr, 'sens80': sens80, 'ppv80': ppv80}


# 收集: per seed, per threshold, anchors 的 (cat, truth, pred)
per_seed = {}   # seed -> threshold -> DataFrame(cat, truth, pred)
for s in SEEDS:
    d = np.load(os.path.join(_HERE, f'predictions_eventanchored_h{HORIZON}_{REGTAG}_seed{s}.npz'),
                allow_pickle=True)
    per_seed[s] = {}
    for thr in THRESHOLDS:
        pid = d[f'thr{thr}_pid']; pred = d[f'thr{thr}_pred']; truth = d[f'thr{thr}_truth']
        cat = np.array([pid2cat.get(int(p), 'Unknown') for p in pid])
        per_seed[s][thr] = pd.DataFrame({'cat': cat, 'truth': truth, 'pred': pred})

# 全 category 分布 (用 seed 42 test 看大致 anchor 數)
cats = sorted(set(per_seed[42][0.5]['cat']),
              key=lambda c: -(per_seed[42][0.5]['cat'] == c).sum())

print("=" * 100)
print(f"Cancer-subtype subgroup analysis | event-anchored block mode | h{HORIZON} | 3 seeds")
print("  (cohort ③: 'Hematologic' = lymphoma only, leukemia/myeloma 已排除)")
print("=" * 100)
print(f"{'Category':<20}{'Threshold':<16}{'avg n_anchor':<14}{'avg n_pos':<12}"
      f"{'AUROC (mean±SD)':<20}{'Sens@Spec80':<14}{'PPV@Sens0.8':<12}")
print("-" * 108)

for cat in cats:
    if cat == 'Unknown':
        continue
    for thr in THRESHOLDS:
        aucs, sens, ppvs, npos, nanc = [], [], [], [], []
        for s in SEEDS:
            sub = per_seed[s][thr]
            m = sub[sub['cat'] == cat]
            r = metrics(m['truth'].values, m['pred'].values, thr) if len(m) else None
            if r:
                aucs.append(r['auc']); sens.append(r['sens80']); ppvs.append(r['ppv80'])
                npos.append(r['n_pos']); nanc.append(r['n'])
        if len(aucs) >= 2:    # 至少 2 seed 有足夠陽性才報
            auc_s = f"{np.mean(aucs):.3f} ± {np.std(aucs):.3f}"
            print(f"{cat:<20}{LABELS[thr]:<16}{np.mean(nanc):<14.0f}{np.mean(npos):<12.0f}"
                  f"{auc_s:<20}{np.nanmean(sens):<14.3f}{np.nanmean(ppvs)*100:<11.1f}%")
    print()

print(f"(只報「≥2 seeds 且每 seed 陽性 ≥ {MIN_POS_PER_SEED}」的 subgroup × threshold; "
      f"其餘樣本太少 AUROC 不穩, 略過)")
print("對照: 整體 cohort <500 = 0.854±0.013, <1000 = 0.840, <1500 = 0.816")
