"""
subgroup_cancer_lgb.py — LightGBM AUROC by cancer subtype (uses existing NPZ prediction files)

Uses predictions_eventanchored_h{H}_block36h_noleukadj_seed{S}.npz
Joins patient IDs with patient_cancer_subgroups.csv to get cancer_category.
Reports mean ± SD AUROC per cancer subtype across seeds.

Usage:
    python subgroup_cancer_lgb.py
    python subgroup_cancer_lgb.py --horizon 7
    python subgroup_cancer_lgb.py --threshold 0.5   # ANC<500
    python subgroup_cancer_lgb.py --min-anchors 30  # hide subtypes with too few samples
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
_CANCER_MAP = os.path.join(_HERE, 'patient_cancer_subgroups.csv')

SEEDS = [40, 41, 42]

parser = argparse.ArgumentParser()
parser.add_argument('--horizon',    type=int,   default=28,   help='horizon in days (default 28)')
parser.add_argument('--threshold',  type=float, default=0.5,  help='ANC threshold (0.5=<500, 1.0=<1000, 1.5=<1500)')
parser.add_argument('--min-anchors',type=int,   default=20,   help='min anchors per subtype to show (default 20)')
args = parser.parse_args()

H   = args.horizon
THR = args.threshold
thr_str = {0.5: 'ANC<500 (severe)', 1.0: 'ANC<1000 (moderate)', 1.5: 'ANC<1500 (mild)'}.get(THR, f'thr={THR}')

# Load cancer category map
cm = pd.read_csv(_CANCER_MAP, usecols=['Newid', 'cancer_category'])
pid2cat = cm.set_index('Newid')['cancer_category'].to_dict()

print("=" * 70)
print(f"LightGBM Subgroup Analysis | horizon={H}d | {thr_str}")
print(f"Seeds: {SEEDS}  |  min_anchors={args.min_anchors}")
print("=" * 70)

# Collect per-seed per-subtype AUROCs
seed_results = {}   # seed → DataFrame with columns [cat, n, n_pos, auc]

for seed in SEEDS:
    npz_path = os.path.join(_HERE, f'predictions_eventanchored_h{H}_block36h_noleukadj_seed{seed}.npz')
    if not os.path.exists(npz_path):
        print(f"  [WARN] missing: {os.path.basename(npz_path)}")
        continue

    npz  = np.load(npz_path, allow_pickle=True)
    pids  = npz[f'thr{THR}_pid']
    pred  = npz[f'thr{THR}_pred']
    truth = (npz[f'thr{THR}_truth'] < THR).astype(int)   # raw ANC → binary label

    # Map each anchor's patient → cancer_category
    cats = np.array([pid2cat.get(int(p), 'Unknown') for p in pids])

    rows = []
    for cat in sorted(set(cats)):
        mask = cats == cat
        n = mask.sum()
        n_pos = truth[mask].sum()
        if n < args.min_anchors or n_pos < 5:
            continue
        try:
            auc = roc_auc_score(truth[mask], pred[mask])
        except Exception:
            auc = float('nan')
        rows.append({'cat': cat, 'n': n, 'n_pos': int(n_pos),
                     'prev': n_pos / n, 'auc': auc})

    seed_results[seed] = pd.DataFrame(rows)
    print(f"\n  Seed {seed} — {int(truth.sum())} pos / {len(truth)} anchors total")

# Merge across seeds — align by cancer_category
all_cats = sorted(set(c for df in seed_results.values() for c in df['cat']))
print(f"\n\n{'Category':<22} {'N(anchors)':>12} {'N(pos)':>8} {'Prev%':>7}  ", end='')
for s in SEEDS:
    print(f"  S{s}", end='')
print(f"  {'Mean±SD':>12}")
print("-" * 90)

for cat in all_cats:
    aucs = []
    ns   = []
    npos = []
    for seed in SEEDS:
        if seed not in seed_results:
            continue
        row = seed_results[seed][seed_results[seed]['cat'] == cat]
        if len(row) == 0:
            aucs.append(float('nan'))
        else:
            aucs.append(row.iloc[0]['auc'])
            ns.append(row.iloc[0]['n'])
            npos.append(row.iloc[0]['n_pos'])
    valid = [a for a in aucs if not np.isnan(a)]
    if not valid:
        continue
    mean_n   = int(np.mean(ns))   if ns   else 0
    mean_npos = int(np.mean(npos)) if npos else 0
    prev_pct = 100 * mean_npos / mean_n if mean_n > 0 else 0
    auc_str  = f"{np.mean(valid):.3f}±{np.std(valid, ddof=1):.3f}" if len(valid) > 1 else f"{valid[0]:.3f}"
    per_seed = ''.join(f"  {'—' if np.isnan(a) else f'{a:.3f}'}" for a in aucs)
    print(f"  {cat:<20} {mean_n:>12,} {mean_npos:>8,} {prev_pct:>6.1f}%{per_seed}  {auc_str:>12}")

# Overall (all anchors pooled across seeds — just show seed-level overall)
print(f"\n{'Overall':<22}", end='')
overall_aucs = []
for seed in SEEDS:
    npz_path = os.path.join(_HERE, f'predictions_eventanchored_h{H}_block36h_noleukadj_seed{seed}.npz')
    if not os.path.exists(npz_path):
        overall_aucs.append(float('nan'))
        continue
    npz = np.load(npz_path, allow_pickle=True)
    pred  = npz[f'thr{THR}_pred']
    truth = (npz[f'thr{THR}_truth'] < THR).astype(int)
    overall_aucs.append(roc_auc_score(truth, pred))

total_n    = len(truth)
total_npos = int(truth.sum())
print(f" {total_n:>12,} {total_npos:>8,} {100*total_npos/total_n:>6.1f}%", end='')
for a in overall_aucs:
    print(f"  {'—' if np.isnan(a) else f'{a:.3f}'}", end='')
v = [a for a in overall_aucs if not np.isnan(a)]
print(f"  {np.mean(v):.3f}±{np.std(v, ddof=1):.3f}" if len(v) > 1 else f"  {v[0]:.3f}")

print()
