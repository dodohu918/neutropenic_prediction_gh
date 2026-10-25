"""
plot_risk_coverage.py — aggregate riskcov_h*.csv / margin_h*.csv and produce:
  1. Main figure: 28-day risk-coverage (x=acceptance rate, y=selective error rate),
     3 seeds + mean+-SD band, alpha=0.10 marked.
  2. Supplementary figure: same for 7/14/21-day (3-panel).
  3. Supplementary figure: complementary probability-margin analysis (renamed
     |p-0.5| AUROC-vs-coverage curve), 4-panel (7/14/21/28d).
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
HORIZONS = [7, 14, 21, 28]
SEEDS = [40, 41, 42]
TAG = 'block36h_noleukadj'
ALPHA_PRIMARY = 0.10


def load_riskcov(h):
    files = [os.path.join(_HERE, f'riskcov_h{h}_{TAG}_seed{s}.csv') for s in SEEDS]
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def load_margin(h):
    files = [os.path.join(_HERE, f'margin_h{h}_{TAG}_seed{s}.csv') for s in SEEDS]
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# 1. Main figure — 28-day risk-coverage curve
# ---------------------------------------------------------------------------
df28 = load_riskcov(28).sort_values(['seed', 'alpha'])

fig, ax = plt.subplots(figsize=(6.5, 5.5))
colors = dict(zip(SEEDS, ['#1f77b4', '#ff7f0e', '#2ca02c']))
for s in SEEDS:
    d = df28[df28['seed'] == s].sort_values('singleton_acceptance_rate')
    ax.plot(d['singleton_acceptance_rate'] * 100, d['selective_error_rate_among_singletons'] * 100,
            'o-', color=colors[s], alpha=0.55, lw=1.3, ms=4, label=f'seed {s}')

# mean +- SD band (aligned on alpha grid, not on x since acceptance_rate varies slightly per seed)
agg = df28.groupby('alpha').agg(
    acc_mean=('singleton_acceptance_rate', 'mean'), acc_sd=('singleton_acceptance_rate', 'std'),
    err_mean=('selective_error_rate_among_singletons', 'mean'),
    err_sd=('selective_error_rate_among_singletons', 'std'),
).reset_index().sort_values('acc_mean')
ax.plot(agg['acc_mean'] * 100, agg['err_mean'] * 100, 'k-', lw=2.2, label='mean (3 seeds)')
ax.fill_between(agg['acc_mean'] * 100,
                 (agg['err_mean'] - agg['err_sd']) * 100,
                 (agg['err_mean'] + agg['err_sd']) * 100,
                 color='gray', alpha=0.2, label='+-1 SD')

# mark alpha=0.10 (primary)
prim = agg[np.isclose(agg['alpha'], ALPHA_PRIMARY)]
ax.scatter(prim['acc_mean'] * 100, prim['err_mean'] * 100, marker='*', s=300,
           color='red', zorder=5, label=f'primary (alpha={ALPHA_PRIMARY:.2f})')

ax.set_xlabel('Acceptance rate (% of anchors with singleton prediction set)')
ax.set_ylabel('Selective error rate among singletons (%)')
ax.set_title('Conformal selective risk-coverage — ANC<500 (severe), 28-day horizon')
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc='upper left')
plt.tight_layout()
main_png = os.path.join(_HERE, 'fig_riskcoverage_h28_block36h_noleukadj_main.png')
plt.savefig(main_png, dpi=140, bbox_inches='tight')
plt.close()
print(f"✓ main figure -> {main_png}")

# ---------------------------------------------------------------------------
# 2. Supplementary figure — 7/14/21-day risk-coverage (3-panel)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for ax, h in zip(axes, [7, 14, 21]):
    dfh = load_riskcov(h).sort_values(['seed', 'alpha'])
    for s in SEEDS:
        d = dfh[dfh['seed'] == s].sort_values('singleton_acceptance_rate')
        ax.plot(d['singleton_acceptance_rate'] * 100, d['selective_error_rate_among_singletons'] * 100,
                'o-', color=colors[s], alpha=0.55, lw=1.3, ms=4, label=f'seed {s}')
    aggh = dfh.groupby('alpha').agg(
        acc_mean=('singleton_acceptance_rate', 'mean'),
        err_mean=('selective_error_rate_among_singletons', 'mean'),
        err_sd=('selective_error_rate_among_singletons', 'std'),
    ).reset_index().sort_values('acc_mean')
    ax.plot(aggh['acc_mean'] * 100, aggh['err_mean'] * 100, 'k-', lw=2, label='mean')
    ax.fill_between(aggh['acc_mean'] * 100,
                     (aggh['err_mean'] - aggh['err_sd']) * 100,
                     (aggh['err_mean'] + aggh['err_sd']) * 100,
                     color='gray', alpha=0.2, label='+-1 SD')
    primh = aggh[np.isclose(aggh['alpha'], ALPHA_PRIMARY)]
    ax.scatter(primh['acc_mean'] * 100, primh['err_mean'] * 100, marker='*', s=220,
               color='red', zorder=5, label='alpha=0.10')
    ax.set_title(f'{h}-day horizon')
    ax.set_xlabel('Acceptance rate (%)')
    ax.grid(alpha=0.3)
axes[0].set_ylabel('Selective error rate among singletons (%)')
axes[0].legend(fontsize=7, loc='upper left')
plt.suptitle('Conformal selective risk-coverage — ANC<500 (severe), horizon sensitivity (Supplementary)')
plt.tight_layout()
supp_png = os.path.join(_HERE, 'fig_riskcoverage_h7-14-21_block36h_supplementary.png')
plt.savefig(supp_png, dpi=140, bbox_inches='tight')
plt.close()
print(f"✓ supplementary risk-coverage figure -> {supp_png}")

# ---------------------------------------------------------------------------
# 3. Supplementary — complementary probability-margin analysis (renamed |p-0.5| curve)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(19, 4.5), sharey=False)
for ax, h in zip(axes, HORIZONS):
    dfm = load_margin(h)
    for s in SEEDS:
        d = dfm[dfm['seed'] == s].sort_values('coverage_pct')
        ax.plot(d['coverage_pct'], d['auroc'], 'o-', color=colors[s], alpha=0.6,
                lw=1.3, ms=4, label=f'seed {s}')
    ax.set_title(f'{h}-day horizon')
    ax.set_xlabel('Coverage (% of anchors kept, most-confident first)')
    ax.grid(alpha=0.3)
axes[0].set_ylabel('AUROC on retained anchors')
axes[0].legend(fontsize=7, loc='lower right')
plt.suptitle('Complementary probability-margin analysis (|p-0.5| ranked retention) — '
             'ANC<500 (severe), Supplementary')
plt.tight_layout()
margin_png = os.path.join(_HERE, 'fig_probmargin_h7-14-21-28_supplementary.png')
plt.savefig(margin_png, dpi=140, bbox_inches='tight')
plt.close()
print(f"✓ probability-margin supplementary figure -> {margin_png}")

# ---------------------------------------------------------------------------
# 4. Summary table (alpha grid x horizon, mean+-SD) printed to stdout
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("Summary: mean +- SD across 3 seeds, ANC<500 severe")
print("=" * 100)
for h in HORIZONS:
    dfh = load_riskcov(h)
    agg_kwargs = dict(
        overall_coverage_mean=('overall_coverage', 'mean'), overall_coverage_sd=('overall_coverage', 'std'),
        positive_class_coverage_mean=('positive_class_coverage', 'mean'),
        positive_class_coverage_sd=('positive_class_coverage', 'std'),
        negative_class_coverage_mean=('negative_class_coverage', 'mean'),
        negative_class_coverage_sd=('negative_class_coverage', 'std'),
        singleton_acceptance_rate_mean=('singleton_acceptance_rate', 'mean'),
        singleton_acceptance_rate_sd=('singleton_acceptance_rate', 'std'),
        abstention_rate_mean=('abstention_rate', 'mean'), abstention_rate_sd=('abstention_rate', 'std'),
        selective_error_rate_among_singletons_mean=('selective_error_rate_among_singletons', 'mean'),
        selective_error_rate_among_singletons_sd=('selective_error_rate_among_singletons', 'std'),
        confident_low_ppv_mean=('confident_low_ppv', 'mean'), confident_low_ppv_sd=('confident_low_ppv', 'std'),
    )
    if 'q_pos' in dfh.columns:
        agg_kwargs.update(
            q_pos_mean=('q_pos', 'mean'), q_pos_sd=('q_pos', 'std'),
            q_neg_mean=('q_neg', 'mean'), q_neg_sd=('q_neg', 'std'),
        )
    g = dfh.groupby('alpha').agg(**agg_kwargs).reset_index()
    print(f"\n--- horizon={h}d ---")
    for _, r in g.iterrows():
        line = (f"  alpha={r['alpha']:<6.3f} "
                f"overall_cov={r['overall_coverage_mean']*100:5.1f}±{r['overall_coverage_sd']*100:4.1f}%  "
                f"pos_cov={r['positive_class_coverage_mean']*100:5.1f}±{r['positive_class_coverage_sd']*100:4.1f}%  "
                f"neg_cov={r['negative_class_coverage_mean']*100:5.1f}±{r['negative_class_coverage_sd']*100:4.1f}%  "
                f"accept={r['singleton_acceptance_rate_mean']*100:5.1f}±{r['singleton_acceptance_rate_sd']*100:4.1f}%  "
                f"abstain={r['abstention_rate_mean']*100:5.1f}±{r['abstention_rate_sd']*100:4.1f}%  "
                f"sel_err={r['selective_error_rate_among_singletons_mean']*100:5.1f}±"
                f"{r['selective_error_rate_among_singletons_sd']*100:4.1f}%  "
                f"ppv_low={r['confident_low_ppv_mean']*100:5.1f}±{r['confident_low_ppv_sd']*100:4.1f}%")
        if 'q_pos_mean' in g.columns:
            line += (f"  q+={r['q_pos_mean']:.3f}±{r['q_pos_sd']:.3f}  "
                      f"q-={r['q_neg_mean']:.3f}±{r['q_neg_sd']:.3f}")
        print(line)

    g.to_csv(os.path.join(_HERE, f'riskcov_summary_h{h}.csv'), index=False)
