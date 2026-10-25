"""
uq_eventanchored.py — Uncertainty Quantification on the frozen event-anchored model

建在 frozen config 上 (event-anchored, horizon 28, regimen-days 0, cohort ③,
default LightGBM)。實作三層 UQ:

  1. Calibration (isotonic)        — 讓機率可信 (Brier / ECE before vs after)
  2. Split conformal (Mondrian)    — class-conditional, distribution-free coverage 保證,
                                     產生 per-anchor prediction set {low}/{not-low}/{uncertain}
  3. Selective prediction          — 模型「不確定就 abstain (建議抽 CBC / 醫師複核)」,
                                     confident 子集的 PPV / AUROC 應上升

方法學紀律:
  - 病人層級三分: train (fit model) / calib (fit isotonic + conformal) / TEST (評估), 全 GroupShuffleSplit
  - top_drugs / markers 只從 train 選
  - ⚠️ Exchangeability caveat: 一個病人有多個 anchor → anchor 不可交換 → naive coverage 偏樂觀。
    因此額外報「每病人取 1 個 anchor」的 patient-level 保守 coverage。

用法:
  cd CCH
  python uq_eventanchored.py --seed 42 --alpha 0.1
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
import pipeline_common as pc
import eventanchor_common as eac


# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--horizon', type=int, default=28)
parser.add_argument('--regimen-days', type=int, default=0)
parser.add_argument('--regimen-mode', choices=['day', 'block'], default='block',
                    help='block = 新 primary (leading 連續 block, decision time=block 結束)')
parser.add_argument('--block-gap-hours', type=float, default=36.0)
parser.add_argument('--max-block-days', type=int, default=7)
parser.add_argument('--max-obs-days', type=int, default=28,
                    help='觀察窗 (天); 預設 28')
parser.add_argument('--heme-mode', default='leukemia_only',
                    choices=['none', 'all', 'leukemia_only', 'leukemia_only_adj'],
                    help='leukemia_only_adj = 額外排掉淋巴瘤判讀 ①/②')
parser.add_argument('--alpha', type=float, default=0.1, help='conformal significance (1-alpha = 目標 coverage)')
args = parser.parse_args()

SEED = args.seed
HORIZON = args.horizon
REGIMEN_DAYS = args.regimen_days
REGIMEN_MODE = args.regimen_mode
HEME_MODE = args.heme_mode
MAX_OBS_DAYS = args.max_obs_days
REGTAG = f'reg{REGIMEN_DAYS}' if REGIMEN_MODE == 'day' else f'block{args.block_gap_hours:g}h'
if HEME_MODE != 'leukemia_only':
    REGTAG = f'{REGTAG}_{pc._TAG_MAP[HEME_MODE]}'
ALPHA = args.alpha
THRESHOLDS = [1.5, 1.0, 0.5]
LABELS = {1.5: 'ANC<1500 (mild)', 1.0: 'ANC<1000 (moderate)', 0.5: 'ANC<500 (severe)'}
DEFAULT_PARAMS = {
    'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05,
    'num_leaves': 31, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
    'bagging_freq': 5, 'verbose': -1,
}

print("=" * 80)
print(f"UQ on event-anchored | horizon={HORIZON} reg={REGIMEN_DAYS} seed={SEED} | "
      f"target coverage = {(1-ALPHA)*100:.0f}% (alpha={ALPHA})")
print("=" * 80)


# ---------------------------------------------------------------------------
# 1. 載入 + 病人層級三分 (train / calib / TEST)
# ---------------------------------------------------------------------------
df = pc.load_lab_df(exclude_heme_mode=HEME_MODE)
chemo_df = pc.load_chemo_filtered()
cm_info = eac.build_cm_info(pd.read_csv(pc.CANCER_MAP_PATH))

patients = df['Newid'].unique()
# 先切 20% TEST
gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
rest_idx, test_idx = next(gss1.split(patients, groups=patients))
rest_pat = patients[rest_idx]
test_pat = set(patients[test_idx])
# 再把 rest 切成 train (75%) / calib (25%)  → 全體約 60/20/20
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
tr_idx, ca_idx = next(gss2.split(rest_pat, groups=rest_pat))
train_pat = set(rest_pat[tr_idx])
calib_pat = set(rest_pat[ca_idx])
print(f"\nPatients — train: {len(train_pat)} | calib: {len(calib_pat)} | TEST: {len(test_pat)}")

EXPECTED_MARKERS = sorted(df[df['Newid'].isin(train_pat)]['NAME'].dropna().unique().tolist())
top_drugs = pc.select_top_k_drugs(chemo_df, train_pat, pc.TOP_K_DRUGS)

data = eac.build_anchor_dataset(df, chemo_df, cm_info, top_drugs, EXPECTED_MARKERS,
                                horizon=HORIZON, gap=10, regimen_days=REGIMEN_DAYS,
                                regimen_mode=REGIMEN_MODE,
                                block_gap_hours=args.block_gap_hours,
                                max_block_days=args.max_block_days,
                                max_obs_days=MAX_OBS_DAYS)
feat_cols = [c for c in data.columns if c not in ('Newid', 'min_win_anc')]


def subset(pat_set):
    m = data['Newid'].isin(pat_set)
    return (data.loc[m, feat_cols].values,
            data.loc[m, 'min_win_anc'].values,
            data.loc[m, 'Newid'].values)


Xtr, anc_tr, _ = subset(train_pat)
Xca, anc_ca, pid_ca = subset(calib_pat)
Xte, anc_te, pid_te = subset(test_pat)
print(f"Anchors — train: {len(Xtr)} | calib: {len(Xca)} | TEST: {len(Xte)}")


# ---------------------------------------------------------------------------
# 2. Conformal helpers (Mondrian / class-conditional split conformal)
# ---------------------------------------------------------------------------
def conformal_q(scores, alpha):
    """split-conformal 的有限樣本 quantile (含 +1 修正)。"""
    n = len(scores)
    if n == 0:
        return 1.0
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level, method='higher'))


def first_anchor_mask(pids):
    """每位病人只留第一個 anchor → patient-level 保守 coverage 檢查。"""
    seen = set()
    keep = np.zeros(len(pids), dtype=bool)
    for i, p in enumerate(pids):
        if p not in seen:
            seen.add(p); keep[i] = True
    return keep


# ---------------------------------------------------------------------------
# 3. 每個 threshold: train → calibrate → conformal → selective
# ---------------------------------------------------------------------------
fig_cal, ax_cal = plt.subplots(1, 3, figsize=(15, 4.5))
summary = []
uq_preds = {}   # calib→test 校正後預測 (Figure 8)

for ti, thr in enumerate(THRESHOLDS):
    ytr = (anc_tr < thr).astype(int)
    yca = (anc_ca < thr).astype(int)
    yte = (anc_te < thr).astype(int)
    n_pos = int(ytr.sum()); n_neg = len(ytr) - n_pos
    if n_pos < 10 or yca.sum() < 5 or yte.sum() < 5:
        print(f"\n[{LABELS[thr]}] 陽性不足, 跳過")
        continue

    # --- train model ---
    params = dict(DEFAULT_PARAMS); params['scale_pos_weight'] = n_neg / n_pos
    cut = int(0.85 * len(Xtr))
    dtr = lgb.Dataset(Xtr[:cut], label=ytr[:cut])
    dva = lgb.Dataset(Xtr[cut:], label=ytr[cut:], reference=dtr)
    model = lgb.train(params, dtr, num_boost_round=1000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
    p_ca_raw = model.predict(Xca, num_iteration=model.best_iteration)
    p_te_raw = model.predict(Xte, num_iteration=model.best_iteration)

    # --- (1) Isotonic calibration (fit on calib, apply to test) ---
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(p_ca_raw, yca)
    p_te = iso.predict(p_te_raw)
    p_ca = iso.predict(p_ca_raw)

    brier_raw = brier_score_loss(yte, p_te_raw)
    brier_cal = brier_score_loss(yte, p_te)
    auc_te = roc_auc_score(yte, p_te)

    def ece(y, p, bins=10):
        edges = np.linspace(0, 1, bins + 1)
        idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
        e = 0.0
        for b in range(bins):
            m = idx == b
            if m.sum() == 0:
                continue
            e += (m.sum() / len(p)) * abs(p[m].mean() - y[m].mean())
        return e
    ece_raw, ece_cal = ece(yte, p_te_raw), ece(yte, p_te)

    # --- (2) Mondrian split conformal (用 calibrated 機率) ---
    s_pos = 1 - p_ca[yca == 1]            # 真陽性的 nonconformity (對 label=low)
    s_neg = p_ca[yca == 0]                # 真陰性的 nonconformity (對 label=not-low)
    q_pos = conformal_q(s_pos, ALPHA)
    q_neg = conformal_q(s_neg, ALPHA)

    in_low = (1 - p_te) <= q_pos          # 預測集是否含 "low"
    in_notlow = p_te <= q_neg             # 預測集是否含 "not-low"

    covered = np.where(yte == 1, in_low, in_notlow)
    cov_marg = covered.mean()
    cov_pos = covered[yte == 1].mean()
    cov_neg = covered[yte == 0].mean()

    singleton = in_low ^ in_notlow
    both = in_low & in_notlow
    empty = ~in_low & ~in_notlow
    abstain = both | empty
    set_size = in_low.astype(int) + in_notlow.astype(int)

    # patient-level 保守 coverage (每病人取第一個 anchor)
    fam = first_anchor_mask(pid_te)
    cov_patient = covered[fam].mean()

    # --- (3) Selective prediction: confident (singleton) 子集 ---
    conf = singleton
    # 在 confident-low singleton 上的 PPV
    pred_low_conf = in_low & ~in_notlow
    pred_notlow_conf = in_notlow & ~in_low
    ppv_conf = yte[pred_low_conf].mean() if pred_low_conf.sum() > 0 else float('nan')
    # 整體 (treat-all-positive at 0.5) 對照
    prev = yte.mean()

    # AUROC vs coverage 曲線 (依 |p-0.5| 由信心高到低保留)
    order = np.argsort(-np.abs(p_te - 0.5))
    cov_grid = np.linspace(0.2, 1.0, 9)
    auc_curve = []
    for c in cov_grid:
        k = max(int(c * len(p_te)), 10)
        idx = order[:k]
        yy = yte[idx]
        if yy.sum() >= 2 and (1 - yy).sum() >= 2:
            auc_curve.append(roc_auc_score(yy, p_te[idx]))
        else:
            auc_curve.append(np.nan)

    # --- reliability diagram (post-calibration) ---
    ax = ax_cal[ti]
    qs = np.linspace(0, 1, 11)
    edges = np.quantile(p_te, qs); edges[0] -= 1e-6; edges[-1] += 1e-6
    bi = np.clip(np.digitize(p_te, edges) - 1, 0, 9)
    mp, fp = [], []
    for b in range(10):
        m = bi == b
        if m.sum() == 0:
            continue
        mp.append(p_te[m].mean()); fp.append(yte[m].mean())
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='perfect')
    ax.plot(mp, fp, 'o-', lw=2, label=f'calibrated (Brier={brier_cal:.3f})')
    ax.set_title(LABELS[thr]); ax.set_xlabel('predicted prob'); ax.set_ylabel('observed freq')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    print(f"\n{'='*78}\n{LABELS[thr]}   (TEST n={len(yte)}, n_pos={int(yte.sum())}, prev={prev*100:.1f}%)\n{'='*78}")
    print(f"  [Calibration]  AUC={auc_te:.3f} | Brier {brier_raw:.4f} → {brier_cal:.4f} | "
          f"ECE {ece_raw:.4f} → {ece_cal:.4f}")
    print(f"  [Conformal α={ALPHA}]  target≥{(1-ALPHA)*100:.0f}%  | "
          f"marginal cov={cov_marg*100:.1f}%  (low={cov_pos*100:.1f}%, not-low={cov_neg*100:.1f}%)")
    print(f"                 patient-level 保守 cov={cov_patient*100:.1f}%  (每病人 1 anchor, n={int(fam.sum())})")
    print(f"  [Prediction sets]  singleton(confident)={singleton.mean()*100:.1f}% | "
          f"uncertain(abstain)={abstain.mean()*100:.1f}% (both={both.mean()*100:.1f}%, empty={empty.mean()*100:.1f}%)")
    print(f"                     set composition: {{low}}={pred_low_conf.mean()*100:.1f}% | "
          f"{{not-low}}={pred_notlow_conf.mean()*100:.1f}% | "
          f"{{low,not-low}}={both.mean()*100:.1f}% | {{}}={empty.mean()*100:.1f}%")
    print(f"  [Selective]  PPV on confident-low singletons = {ppv_conf*100:.1f}%  (vs prevalence {prev*100:.1f}%)")
    print(f"               AUROC@coverage: " +
          ", ".join(f"{int(c*100)}%:{a:.3f}" for c, a in zip(cov_grid[::2], auc_curve[::2]) if not np.isnan(a)))

    uq_preds[f'thr{thr}_p_cal'] = p_te
    uq_preds[f'thr{thr}_p_raw'] = p_te_raw
    uq_preds[f'thr{thr}_y'] = yte
    uq_preds[f'thr{thr}_ece_cal'] = np.array([ece_cal])

    summary.append({
        'thr': thr, 'auc': auc_te, 'brier_raw': brier_raw, 'brier_cal': brier_cal,
        'ece_raw': ece_raw, 'ece_cal': ece_cal, 'cov_marg': cov_marg,
        'cov_pos': cov_pos, 'cov_neg': cov_neg, 'cov_patient': cov_patient,
        'singleton': singleton.mean(), 'abstain': abstain.mean(),
        'ppv_conf': ppv_conf, 'prev': prev,
        'cov_grid': cov_grid, 'auc_curve': auc_curve,
    })

uq_npz = os.path.join(_HERE, f'uq_predictions_h{HORIZON}_{REGTAG}_seed{SEED}.npz')
np.savez_compressed(uq_npz, **uq_preds)
print(f"\n✓ UQ test predictions (isotonic fit on calib, applied to test) → {uq_npz}")

plt.tight_layout()
cal_png = os.path.join(_HERE, f'fig_uq_calibration_h{HORIZON}_{REGTAG}_seed{SEED}.png')
plt.savefig(cal_png, dpi=140, bbox_inches='tight'); plt.close()
print(f"\n✓ Calibration figure → {cal_png}")

# --- selective-prediction curve figure (<500 severe) ---
sev = next((s for s in summary if s['thr'] == 0.5), None)
if sev is not None:
    plt.figure(figsize=(6, 4.5))
    plt.plot(sev['cov_grid'] * 100, sev['auc_curve'], 'o-', lw=2)
    plt.xlabel('Coverage (% of anchors kept, most-confident first)')
    plt.ylabel('AUROC on retained anchors')
    plt.title('Selective prediction — ANC<500 (severe)')
    plt.grid(alpha=0.3)
    sel_png = os.path.join(_HERE, f'fig_uq_selective_h{HORIZON}_{REGTAG}_seed{SEED}.png')
    plt.savefig(sel_png, dpi=140, bbox_inches='tight'); plt.close()
    print(f"✓ Selective-prediction figure → {sel_png}")

print(f"\n說明:")
print(f"  - Conformal 若 marginal/per-class coverage ≈ {(1-ALPHA)*100:.0f}% → 保證有效。")
print(f"  - patient-level 保守 cov 若 ≈ 一樣 → 不可交換造成的樂觀很小; 若明顯偏低 → 要在 paper 揭露。")
print(f"  - 'uncertain(abstain)' = 模型說『我不確定 → 建議抽 CBC / 醫師複核』的比例。")
print(f"  - confident-low PPV 應 > prevalence → 模型在它有把握時更準, 支持 selective deployment。")
