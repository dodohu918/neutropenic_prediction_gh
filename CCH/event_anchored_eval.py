"""
event_anchored_eval.py — Event-anchored (以化療週期起點為錨點) 嚴重 neutropenia 預測

設計對齊 Wiberg 2021 (JCO CCI): 在「化療週期起點」評估「未來 N 天內」是否發生 ANC<500。
解決舊版 equal-duration bin 把 chemo→nadir timing 抹平的問題。

跟舊 bin 版的差異:
  - 預測單位: 從「等長 bin 的下一格」→「每個化療週期起點」(anchor)
  - 每個 anchor = 一筆 tabular 樣本 (不再有 7 個 resolution、不再有 sliding window)
  - 標籤: (anchor, min(anchor+horizon, 下一個 anchor)] 窗內任一 ANC < threshold
  - 特徵: 只用 anchor「之前」的資料 (嚴防 leakage), 外加「本週期起始時已知的方案」

鎖定設定:
  - Cohort ③: 排除 leukemia/myeloma (bone marrow primary), 保留 lymphoma + solid + unknown
  - Default LightGBM 參數 (tuning 已證實沒幫助)
  - 3 seeds (42/43/44), 同一套評估指標

用法 (從 repo root 或 CCH/ 執行都可):
  python CCH/event_anchored_eval.py                  # horizon 28, seed 42
  python CCH/event_anchored_eval.py --horizon 21
  SPLIT_SEED=43 python CCH/event_anchored_eval.py
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

# --- 讓本檔不論從哪裡執行都能 import pipeline_common (在 CCH/) ---
# pipeline_common 內部用絕對路徑找資料, 所以這裡不需要 os.chdir。
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import pipeline_common as pc
import eventanchor_common as eac


# ---------------------------------------------------------------------------
# Args / config
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--horizon', type=int, default=28, help='預測窗長度 (天); 預設 28, 可切 21')
parser.add_argument('--gap', type=int, default=10, help='相隔幾天算新週期 (天); 預設 10')
parser.add_argument('--regimen-days', type=int, default=0,
                    help='「本週期起始方案」採計 anchor 當天 + 後 N 個日曆天的給藥。'
                         '0 = 只用 anchor 當天 (primary, 無 future leakage); '
                         '3 = 含後 3 天 (sensitivity; 若資料為實際給藥則 day1-3 屬未來資訊)')
parser.add_argument('--regimen-mode', choices=['day', 'block'], default='day',
                    help="day = anchor 當天(+regimen-days)的方案 (frozen 預設); "
                         "block = 整個 leading 連續 block, decision time 移到 block 結束 "
                         "(抓得到 multi-day induction 如 H&N 的 Taxotere d1 + Cisplatin d2, 仍無 leakage)")
parser.add_argument('--block-gap-hours', type=float, default=36.0,
                    help='block 模式: 連續給藥的最大間隔 (小時); 預設 36 (涵蓋隔天)')
parser.add_argument('--max-block-days', type=int, default=7, help='block 模式: block 最長天數 (防呆)')
parser.add_argument('--max-obs-days', type=int, default=28,
                    help='觀察窗 (天); 特徵只取 T 前 N 天內的 ANC/lab 資料. None = 全程歷史. 預設 28.')
parser.add_argument('--heme-mode', default='leukemia_only',
                    choices=['none', 'all', 'leukemia_only', 'leukemia_only_adj'],
                    help="cohort 排除模式; leukemia_only_adj = 額外排掉淋巴瘤判讀 ①(實為白血病)/②(骨髓侵犯)")
parser.add_argument('--seed', type=int, default=int(os.environ.get('SPLIT_SEED', 42)))
args = parser.parse_args()

HORIZON = args.horizon
GAP = args.gap
REGIMEN_DAYS = args.regimen_days
REGIMEN_MODE = args.regimen_mode
HEME_MODE = args.heme_mode
SEED = args.seed
MAX_OBS_DAYS = args.max_obs_days
THRESHOLDS = [1.5, 1.0, 0.5]
LABELS = {1.5: '<1500 (mild)', 1.0: '<1000 (moderate)', 0.5: '<500 (severe/FN)'}
NO_PREV_CYCLE_CAP = 365.0
_REGTAG = (f'reg{REGIMEN_DAYS}' if REGIMEN_MODE == 'day'
           else f'block{args.block_gap_hours:g}h')
# 非預設 cohort 才加 tag, 避免覆蓋既有 noleuk 檔名
_COHORT = '' if HEME_MODE == 'leukemia_only' else f'_{pc._TAG_MAP[HEME_MODE]}'
PRED_OUT = os.path.join(_HERE, f'predictions_eventanchored_h{HORIZON}_{_REGTAG}{_COHORT}_seed{SEED}.npz')

print("=" * 80)
print(f"Event-anchored eval | horizon={HORIZON}d | cycle-gap={GAP}d | "
      f"regimen-mode={REGIMEN_MODE} ({_REGTAG}) | heme={HEME_MODE} | max_obs={MAX_OBS_DAYS}d | seed={SEED}")
print("=" * 80)


# ---------------------------------------------------------------------------
# 1. 載入資料 (cohort)
# ---------------------------------------------------------------------------
df = pc.load_lab_df(exclude_heme_mode=HEME_MODE)
chemo_df = pc.load_chemo_filtered()

# demographics / cancer map
cm = pd.read_csv(pc.CANCER_MAP_PATH)
cm_info = eac.build_cm_info(cm)


# ---------------------------------------------------------------------------
# 2. Train/TEST split (病人層級, 跟 bin 版同一個 GroupShuffleSplit)
# ---------------------------------------------------------------------------
unique_patients = df['Newid'].unique()
gss = GroupShuffleSplit(test_size=0.2, random_state=SEED)
tr_idx, te_idx = next(gss.split(unique_patients, groups=unique_patients))
train_patients = set(unique_patients[tr_idx])
test_patients = set(unique_patients[te_idx])
print(f"\nTrain patients: {len(train_patients)} | TEST patients: {len(test_patients)}")

EXPECTED_MARKERS = sorted(
    df[df['Newid'].isin(train_patients)]['NAME'].dropna().unique().tolist())
ANC_NAME = 'Absolute Neutrophil Count'
OTHER_MARKERS = [m for m in EXPECTED_MARKERS if m != ANC_NAME]
top_drugs = pc.select_top_k_drugs(chemo_df, train_patients, pc.TOP_K_DRUGS)
print(f"Markers ({len(EXPECTED_MARKERS)}): {EXPECTED_MARKERS}")
print(f"Top-{pc.TOP_K_DRUGS} drugs: {top_drugs}")


# ---------------------------------------------------------------------------
# 3. 建 anchor-level dataset (共用 eventanchor_common, single source of truth)
# ---------------------------------------------------------------------------
data = eac.build_anchor_dataset(df, chemo_df, cm_info, top_drugs, EXPECTED_MARKERS,
                                horizon=HORIZON, gap=GAP, regimen_days=REGIMEN_DAYS,
                                regimen_mode=REGIMEN_MODE,
                                block_gap_hours=args.block_gap_hours,
                                max_block_days=args.max_block_days,
                                max_obs_days=MAX_OBS_DAYS)

feature_cols = [c for c in data.columns if c not in ('Newid', 'min_win_anc')]
print(f"特徵數: {len(feature_cols)}")

train_mask = data['Newid'].isin(train_patients)
test_mask = data['Newid'].isin(test_patients)
Xtr_all = data.loc[train_mask, feature_cols].values
Xte = data.loc[test_mask, feature_cols].values
anc_tr = data.loc[train_mask, 'min_win_anc'].values
anc_te = data.loc[test_mask, 'min_win_anc'].values
pid_te = data.loc[test_mask, 'Newid'].values

print(f"\nTrain anchors: {len(Xtr_all)} | TEST anchors: {len(Xte)}")


# ---------------------------------------------------------------------------
# 4. 對比舊 bin 版的 trainable 數量 (參考數字來自 cohort ③ 的 run log)
# ---------------------------------------------------------------------------
print(f"\n{'='*80}")
print("=== Trainable 數量對比 (event-anchored vs 舊 bin 版) ===")
print(f"{'='*80}")
print(f"  舊 bin 版 (7 resolutions × sliding window, cohort ③):")
print(f"      ~58,553 train sequences / ~13,868 test sequences  (高度冗餘: 同病人重複)")
print(f"  新 event-anchored (本檔):")
print(f"      {len(Xtr_all)} train anchors / {len(Xte)} test anchors  (每個化療週期 1 筆, 低冗餘)")
print(f"  → 樣本數變少但每筆對齊化療事件、彼此較獨立。關鍵看下面各 threshold 的陽性數。")


# ---------------------------------------------------------------------------
# 5. 訓練 + 評估 (default LightGBM, 每個 threshold 一個 binary classifier)
# ---------------------------------------------------------------------------
def binary_metrics(min_anc_true, prob, thr):
    y = (np.asarray(min_anc_true) < thr).astype(int)
    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    if n_pos < 2 or n_neg < 2:
        return None
    auc = roc_auc_score(y, prob)
    pr = average_precision_score(y, prob)
    fpr, tpr, _ = roc_curve(y, prob)
    valid = (1 - fpr) >= 0.8
    sens_at_spec80 = float(tpr[valid].max()) if valid.any() else float('nan')
    m = tpr >= 0.8
    if m.any():
        idx = np.where(m)[0]
        b = idx[np.argmin(fpr[idx])]
        tp = tpr[b] * n_pos; fp = fpr[b] * n_neg
        ppv80 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    else:
        ppv80 = None
    return {'n': len(y), 'n_pos': n_pos, 'prev': n_pos / len(y), 'auc': auc,
            'pr_auc': pr, 'sens_at_spec80': sens_at_spec80, 'ppv_at_sens80': ppv80}


DEFAULT_PARAMS = {
    'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05,
    'num_leaves': 31, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
    'bagging_freq': 5, 'verbose': -1,
}

# train 內部切 85/15 做 early stopping (test 已分開)
cut = int(0.85 * len(Xtr_all))
save_dict = {}
importances = {}
models = {}
print(f"\n{'Threshold':<22}{'n_test':<8}{'n_pos':<8}{'prev':<8}{'AUC':<9}{'PR-AUC':<9}{'Sens@Spec80':<13}{'PPV@Sens0.8':<12}")
print("-" * 90)
for thr in THRESHOLDS:
    ytr = (anc_tr < thr).astype(int)
    n_pos = int(ytr.sum()); n_neg = len(ytr) - n_pos
    if n_pos < 10 or n_neg < 10:
        print(f"{LABELS[thr]:<22} 跳過 (n_pos={n_pos})")
        continue
    spw = float(n_neg) / float(n_pos)
    params = dict(DEFAULT_PARAMS); params['scale_pos_weight'] = spw
    dtr = lgb.Dataset(Xtr_all[:cut], label=ytr[:cut])
    dva = lgb.Dataset(Xtr_all[cut:], label=ytr[cut:], reference=dtr)
    model = lgb.train(params, dtr, num_boost_round=1000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
    prob = model.predict(Xte, num_iteration=model.best_iteration)
    importances[thr] = model.feature_importance(importance_type='gain')
    models[thr] = model
    r = binary_metrics(anc_te, prob, thr)
    if r is None:
        print(f"{LABELS[thr]:<22} 測試集陽性不足")
        continue
    print(f"{LABELS[thr]:<22}{r['n']:<8}{r['n_pos']:<8}{r['prev']*100:<7.1f}%"
          f"{r['auc']:<9.3f}{r['pr_auc']:<9.3f}{r['sens_at_spec80']:<13.3f}"
          f"{(r['ppv_at_sens80']*100 if r['ppv_at_sens80'] is not None else float('nan')):<11.1f}%")
    save_dict[f'thr{thr}_pid'] = pid_te
    save_dict[f'thr{thr}_pred'] = prob
    save_dict[f'thr{thr}_truth'] = anc_te

# --- Feature importance (gain) — leakage sanity check + reviewer 需求 ---
print(f"\n{'='*80}")
print("=== Feature importance (gain) — 確認驅動的是臨床合理特徵 (非 leakage) ===")
print(f"{'='*80}")
for thr in THRESHOLDS:
    if thr not in importances:
        continue
    imp = importances[thr]
    order = np.argsort(imp)[::-1][:12]
    total = imp.sum() if imp.sum() > 0 else 1.0
    print(f"\n{LABELS[thr]} — Top 12:")
    for j in order:
        print(f"    {feature_cols[j]:28s} {imp[j]/total*100:5.1f}%")

save_dict['_thresholds'] = np.array(THRESHOLDS)
save_dict['_horizon'] = np.array([HORIZON])
save_dict['_seed'] = np.array([SEED])
np.savez_compressed(PRED_OUT, **save_dict)
print(f"\n✓ 預測存到 {PRED_OUT}")
print(f"  (對比舊 bin 版 cohort ③ headline: <500 0.747±0.016, <1000 0.761±0.008, <1500 0.756±0.010)")


# --- Model complexity (LightGBM 樹規模 ≈ 參數量) ---
print(f"\n{'='*80}")
print("=== Model complexity (LightGBM 樹規模 ≈ 參數量) ===")
print(f"{'='*80}")
print(f"{'Threshold':<22}{'trees':<8}{'leaves':<10}{'splits':<10}{'~params':<10}{'n_feat':<8}")
print("-" * 68)
_tot_leaves = 0; _tot_params = 0
for thr in THRESHOLDS:
    if thr not in models:
        continue
    m = models[thr]
    tdf = m.trees_to_dataframe()
    n_trees = m.num_trees()
    n_leaves = int(tdf['split_feature'].isna().sum())
    n_splits = int(tdf['split_feature'].notna().sum())
    approx = n_leaves + 2 * n_splits            # 葉值 + 每分裂(feature+threshold)
    _tot_leaves += n_leaves; _tot_params += approx
    print(f"{LABELS[thr]:<22}{n_trees:<8}{n_leaves:<10}{n_splits:<10}{approx:<10}{m.num_feature():<8}")
print("-" * 68)
print(f"{'3 classifiers 合計':<22}{'':<8}{_tot_leaves:<10}{'':<10}{_tot_params:<10}")
print("註: leaves = 學到的輸出值數(最接近參數量); ~params = leaves + 2*splits (每分裂存 feature+threshold)。")
