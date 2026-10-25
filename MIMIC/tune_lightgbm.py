"""
tune_lightgbm.py — LightGBM 超參數調校 (Optuna + GroupKFold + 一次性 held-out test)

方法學紀律 (這是這支 script 的重點):
  1. 病人先用 GroupShuffleSplit(random_state=SEED) 切成 train / TEST — 跟主 eval 完全一樣。
     TEST 在「整個搜尋過程」全程不碰, 只在最後用最佳參數評估「一次」。
  2. 在 TRAIN 內用 GroupKFold(on Newid) 做 CV → Optuna 用 CV 的 mean AUC 找參數。
     病人不會跨 fold 洩漏。
  3. 最後報兩個數字:
       - CV-best AUC  (搜尋過程挑到的最佳, 帶有 selection optimism, 偏高)
       - TEST AUC     (一次性 held-out, 誠實數字)
     兩者的差距 = selection overfitting 的幅度, 直接寫進 paper 才誠實。
  4. 只調超參數, 不動 feature / cohort / split → 可重現、可寫進 Methods。

把所有 resolution 的 train sequences「pool 在一起」調「一組」共用超參數 (不是每個
resolution / threshold 各調一組 — 那會 21× 放大 selection overfitting 且難辯護)。
LightGBM 是 tree-based, scale-invariant, 所以這裡不做 z-score 標準化 (對結果無影響)。

用法:
  cd anthony
  python tune_lightgbm.py                         # 預設: cohort ③ + drop G-CSF, threshold<1000, 100 trials
  python tune_lightgbm.py --threshold 0.5 --n-trials 200
  python tune_lightgbm.py --heme-mode all --keep-gcsf   # 換 cohort 設定

輸出: best_params_<tag>_thr<thr>_seed<seed>.json
"""
import argparse
import json
import warnings
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import optuna

import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
import pipeline_common as pc

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42,
                    help='train/test split seed (跟主 eval 對齊; default 42)')
parser.add_argument('--heme-mode', type=str, default='leukemia_only',
                    choices=['none', 'all', 'leukemia_only'])
parser.add_argument('--keep-gcsf', action='store_true',
                    help='保留 G-CSF features (預設會 drop)')
parser.add_argument('--threshold', type=float, default=1.0,
                    help='調校的目標 threshold (1.5/1.0/0.5; default 1.0 moderate)')
parser.add_argument('--n-trials', type=int, default=100)
parser.add_argument('--n-folds', type=int, default=5)
parser.add_argument('--optuna-seed', type=int, default=2025,
                    help='Optuna TPE sampler 的 seed (可重現)')
args = parser.parse_args()

DROP_GCSF = not args.keep_gcsf
THR = args.threshold
TAG = pc._TAG_MAP[args.heme_mode] + ('_nogcsf' if DROP_GCSF else '')
OUT_JSON = f'best_params_{TAG}_thr{THR}_seed{args.seed}.json'

print("=" * 78)
print(f"LightGBM tuning | cohort={TAG} | threshold ANC<{THR*1000:.0f} | "
      f"split_seed={args.seed} | n_trials={args.n_trials} | folds={args.n_folds}")
print("=" * 78)


# ---------------------------------------------------------------------------
# 1. 載入資料 + cohort 排除 + 一次性 train/TEST split
# ---------------------------------------------------------------------------
df = pc.load_lab_df(exclude_heme_mode=args.heme_mode)
chemo_df = pc.load_chemo_filtered(drop_gcsf=DROP_GCSF)
chemo_lookup = pc.build_chemo_lookup(chemo_df)

unique_patients = df['Newid'].unique()
gss = GroupShuffleSplit(test_size=0.2, random_state=args.seed)
train_idx, test_idx = next(gss.split(unique_patients, groups=unique_patients))
train_patients = unique_patients[train_idx]
test_patients = unique_patients[test_idx]
print(f"\nTrain patients: {len(train_patients)} | TEST patients (held-out): {len(test_patients)}")

EXPECTED_MARKERS = sorted(
    df[df['Newid'].isin(train_patients)]['NAME'].dropna().unique().tolist())
top_drugs = pc.select_top_k_drugs(chemo_df, set(train_patients), pc.TOP_K_DRUGS)
print(f"Top-{pc.TOP_K_DRUGS} drugs: {top_drugs}")

df_train = df[df['Newid'].isin(train_patients)]
df_test = df[df['Newid'].isin(test_patients)]


# ---------------------------------------------------------------------------
# 2. 產生所有 resolution 的 sequences (train pool + 保留 test per-resolution)
# ---------------------------------------------------------------------------
def build_sequences(df_sub):
    """回傳 {res: (X_flat, y, pid)};  X_flat = (N, window*nfeat)"""
    out = {}
    for res in pc.RESOLUTIONS:
        Xf = pc.generate_features_for_resolution(
            df_sub, res, EXPECTED_MARKERS, chemo_lookup, top_drugs)
        if Xf is None:
            continue
        feat_cols = [c for c in Xf.columns if c not in ('Newid', 'current_bin', 'target')]
        Xs, ys, pids = pc.generate_sequences(Xf, feat_cols, pc.WINDOW_SIZE)
        if len(Xs) == 0:
            continue
        Xs = np.nan_to_num(Xs, nan=0.0).reshape(len(Xs), -1)  # flatten for GBM
        out[res] = (Xs, ys, pids)
    return out

print("\n產生 train sequences (所有 resolution)...")
train_seqs = build_sequences(df_train)
print("產生 TEST sequences (所有 resolution, 全程不參與調校)...")
test_seqs = build_sequences(df_test)

# Pool train across resolutions
X_pool = np.concatenate([train_seqs[r][0] for r in train_seqs], axis=0)
y_pool = np.concatenate([train_seqs[r][1] for r in train_seqs], axis=0)
g_pool = np.concatenate([train_seqs[r][2] for r in train_seqs], axis=0)
y_pool_bin = (y_pool < THR).astype(int)
print(f"\nPooled train sequences: {X_pool.shape}, positives(<{THR})={int(y_pool_bin.sum())} "
      f"({y_pool_bin.mean()*100:.1f}%)")

if y_pool_bin.sum() < 50:
    print(f"⚠️ 警告: 陽性數只有 {int(y_pool_bin.sum())}, 這個 threshold 的調校會很不穩 — "
          f"考慮改 --threshold 1.0 或 1.5")


# ---------------------------------------------------------------------------
# 3. Optuna objective: GroupKFold CV mean AUC
# ---------------------------------------------------------------------------
BASE_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'verbose': -1,
    'bagging_freq': 5,
}
gkf = GroupKFold(n_splits=args.n_folds)
FOLDS = list(gkf.split(X_pool, y_pool_bin, groups=g_pool))


def objective(trial):
    params = dict(BASE_PARAMS)
    params.update({
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves':        trial.suggest_int('num_leaves', 15, 255),
        'max_depth':         trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 200),
        'feature_fraction':  trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction':  trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'lambda_l1':         trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True),
        'lambda_l2':         trial.suggest_float('lambda_l2', 1e-3, 10.0, log=True),
        'min_split_gain':    trial.suggest_float('min_split_gain', 0.0, 0.5),
    })
    fold_aucs = []
    for tr, va in FOLDS:
        ytr = y_pool_bin[tr]
        n_pos = int(ytr.sum())
        n_neg = len(ytr) - n_pos
        if n_pos < 5 or n_neg < 5:
            continue
        params['scale_pos_weight'] = float(n_neg) / float(n_pos)  # 只用 fold-train 算, 不洩漏
        dtr = lgb.Dataset(X_pool[tr], label=ytr)
        dva = lgb.Dataset(X_pool[va], label=y_pool_bin[va], reference=dtr)
        model = lgb.train(params, dtr, num_boost_round=1000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(30, verbose=False)])
        pred = model.predict(X_pool[va], num_iteration=model.best_iteration)
        yva = y_pool_bin[va]
        if yva.sum() >= 2 and (1 - yva).sum() >= 2:
            fold_aucs.append(roc_auc_score(yva, pred))
    return float(np.mean(fold_aucs)) if fold_aucs else 0.5


print(f"\n開始 Optuna 搜尋 ({args.n_trials} trials, TPE seed={args.optuna_seed})...")
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction='maximize',
                            sampler=optuna.samplers.TPESampler(seed=args.optuna_seed))
study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

cv_best_auc = study.best_value
best_params = dict(BASE_PARAMS)
best_params.update(study.best_params)
print(f"\n=== Optuna 完成 ===")
print(f"CV-best AUC (帶 selection optimism, 偏高): {cv_best_auc:.4f}")
print(f"Best params: {json.dumps(study.best_params, indent=2)}")


# ---------------------------------------------------------------------------
# 4. 一次性 held-out TEST 評估 (誠實數字)
#    用 best_params 在每個 resolution 的 full train 上訓練, 預測 test, union 起來算 AUC。
#    (跟主 eval 的 union sequence-level 一致, 可直接比較)
# ---------------------------------------------------------------------------
def eval_on_test(params):
    all_truth, all_pred = [], []
    for res in train_seqs:
        if res not in test_seqs:
            continue
        Xtr, ytr_cont, _ = train_seqs[res]
        Xte, yte_cont, _ = test_seqs[res]
        ytr = (ytr_cont < THR).astype(int)
        n_pos = int(ytr.sum()); n_neg = len(ytr) - n_pos
        if n_pos < 5 or n_neg < 5:
            continue
        p = dict(params)
        p['scale_pos_weight'] = float(n_neg) / float(n_pos)
        # 用 train 內部切一小塊 (grouped 不需要, 因為 test 已分開) 做 early stopping
        cut = int(0.85 * len(Xtr))
        dtr = lgb.Dataset(Xtr[:cut], label=ytr[:cut])
        dva = lgb.Dataset(Xtr[cut:], label=ytr[cut:], reference=dtr)
        model = lgb.train(p, dtr, num_boost_round=1000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(30, verbose=False)])
        pred = model.predict(Xte, num_iteration=model.best_iteration)
        all_truth.append((yte_cont < THR).astype(int))
        all_pred.append(pred)
    yt = np.concatenate(all_truth)
    yp = np.concatenate(all_pred)
    return roc_auc_score(yt, yp), len(yt), int(yt.sum())


test_auc, n_test, n_pos_test = eval_on_test(best_params)
# baseline (目前主 eval 用的 default params) 在 test 上的數字, 看 tuning 到底賺多少
DEFAULT_PARAMS = dict(BASE_PARAMS)
DEFAULT_PARAMS.update({'learning_rate': 0.05, 'num_leaves': 31,
                       'feature_fraction': 0.8, 'bagging_fraction': 0.8})
default_test_auc, _, _ = eval_on_test(DEFAULT_PARAMS)

print(f"\n{'='*78}")
print(f"=== 誠實對比 (threshold ANC<{THR*1000:.0f}, held-out TEST n={n_test}, n_pos={n_pos_test}) ===")
print(f"{'='*78}")
print(f"  Default params  → TEST union AUC = {default_test_auc:.4f}   (目前主 eval 用的)")
print(f"  Tuned  params   → TEST union AUC = {test_auc:.4f}")
print(f"  Tuning 實際淨賺  → {test_auc - default_test_auc:+.4f}")
print(f"  ----")
print(f"  CV-best AUC     = {cv_best_auc:.4f}  (搜尋過程, 偏高)")
print(f"  TEST AUC        = {test_auc:.4f}")
print(f"  Selection gap   = {cv_best_auc - test_auc:+.4f}  "
      f"(>0 表示 CV 比 test 樂觀, 即 selection overfitting 的幅度)")


# ---------------------------------------------------------------------------
# 5. 存最佳參數
# ---------------------------------------------------------------------------
result = {
    'cohort_tag': TAG,
    'threshold': THR,
    'split_seed': args.seed,
    'n_trials': args.n_trials,
    'n_folds': args.n_folds,
    'cv_best_auc': cv_best_auc,
    'test_auc_tuned': test_auc,
    'test_auc_default': default_test_auc,
    'tuning_gain': test_auc - default_test_auc,
    'selection_gap': cv_best_auc - test_auc,
    'best_params': best_params,
}
with open(OUT_JSON, 'w') as f:
    json.dump(result, f, indent=2)
print(f"\n✓ 最佳參數存到 {OUT_JSON}")
print(f"  要套用: 把 best_params 套用到 LightGBM 的 train params, 然後重跑 3 seeds。")
print(f"\n⚠️ 報 paper 時請用 TEST AUC ({test_auc:.4f}) 而非 CV-best AUC ({cv_best_auc:.4f})。")
