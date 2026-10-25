"""
mlp_vs_lgb_eventanchored.py — 神經網路 (MLP) vs LightGBM, 同一份 event-anchored 資料

說明: event-anchored 是「每個 anchor 一列、50 特徵」的 tabular 資料, 沒有序列維度,
      所以 LSTM 在這裡會退化成 MLP (length-1 sequence)。故神經 baseline 用 MLP。
      兩個模型用「完全相同」的 features / 2-way split / threshold, 公平對比 AUROC。

用法 (從 MIMIC/ 執行):
  python mlp_vs_lgb_eventanchored.py --seed 42
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import pipeline_common as pc
import eventanchor_common as eac

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--horizon', type=int, default=28)
parser.add_argument('--regimen-days', type=int, default=0)
args = parser.parse_args()
SEED, HORIZON, REG = args.seed, args.horizon, args.regimen_days
THRESHOLDS = [1.5, 1.0, 0.5]
LABELS = {1.5: '<1500 (mild)', 1.0: '<1000 (moderate)', 0.5: '<500 (severe/FN)'}

print(f"MLP vs LightGBM | event-anchored h{HORIZON} reg{REG} seed{SEED}")

# --- 資料 (cohort ③ + drop G-CSF) + 2-way split (跟 frozen headline 一致) ---
df = pc.load_lab_df(exclude_heme_mode='leukemia_only')
chemo_df = pc.load_chemo_filtered(drop_gcsf=True)
cm_info = eac.build_cm_info(pd.read_csv(pc.CANCER_MAP_PATH))

pats = df['Newid'].unique()
tr_idx, te_idx = next(GroupShuffleSplit(test_size=0.2, random_state=SEED).split(pats, groups=pats))
train_pat, test_pat = set(pats[tr_idx]), set(pats[te_idx])
markers = sorted(df[df['Newid'].isin(train_pat)]['NAME'].dropna().unique().tolist())
top_drugs = pc.select_top_k_drugs(chemo_df, train_pat, pc.TOP_K_DRUGS)

data = eac.build_anchor_dataset(df, chemo_df, cm_info, top_drugs, markers,
                                horizon=HORIZON, gap=10, regimen_days=REG)
fc = [c for c in data.columns if c not in ('Newid', 'min_win_anc')]
tr = data['Newid'].isin(train_pat); te = data['Newid'].isin(test_pat)
Xtr, Xte = data.loc[tr, fc].values, data.loc[te, fc].values
anc_tr, anc_te = data.loc[tr, 'min_win_anc'].values, data.loc[te, 'min_win_anc'].values

# NN 需要標準化 (用 train 算); LightGBM 不需要但用同一份不影響 tree
sc = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

LGB_PARAMS = {'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05,
              'num_leaves': 31, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
              'bagging_freq': 5, 'verbose': -1}

print(f"\n{'Threshold':<22}{'n_pos':<8}{'LGB AUC':<10}{'MLP AUC':<10}{'LGB PR':<9}{'MLP PR':<9}")
print("-" * 70)
for thr in THRESHOLDS:
    ytr = (anc_tr < thr).astype(int); yte = (anc_te < thr).astype(int)
    n_pos = int(ytr.sum()); n_neg = len(ytr) - n_pos
    if n_pos < 10:
        continue

    # --- LightGBM (class-weighted, 跟 frozen 設定相同) ---
    cut = int(0.85 * len(Xtr))
    p = dict(LGB_PARAMS); p['scale_pos_weight'] = n_neg / n_pos
    dtr = lgb.Dataset(Xtr[:cut], label=ytr[:cut])
    dva = lgb.Dataset(Xtr[cut:], label=ytr[cut:], reference=dtr)
    m_lgb = lgb.train(p, dtr, num_boost_round=1000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
    p_lgb = m_lgb.predict(Xte, num_iteration=m_lgb.best_iteration)

    # --- MLP (神經 baseline; 用 class_weight='balanced' 補不平衡) ---
    # MLPClassifier 不吃 class_weight → 用手動 minority 過採樣近似 class weighting
    pos_idx = np.where(ytr == 1)[0]
    reps = max(int(round(n_neg / n_pos)) - 1, 0)
    idx = np.concatenate([np.arange(len(ytr))] + [pos_idx] * reps) if reps > 0 else np.arange(len(ytr))
    rng = np.random.RandomState(SEED); rng.shuffle(idx)
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', alpha=1e-3,
                        early_stopping=True, n_iter_no_change=15, max_iter=300,
                        random_state=SEED)
    mlp.fit(Xtr_s[idx], ytr[idx])
    p_mlp = mlp.predict_proba(Xte_s)[:, 1]

    print(f"{LABELS[thr]:<22}{int(yte.sum()):<8}"
          f"{roc_auc_score(yte, p_lgb):<10.3f}{roc_auc_score(yte, p_mlp):<10.3f}"
          f"{average_precision_score(yte, p_lgb):<9.3f}{average_precision_score(yte, p_mlp):<9.3f}")

print("\n註: MLP = 神經網路 baseline (LSTM 在無序列維度的 tabular 上退化為 MLP)。")
