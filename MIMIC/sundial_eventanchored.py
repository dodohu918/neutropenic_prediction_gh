"""
sundial_eventanchored.py — 用 Sundial (TS foundation model) 在 event-anchored 設定上

Sundial 是 univariate 生成式預測模型, 吃不進 covariate。所以這裡的用法是:
  Sundial 從「anchor 之前的 ANC 每日序列」zero-shot 預測未來 28 天軌跡 (機率分布),
  把軌跡轉成幾個特徵 (P(ANC<thr)、預測 nadir 等), 再餵進 LightGBM (late fusion)。

三個 mode:
  --mode features   : (重, 建議 GPU) 對「所有 anchor」算 Sundial 預測特徵, 存到 cache。
                      此步 seed-independent, 只需跑一次。
  --mode benchmark  : (輕) 純 Sundial — 直接用 P(low) 當風險分數算 AUROC (對照組 A)。
  --mode hybrid     : (輕) LightGBM: covariate-only vs covariate+Sundial 特徵, 比 AUROC (對照組 B)。

資料: cohort ③ + drop G-CSF, horizon 28, daily 網格, lookback 預設 112 天。
對照基準: 現有 LightGBM event-anchored <500 AUROC ≈ 0.845。

典型流程:
  # 1. (EC2 GPU) 算 Sundial 特徵一次
  python sundial_eventanchored.py --mode features --device cuda
  # 2. (本機) 用 cache 跑 hybrid / benchmark
  python sundial_eventanchored.py --mode hybrid --seed 42
  python sundial_eventanchored.py --mode benchmark --seed 42
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import pipeline_common as pc
import eventanchor_common as eac

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['features', 'benchmark', 'hybrid'], required=True)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--horizon', type=int, default=28, help='預測天數 (= 標籤窗)')
parser.add_argument('--lookback-days', type=int, default=112, help='Sundial context 天數 (daily 網格)')
parser.add_argument('--num-samples', type=int, default=20, help='Sundial 生成軌跡數')
parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'])
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--model-id', default='thuml/sundial-base-128m')
parser.add_argument('--regimen-mode', choices=['day', 'block'], default='block',
                    help='對齊 primary; block = decision time 為 block 結束 (預設)')
parser.add_argument('--block-gap-hours', type=float, default=36.0)
parser.add_argument('--max-block-days', type=int, default=7)
parser.add_argument('--heme-mode', default='leukemia_only',
                    choices=['none', 'all', 'leukemia_only', 'leukemia_only_adj'],
                    help='對齊 cohort; leukemia_only_adj = 最終 primary (額外排淋巴瘤判讀 ①/②)')
parser.add_argument('--cache', default=None,
                    help='預設依 heme-mode 自動命名 (adj → sundial_features_h28_block36h_noleukadj.parquet)')
parser.add_argument('--limit', type=int, default=0, help='只算前 N 個 anchor (除錯用; 0=全部)')
args = parser.parse_args()

# cache 預設名稱: 非 leukemia_only 的 cohort 加 tag, 避免覆蓋
if args.cache is None:
    _cohort = '' if args.heme_mode == 'leukemia_only' else f'_{pc._TAG_MAP[args.heme_mode]}'
    args.cache = os.path.join(_HERE, f'sundial_features_h28_block36h{_cohort}.parquet')

THRESHOLDS = [1.5, 1.0, 0.5]
LABELS = {1.5: '<1500', 1.0: '<1000', 0.5: '<500'}
DAY = np.timedelta64(1, 'D')


# ---------------------------------------------------------------------------
# 共用: 載入資料 + 建「canonical anchor 表」(seed-independent)
# ---------------------------------------------------------------------------
def load_anchor_table():
    """回傳 (anchor_df, anc_by_pid)。anchor_df 有 Newid, anchor_time, min_win_anc。"""
    df = pc.load_lab_df(exclude_heme_mode=args.heme_mode)
    chemo_df = pc.load_chemo_filtered(drop_gcsf=True)
    cm_info = eac.build_cm_info(pd.read_csv(pc.CANCER_MAP_PATH))
    # top_drugs 用全 cohort 算 (只為了拿到一致的 anchor 列; 列集合與 top_drugs 無關)
    all_pat = set(df['Newid'].unique())
    markers = sorted(df['NAME'].dropna().unique().tolist())
    top_drugs = pc.select_top_k_drugs(chemo_df, all_pat, pc.TOP_K_DRUGS)
    data = eac.build_anchor_dataset(df, chemo_df, cm_info, top_drugs, markers,
                                    horizon=args.horizon, gap=10, regimen_days=0,
                                    regimen_mode=args.regimen_mode,
                                    block_gap_hours=args.block_gap_hours,
                                    max_block_days=args.max_block_days,
                                    include_anchor_time=True)
    anchor_df = data[['Newid', 'anchor_time', 'min_win_anc']].copy()

    # 每位病人的 ANC (times, values), 供 daily 重採樣
    anc = df[df['NAME'] == 'Absolute Neutrophil Count'].dropna(subset=['SIGNIN', 'RESULT'])
    anc_by_pid = {}
    for pid, g in anc.groupby('Newid'):
        g = g.sort_values('SIGNIN')
        anc_by_pid[pid] = (g['SIGNIN'].values.astype('datetime64[D]'),
                           g['RESULT'].values.astype(float))
    return anchor_df, anc_by_pid


def daily_context(anc_times_D, anc_vals, anchor_time, lookback):
    """把 anchor 之前的 ANC 重採樣成 daily 網格 (forward-fill + backfill)。回傳長度 lookback 的 array。"""
    anchor_D = np.datetime64(pd.Timestamp(anchor_time).to_datetime64(), 'D')
    grid = anchor_D - np.arange(lookback - 1, -1, -1)        # lookback 天, 結束於 anchor 當天
    # 對每個 grid 日, 取 <= 該日的最後一筆 ANC
    idx = np.searchsorted(anc_times_D, grid, side='right') - 1
    out = np.where(idx >= 0, anc_vals[np.clip(idx, 0, len(anc_vals) - 1)], np.nan)
    # 開頭若全 NaN (grid 早於第一筆) → backfill 用第一筆值
    if np.isnan(out).any():
        first = anc_vals[0]
        out = np.where(np.isnan(out), first, out)
    return out.astype(np.float32)


# ===========================================================================
# MODE: features  — 對所有 anchor 算 Sundial 預測特徵, 存 cache
# ===========================================================================
def run_features():
    import torch
    from transformers import AutoModelForCausalLM

    anchor_df, anc_by_pid = load_anchor_table()
    if args.limit > 0:
        anchor_df = anchor_df.iloc[:args.limit].copy()
    n = len(anchor_df)
    print(f"要算 Sundial 特徵的 anchor 數: {n} (lookback={args.lookback_days}d, "
          f"horizon={args.horizon}d, num_samples={args.num_samples}, device={args.device})")

    # 建所有 context 序列
    print("建 daily context 序列...")
    ctx = np.zeros((n, args.lookback_days), dtype=np.float32)
    for i, (_, r) in enumerate(anchor_df.iterrows()):
        at, av = anc_by_pid[r['Newid']]
        ctx[i] = daily_context(at, av, r['anchor_time'], args.lookback_days)

    print(f"載入 Sundial: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(args.model_id, trust_remote_code=True)
    model = model.to(args.device).eval()

    feats = {f'sundial_plow_{LABELS[t]}': np.zeros(n) for t in THRESHOLDS}
    feats['sundial_pred_min_mean'] = np.zeros(n)
    feats['sundial_pred_min_q10'] = np.zeros(n)
    feats['sundial_pred_min_q50'] = np.zeros(n)
    feats['sundial_pred_min_q90'] = np.zeros(n)
    feats['sundial_pred_end_mean'] = np.zeros(n)

    B = args.batch_size
    with torch.no_grad():
        for s in range(0, n, B):
            e = min(s + B, n)
            seqs = torch.tensor(ctx[s:e], dtype=torch.float32, device=args.device)
            # Sundial: 回傳 [batch, num_samples, horizon]
            out = model.generate(seqs, max_new_tokens=args.horizon, num_samples=args.num_samples)
            out = out.detach().cpu().numpy()
            if out.ndim == 2:                       # 防呆: [batch, horizon] → 補 sample 軸
                out = out[:, None, :]
            samp_min = out.min(axis=2)              # [batch, num_samples] 每條軌跡的 nadir
            for t in THRESHOLDS:
                feats[f'sundial_plow_{LABELS[t]}'][s:e] = (samp_min < t).mean(axis=1)
            feats['sundial_pred_min_mean'][s:e] = samp_min.mean(axis=1)
            feats['sundial_pred_min_q10'][s:e] = np.quantile(samp_min, 0.10, axis=1)
            feats['sundial_pred_min_q50'][s:e] = np.quantile(samp_min, 0.50, axis=1)
            feats['sundial_pred_min_q90'][s:e] = np.quantile(samp_min, 0.90, axis=1)
            feats['sundial_pred_end_mean'][s:e] = out[:, :, -1].mean(axis=1)
            if (s // B) % 10 == 0:
                print(f"  {e}/{n}")

    out_df = anchor_df.reset_index(drop=True).copy()
    for k, v in feats.items():
        out_df[k] = v
    out_df.to_parquet(args.cache)
    print(f"✓ Sundial 特徵存到 {args.cache} ({len(out_df)} rows, {len(feats)} feats)")


# ===========================================================================
# 共用: 載 cache + 對齊 + 2-way patient split
# ===========================================================================
def load_cache_and_split():
    from sklearn.model_selection import GroupShuffleSplit
    if not os.path.exists(args.cache):
        raise FileNotFoundError(f"{args.cache} 不存在 — 請先跑 --mode features")
    sf = pd.read_parquet(args.cache)
    sf['anchor_time'] = pd.to_datetime(sf['anchor_time'])
    sundial_cols = [c for c in sf.columns if c.startswith('sundial_')]

    pats = sf['Newid'].unique()
    tr_idx, te_idx = next(GroupShuffleSplit(test_size=0.2, random_state=args.seed)
                          .split(pats, groups=pats))
    train_pat, test_pat = set(pats[tr_idx]), set(pats[te_idx])
    return sf, sundial_cols, train_pat, test_pat


# ===========================================================================
# MODE: benchmark — 純 Sundial P(low) 當風險分數
# ===========================================================================
def run_benchmark():
    from sklearn.metrics import roc_auc_score, average_precision_score
    sf, _, train_pat, test_pat = load_cache_and_split()
    te = sf[sf['Newid'].isin(test_pat)]
    print(f"\n[Benchmark 純 Sundial] TEST anchors={len(te)}")
    print(f"{'Threshold':<12}{'n_pos':<8}{'Sundial AUC':<14}{'PR-AUC':<10}")
    for t in THRESHOLDS:
        y = (te['min_win_anc'].values < t).astype(int)
        if y.sum() < 5:
            continue
        score = te[f'sundial_plow_{LABELS[t]}'].values
        print(f"{LABELS[t]:<12}{int(y.sum()):<8}{roc_auc_score(y, score):<14.3f}"
              f"{average_precision_score(y, score):<10.3f}")
    print("(對照: 現有 LightGBM event-anchored <500 ≈ 0.845)")


# ===========================================================================
# MODE: hybrid — LightGBM covariate-only vs covariate + Sundial 特徵
# ===========================================================================
def run_hybrid():
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, average_precision_score
    sf, sundial_cols, train_pat, test_pat = load_cache_and_split()

    # 重建 per-seed event-anchored 特徵 (train top_drugs), 含 anchor_time 以便 merge
    df = pc.load_lab_df(exclude_heme_mode=args.heme_mode)
    chemo_df = pc.load_chemo_filtered(drop_gcsf=True)
    cm_info = eac.build_cm_info(pd.read_csv(pc.CANCER_MAP_PATH))
    markers = sorted(df[df['Newid'].isin(train_pat)]['NAME'].dropna().unique().tolist())
    top_drugs = pc.select_top_k_drugs(chemo_df, train_pat, pc.TOP_K_DRUGS)
    data = eac.build_anchor_dataset(df, chemo_df, cm_info, top_drugs, markers,
                                    horizon=args.horizon, gap=10, regimen_days=0,
                                    regimen_mode=args.regimen_mode,
                                    block_gap_hours=args.block_gap_hours,
                                    max_block_days=args.max_block_days,
                                    include_anchor_time=True)
    data['anchor_time'] = pd.to_datetime(data['anchor_time'])

    merged = data.merge(sf[['Newid', 'anchor_time'] + sundial_cols],
                        on=['Newid', 'anchor_time'], how='inner')
    print(f"\n[Hybrid] merge 後 anchors={len(merged)} (原 {len(data)})")

    cov_cols = [c for c in data.columns if c not in ('Newid', 'min_win_anc', 'anchor_time')]
    tr = merged['Newid'].isin(train_pat); te = merged['Newid'].isin(test_pat)
    anc_tr = merged.loc[tr, 'min_win_anc'].values
    anc_te = merged.loc[te, 'min_win_anc'].values

    LGB = {'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05, 'num_leaves': 31,
           'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'verbose': -1}

    def train_eval(cols, anc_tr, anc_te, thr):
        Xtr = merged.loc[tr, cols].values; Xte = merged.loc[te, cols].values
        ytr = (anc_tr < thr).astype(int); yte = (anc_te < thr).astype(int)
        n_pos = int(ytr.sum()); n_neg = len(ytr) - n_pos
        if n_pos < 10:
            return None
        cut = int(0.85 * len(Xtr))
        p = dict(LGB); p['scale_pos_weight'] = n_neg / n_pos
        dtr = lgb.Dataset(Xtr[:cut], label=ytr[:cut])
        dva = lgb.Dataset(Xtr[cut:], label=ytr[cut:], reference=dtr)
        m = lgb.train(p, dtr, num_boost_round=1000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
        pred = m.predict(Xte, num_iteration=m.best_iteration)
        return roc_auc_score(yte, pred), average_precision_score(yte, pred)

    print(f"{'Threshold':<12}{'cov-only AUC':<15}{'cov+Sundial AUC':<18}{'Δ AUC':<10}")
    for t in THRESHOLDS:
        a = train_eval(cov_cols, anc_tr, anc_te, t)
        b = train_eval(cov_cols + sundial_cols, anc_tr, anc_te, t)
        if a is None or b is None:
            continue
        print(f"{LABELS[t]:<12}{a[0]:<15.3f}{b[0]:<18.3f}{b[0]-a[0]:<+10.3f}")
    print("Δ>0 → Sundial 特徵有加分; ≈0 → LightGBM 的 covariate 已涵蓋同樣訊號")


if __name__ == '__main__':
    if args.mode == 'features':
        run_features()
    elif args.mode == 'benchmark':
        run_benchmark()
    elif args.mode == 'hybrid':
        run_hybrid()
