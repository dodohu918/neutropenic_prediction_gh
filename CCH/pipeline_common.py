"""
pipeline_common.py — 共用的 feature pipeline

目的: 讓 tuning 跟各 eval script 共用「完全相同」的 preprocessing / feature / cohort
      邏輯, 避免重複實作造成 drift。
"""
import re
import os
import pandas as pd
import numpy as np


# ===================================================================
# 路徑 (絕對路徑, 不論從哪個資料夾 / cwd 執行都正確)
#   pipeline_common.py 位於 CCH/
#   資料 (懶人包/, chemo_result.csv) 在 repo root (往上一層)
#   patient_cancer_subgroups.csv 與本檔同資料夾
# ===================================================================
_PC_DIR = os.path.dirname(os.path.abspath(__file__))        # .../CCH
_REPO_ROOT = os.path.dirname(_PC_DIR)                        # repo root
FILTERED_RESULT_PATH = os.path.join(_REPO_ROOT, '懶人包', 'filtered_test_result.csv')
CHEMO_PATH = os.path.join(_REPO_ROOT, 'chemo_result.csv')
CANCER_MAP_PATH = os.path.join(_PC_DIR, 'patient_cancer_subgroups.csv')
# 淋巴瘤骨髓判讀 (final ①=實為白血病, ②=淋巴瘤+骨髓侵犯 → 兩者都排除)
ADJUDICATION_PATH = os.path.join(_PC_DIR, 'lymphoma_marrow_adjudication.csv')

RESOLUTIONS = [8, 12, 16, 20, 24, 28, 32]
WINDOW_SIZE = 4
THRESHOLDS = [1.5, 1.0, 0.5]
TOP_K_DRUGS = 20

_TAG_MAP = {'none': 'withheme', 'all': 'noheme', 'leukemia_only': 'noleuk',
            'leukemia_only_adj': 'noleukadj'}


# ===================================================================
# Drug canonicalization (與主 eval 相同)
# ===================================================================
CANON_RULES = [
    (r'fluoro-?uracil|fluorouracil|5-fu',          'Fluorouracil'),
    (r'cisplatin|abiplatin|kemoplat',              'Cisplatin'),
    (r'cytosar|cytarabine',                        'Cytarabine'),
    (r'taxotere|docetaxel',                        'Docetaxel'),
    (r'pharmorubicin|epirubicin',                  'Epirubicin'),
    (r'vepesid|etoposide',                         'Etoposide'),
    (r'campto|irinotecan',                         'Irinotecan'),
    (r'avastin|bevacizumab',                       'Bevacizumab'),
    (r'oxalip|oxaliplatin',                        'Oxaliplatin'),
    (r'methotrexate|methotrexat',                  'Methotrexate'),
    (r'syklofosfamid|endoxan|cyclophosphamide',    'Cyclophosphamide'),
    (r'holoxan|ifosfamide',                        'Ifosfamide'),
    (r'vincristine',                               'Vincristine'),
    (r'carboplatine|carboplatin',                  'Carboplatin'),
    (r'dacarbazine',                               'Dacarbazine'),
    (r'gemzar|gemcitabine',                        'Gemcitabine'),
    (r'herceptin|trastuzumab',                     'Trastuzumab'),
    (r'mabthera|rituximab',                        'Rituximab'),
    (r'erbitux|cetuximab',                         'Cetuximab'),
    (r'alimta|pemetrexed',                         'Pemetrexed'),
    (r'megatus|allmercap|mercaptopurine|6-mp',     'Mercaptopurine'),
    (r'perjeta|pertuzumab',                        'Pertuzumab'),
    (r'hycamtin|topotecan',                        'Topotecan'),
    (r'vidaza|azacitidine',                        'Azacitidine'),
    (r'zavedos|idarubicin',                        'Idarubicin'),
    (r'formoxol|paclitaxel|taxol',                 'Paclitaxel'),
    (r'keytruda|pembrolizumab',                    'Pembrolizumab'),
    (r'lipo-?dox|doxil|liposomal doxorubicin',     'DoxorubicinLipo'),
    (r'blincyto|blinatumomab',                     'Blinatumomab'),
]

# 其餘單株抗體 / ADC / 雙特異性抗體 / 融合蛋白 (含生物相似藥與試驗用藥)。
# 只比對藥名「開頭」(去掉【規格】前綴), 避免把 "Gemzar Inj.(試用藥BGB-A317)" 這類
# 試驗中的化療誤判為 biologic。CANON_RULES 沒命中才會套用。
BIOLOGIC_RULES = [
    (r'herzuma',                                   'Trastuzumab'),
    (r'mvasi',                                     'Bevacizumab'),
    (r'rixathon',                                  'Rituximab'),
    (r'opdivo|ono-4538|nivolumab|ro7121661',       'Nivolumab'),
    (r'tecentriq|atezolizumab',                    'Atezolizumab'),
    (r'imfinzi',                                   'Durvalumab'),
    (r'yervoy',                                    'Ipilimumab'),
    (r'bavencio',                                  'Avelumab'),
    (r'bgb-a317',                                  'Tislelizumab'),
    (r'bgb-a1217',                                 'Ociperlimab'),
    (r'hlx10',                                     'Serplulimab'),
    (r'hlx07',                                     'HLX07'),
    (r'ab122',                                     'Zimberelimab'),
    (r'tiragolumab',                               'Tiragolumab'),
    (r'gsk3359609',                                'Feladilimab'),
    (r'cyramza|ramucirumab',                       'Ramucirumab'),
    (r'vectibix',                                  'Panitumumab'),
    (r'zaltrap',                                   'Aflibercept'),
    (r'rybrevant',                                 'Amivantamab'),
    (r'lartruvo',                                  'Olaratumab'),
    (r'metmab',                                    'Onartuzumab'),
    (r'darzalex|daratumumab',                      'Daratumumab'),
    (r'sarclisa',                                  'Isatuximab'),
    (r'gazyva',                                    'Obinutuzumab'),
    (r'tafasitamab',                               'Tafasitamab'),
    (r'qarziba',                                   'Dinutuximab'),
    (r'kadcyla',                                   'TrastuzumabEmtansine'),
    (r'enhertu',                                   'TrastuzumabDeruxtecan'),
    (r'adcetris',                                  'BrentuximabVedotin'),
    (r'polivy',                                    'PolatuzumabVedotin'),
    (r'besponsa',                                  'InotuzumabOzogamicin'),
    (r'trodelvy',                                  'SacituzumabGovitecan'),
    (r'sgn-cd33a',                                 'Vadastuximab'),
    (r'asp-1929',                                  'ASP1929'),
    (r'adalimumab',                                'Adalimumab'),
    (r'etanercept',                                'Etanercept'),
]


def canonicalize_drug(name):
    if pd.isna(name):
        return 'OTHER'
    s = str(name).lower()
    for pattern, canon in CANON_RULES:
        if re.search(pattern, s):
            return canon
    head = re.sub(r'^【.*?】', '', s).strip()
    for pattern, canon in BIOLOGIC_RULES:
        if re.match(pattern, head):
            return canon
    return 'OTHER'


def convert_dose_to_mg(item, dose, unit):
    if pd.isna(dose):
        return 0.0, 1
    try:
        dose = float(dose)
    except (TypeError, ValueError):
        return 0.0, 1
    u = '' if pd.isna(unit) else str(unit).strip()
    if u == 'mg':
        return dose, 1
    if u in ('ug', 'mcg'):
        return dose / 1000.0, 1
    if u == 'mL':
        item_s = str(item).lower() if not pd.isna(item) else ''
        if 'megatus' in item_s or 'allmercap' in item_s:
            return dose * 20.0, 1
        return 0.0, 1
    return 0.0, 1


def load_chemo(chemo_path=CHEMO_PATH):
    cdf = pd.read_csv(chemo_path)
    if 'Unnamed: 0' in cdf.columns:
        cdf = cdf.drop(columns=['Unnamed: 0'])
    cdf['time'] = pd.to_datetime(cdf['time'], errors='coerce')
    cdf = cdf.dropna(subset=['Newid', 'time']).copy()
    cdf['canon'] = cdf['item'].apply(canonicalize_drug)
    converted = cdf.apply(lambda r: convert_dose_to_mg(r['item'], r['dose'], r['unit']),
                          axis=1, result_type='expand')
    cdf['dose_mg'] = converted[0].astype(float)
    cdf['admin_cnt'] = converted[1].astype(int)
    cdf = cdf[['Newid', 'time', 'canon', 'dose_mg', 'admin_cnt']]
    cdf = cdf.sort_values(['Newid', 'time']).reset_index(drop=True)
    return cdf


# ===================================================================
# Lab feature generation (與主 eval 相同)
# ===================================================================
STATS = ['mean', 'max', 'min', 'std', 'last', 'last_1', 'last_2', 'last_3']


def _safe(name):
    return name.replace(' ', '_')


def create_features_with_mask(group_df, expected_markers):
    feature_dict = {}
    mask_dict = {}
    available = dict(tuple(group_df.groupby('NAME')))
    for name in expected_markers:
        safe = _safe(name)
        if name in available and not available[name].empty:
            sub_df = available[name].sort_values('SIGNIN')
            vals = sub_df['RESULT']
            agg = vals.agg(['mean', 'max', 'min', 'std'])
            for stat in ['mean', 'max', 'min', 'std']:
                v = agg[stat]
                col = f"{safe}_{stat}"
                feature_dict[col] = v if pd.notna(v) else 0.0
                mask_dict[f"{col}_mask"] = 1 if pd.notna(v) else 0
            last_val = vals.iloc[-1]
            feature_dict[f"{safe}_last"] = last_val if pd.notna(last_val) else 0.0
            mask_dict[f"{safe}_last_mask"] = 1 if pd.notna(last_val) else 0
            last_vals = vals.tail(3).values[::-1]
            for i in range(3):
                col = f"{safe}_last_{i + 1}"
                if i < len(last_vals) and pd.notna(last_vals[i]):
                    feature_dict[col] = last_vals[i]
                    mask_dict[f"{col}_mask"] = 1
                else:
                    feature_dict[col] = 0.0
                    mask_dict[f"{col}_mask"] = 0
        else:
            for stat in STATS:
                col = f"{safe}_{stat}"
                feature_dict[col] = 0.0
                mask_dict[f"{col}_mask"] = 0
    return pd.DataFrame([{**feature_dict, **mask_dict}])


# ===================================================================
# Chemo feature generation (與主 eval 相同)
# ===================================================================
TRAILING_DAYS = 21
NO_CHEMO_CAP_DAYS = 365


def build_chemo_lookup(chemo_df):
    return {pid: g.sort_values('time').reset_index(drop=True)
            for pid, g in chemo_df.groupby('Newid')}


def select_top_k_drugs(chemo_df, train_patients, k=TOP_K_DRUGS):
    sub = chemo_df[chemo_df['Newid'].isin(train_patients)]
    counts = sub['canon'].value_counts()
    counts = counts[counts.index != 'OTHER']
    return counts.head(k).index.tolist()


def create_chemo_features_for_bin(patient_chemo, bin_start, bin_end, top_drugs):
    feats = {}
    for d in top_drugs:
        feats[f'chemo_{d}_dose_in_bin'] = 0.0
        feats[f'chemo_{d}_dose_in_bin_mask'] = 0
        feats[f'chemo_{d}_dose_last_21d'] = 0.0
    feats['chemo_OTHER_n_admin_in_bin'] = 0.0
    feats['chemo_OTHER_n_admin_last_21d'] = 0.0
    feats['chemo_n_admin_in_bin'] = 0.0
    feats['chemo_n_distinct_drugs_in_bin'] = 0.0
    feats['chemo_any_in_bin_mask'] = 0
    feats['chemo_n_admin_last_21d'] = 0.0
    feats['chemo_n_admin_cumulative_to_bin_end'] = 0.0
    feats['chemo_days_since_last_chemo'] = float(NO_CHEMO_CAP_DAYS)

    if patient_chemo is None or len(patient_chemo) == 0:
        return feats

    times = patient_chemo['time'].values
    canons = patient_chemo['canon'].values
    doses = patient_chemo['dose_mg'].values
    cnts = patient_chemo['admin_cnt'].values

    in_bin_mask = (times >= np.datetime64(bin_start)) & (times < np.datetime64(bin_end))
    if in_bin_mask.any():
        feats['chemo_any_in_bin_mask'] = 1
        feats['chemo_n_admin_in_bin'] = float(cnts[in_bin_mask].sum())
        distinct = set()
        for c, d, n in zip(canons[in_bin_mask], doses[in_bin_mask], cnts[in_bin_mask]):
            distinct.add(c)
            if c in top_drugs:
                feats[f'chemo_{c}_dose_in_bin'] += float(d)
                feats[f'chemo_{c}_dose_in_bin_mask'] = 1
            else:
                feats['chemo_OTHER_n_admin_in_bin'] += float(n)
        feats['chemo_n_distinct_drugs_in_bin'] = float(len(distinct))

    trail_start = bin_end - pd.Timedelta(days=TRAILING_DAYS)
    trail_mask = (times > np.datetime64(trail_start)) & (times <= np.datetime64(bin_end))
    if trail_mask.any():
        feats['chemo_n_admin_last_21d'] = float(cnts[trail_mask].sum())
        for c, d, n in zip(canons[trail_mask], doses[trail_mask], cnts[trail_mask]):
            if c in top_drugs:
                feats[f'chemo_{c}_dose_last_21d'] += float(d)
            else:
                feats['chemo_OTHER_n_admin_last_21d'] += float(n)

    cum_mask = times <= np.datetime64(bin_end)
    if cum_mask.any():
        feats['chemo_n_admin_cumulative_to_bin_end'] = float(cnts[cum_mask].sum())
        last_chemo_time = pd.Timestamp(times[cum_mask].max())
        delta_days = (bin_end - last_chemo_time).total_seconds() / 86400.0
        feats['chemo_days_since_last_chemo'] = float(min(max(delta_days, 0.0), NO_CHEMO_CAP_DAYS))
    return feats


def generate_features_for_resolution(data, num_bins, expected_markers,
                                     chemo_lookup, top_drugs):
    feature_dicts = []
    for pid in data['Newid'].unique():
        patient_data = data[data['Newid'] == pid].sort_values('SIGNIN')
        if patient_data.empty or len(patient_data) < 2:
            continue
        start_time = patient_data['SIGNIN'].min()
        end_time = patient_data['SIGNIN'].max()
        if start_time == end_time:
            continue
        patient_data = patient_data.copy()
        patient_data['time_since_start'] = (patient_data['SIGNIN'] - start_time).dt.total_seconds()
        total_seconds = (end_time - start_time).total_seconds()
        time_bins = np.linspace(0, total_seconds, num_bins + 1)
        patient_data['time_bin'] = np.digitize(patient_data['time_since_start'], bins=time_bins) - 1
        patient_chemo = chemo_lookup.get(pid)
        for current_bin in range(num_bins - 1):
            bin_data = patient_data[patient_data['time_bin'] == current_bin]
            if bin_data.empty:
                continue
            features = create_features_with_mask(bin_data, expected_markers)
            next_bin = current_bin + 1
            anc_next_bin = patient_data[(patient_data['time_bin'] == next_bin) &
                                        (patient_data['NAME'] == 'Absolute Neutrophil Count')]
            if anc_next_bin.empty:
                continue
            target_value = anc_next_bin.iloc[0]['RESULT']
            if pd.isnull(target_value):
                continue
            bin_start_dt = start_time + pd.Timedelta(seconds=float(time_bins[current_bin]))
            bin_end_dt = start_time + pd.Timedelta(seconds=float(time_bins[current_bin + 1]))
            chemo_feats = create_chemo_features_for_bin(
                patient_chemo, bin_start_dt, bin_end_dt, top_drugs)
            for k, v in chemo_feats.items():
                features[k] = v
            features['Newid'] = pid
            features['current_bin'] = current_bin
            features['target'] = target_value
            feature_dicts.append(features)
    if len(feature_dicts) == 0:
        return None
    return pd.concat(feature_dicts, ignore_index=True)


def generate_sequences(X_full, feature_columns, window_size=WINDOW_SIZE):
    sequences = []
    targets = []
    seq_patient_ids = []
    for pid in X_full['Newid'].unique():
        group = X_full[X_full['Newid'] == pid].sort_values('current_bin')
        X_features = group[feature_columns].values
        y_values = group['target'].values
        if len(X_features) < window_size + 1:
            continue
        for i in range(len(X_features) - window_size):
            seq = X_features[i:i + window_size]
            target_val = y_values[i + window_size]
            sequences.append(seq)
            targets.append(target_val)
            seq_patient_ids.append(pid)
    return np.array(sequences), np.array(targets), np.array(seq_patient_ids)


# ===================================================================
# 高階: 載入 lab df + cohort 排除 (與主 eval 的 cohort 邏輯相同)
# ===================================================================
def load_lab_df(exclude_heme_mode='leukemia_only', verbose=True):
    """讀 filtered lab 資料, 套用血液腫瘤排除。回傳 df。

    exclude_heme_mode: 'none' / 'all' / 'leukemia_only' (見主 eval 說明)
    """
    if exclude_heme_mode not in _TAG_MAP:
        raise ValueError(f"exclude_heme_mode 必須是 {list(_TAG_MAP)}, 得到 {exclude_heme_mode!r}")

    dtype_spec = {'Unnamed: 0': 'int64', 'Newid': 'int64',
                  'NAME': 'str', 'SIGNIN': 'str', 'RESULT': 'str'}
    df = pd.read_csv(FILTERED_RESULT_PATH, dtype=dtype_spec)
    df['SIGNIN'] = pd.to_datetime(df['SIGNIN'], errors='coerce')
    df['RESULT'] = pd.to_numeric(df['RESULT'], errors='coerce')
    df = df.sort_values(by=['Newid', 'SIGNIN']).copy()
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    if exclude_heme_mode != 'none':
        if not os.path.exists(CANCER_MAP_PATH):
            raise FileNotFoundError(
                f"{CANCER_MAP_PATH} 不存在 — 請先跑 cohort_characterization.py")
        cm = pd.read_csv(CANCER_MAP_PATH, usecols=['Newid', 'cancer_category', 'primary_site'])
        if exclude_heme_mode == 'all':
            excl = cm['cancer_category'] == 'Hematologic'
        else:  # leukemia_only / leukemia_only_adj 都先排骨髓原發
            excl = cm['primary_site'].fillna('').str.lower().str.contains('bone marrow')
        excl_pids = set(cm.loc[excl, 'Newid'].astype(int))

        n_adj = 0
        if exclude_heme_mode == 'leukemia_only_adj':
            # 再排掉淋巴瘤判讀為 ①(實為白血病) / ②(淋巴瘤+骨髓侵犯) 的病人
            if not os.path.exists(ADJUDICATION_PATH):
                raise FileNotFoundError(f"{ADJUDICATION_PATH} 不存在")
            adj = pd.read_csv(ADJUDICATION_PATH, usecols=['Newid', 'final'])
            flagged = adj['final'].astype(str).str.strip().str[0].isin(['①', '②'])
            adj_pids = set(adj.loc[flagged, 'Newid'].astype(int))
            n_adj = len(adj_pids)
            excl_pids |= adj_pids

        before = df['Newid'].nunique()
        df = df[~df['Newid'].isin(excl_pids)].copy()
        if verbose:
            extra = f" (含 adjudication ①②: {n_adj} 位)" if exclude_heme_mode == 'leukemia_only_adj' else ""
            print(f"[exclude_heme={exclude_heme_mode}] {before} → {df['Newid'].nunique()} 病人{extra}")
    return df


MAB_DRUGS = frozenset({
    'Bevacizumab', 'Trastuzumab', 'Rituximab', 'Cetuximab',
    'Pertuzumab', 'Pembrolizumab', 'Blinatumomab',
} | {canon for _, canon in BIOLOGIC_RULES})


def load_chemo_filtered(exclude_mab=True, verbose=True):
    """讀 chemo 資料。
    exclude_mab: 移除單株抗體免疫治療藥物 (預設 True; 保留「純化療」cohort)
    """
    chemo_df = load_chemo(CHEMO_PATH)
    if exclude_mab:
        n0 = len(chemo_df)
        chemo_df = chemo_df[~chemo_df['canon'].isin(MAB_DRUGS)].copy()
        if verbose:
            print(f"[exclude_mab] chemo rows {n0} → {len(chemo_df)} (移除 mAb {n0 - len(chemo_df)} 筆)")
    return chemo_df
