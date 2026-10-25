"""
mimic_fn_labels.py — MIMIC-IV cohort 修正 + febrile neutropenia (FN) proxy 事件抽取

從本地 physionet.org/files/mimiciv/3.1 讀 (不需 BigQuery)，輸出到 mimic_data/：
  mimic_cohort_fixed.csv → Newid, keep, reason, cancer_category, primary_site
                           (修正版癌別；keep=0 的病人在 FN 分析中排除)
  mimic_fn_events.csv    → Newid, time, kind
                           kind ∈ BC (抽 blood culture) / BC_POS (blood culture 長菌) /
                                  INFLAB (CRP 或 Procalcitonin 檢驗) / FEVER (ICU 體溫 ≥38.3°C)
  mimic_fn_icd_adm.csv   → Newid, admittime, dischtime
                           (同一住院有 D70.x + R50.81，或 ICD-9 288.0x + 780.6x)

Cohort 修正 (mimic_extract.py 的 chemo_pts 會混入非癌症 MTX/CTX 病人):
  1. 必須有惡性腫瘤碼: ICD-10 C00–C96 或 ICD-9 140–208
  2. 再排 ICD-9 白血病/骨髓瘤 (203–208)；ICD-10 C90–C95 在抽取時已排
  3. 癌別改用「第一個原發部位碼」: 跳過 C77–C80 / C7B (轉移/未明) 與 ICD-9 196–199

FN proxy 定義 (與 CCH/cch_fn_proxy.py 相同，見 mimic_eval_lgb_fn.py):
  ANC<500 且 ±48h 內同時有 BC 與 INFLAB (CRP 或 Procalcitonin)

用法 (從 MIMIC/ 執行):
  python mimic_fn_labels.py
ICU chartevents 有 3.5GB，約需 10–15 分鐘。
"""

import os
import pandas as pd

_HERE     = os.path.dirname(os.path.abspath(__file__))
MIMIC_DIR = os.path.join(_HERE, 'mimic_data')
RAW       = os.path.join(_HERE, 'physionet.org', 'files', 'mimiciv', '3.1')

# infection work-up lab: 用 d_labitems label 比對 (不含 High-Sensitivity CRP)。
# MIMIC-IV 3.1 hosp 沒有 Procalcitonin item, 實際只會抓到 C-Reactive Protein (itemid 50889)
INFLAB_REGEX = r'^C-Reactive Protein$|procalcitonin'
BC_SPECS    = ['BLOOD CULTURE', 'BLOOD CULTURE ( MYCO/F LYTIC BOTTLE)']
TEMP_ITEMS  = {223761: 'F', 223762: 'C'}
FEVER_C     = 38.3


def icd10_category(code):
    """同 mimic_extract.py 的 SQL CASE; C7A (神經內分泌原發) → Other"""
    c3 = code[:3]
    if ('C00' <= c3 <= 'C14') or c3 in ('C30', 'C31', 'C32'): return 'Head and Neck'
    if c3 in ('C33', 'C34'):                                  return 'Lung'
    if c3 == 'C50':                                           return 'Breast'
    if c3 in ('C22', 'C23', 'C24'):                           return 'Liver/Biliary'
    if 'C18' <= c3 <= 'C21':                                  return 'Colorectal'
    if ('C15' <= c3 <= 'C17') or c3 in ('C25', 'C26'):        return 'GI (Other)'
    if 'C51' <= c3 <= 'C58':                                  return 'Gynecologic'
    if 'C60' <= c3 <= 'C68':                                  return 'GU'
    if c3 in ('C43', 'C44'):                                  return 'Skin'
    if 'C70' <= c3 <= 'C72':                                  return 'CNS'
    if 'C81' <= c3 <= 'C88':                                  return 'Hematologic'
    return 'Other'


def icd9_category(code):
    n = int(code[:3])
    if 140 <= n <= 149 or n in (160, 161): return 'Head and Neck'
    if n == 162:                           return 'Lung'
    if n in (174, 175):                    return 'Breast'
    if n in (155, 156):                    return 'Liver/Biliary'
    if n in (153, 154):                    return 'Colorectal'
    if 150 <= n <= 159:                    return 'GI (Other)'
    if 179 <= n <= 184:                    return 'Gynecologic'
    if 185 <= n <= 189:                    return 'GU'
    if n in (172, 173):                    return 'Skin'
    if n in (191, 192):                    return 'CNS'
    if 200 <= n <= 202:                    return 'Hematologic'
    return 'Other'


print("載入現有 MIMIC cohort...")
pts = pd.read_csv(os.path.join(MIMIC_DIR, 'mimic_patients.csv'))
ids = set(pts['Newid'].astype(int))
print(f"  原 cohort: {len(ids):,} 位")

# ─── 1. Cohort 修正 ───────────────────────────────────────────────────────────
print("\n[1] 修正 cohort (diagnoses_icd)...")
dx = pd.read_csv(os.path.join(RAW, 'hosp', 'diagnoses_icd.csv.gz'), dtype=str)
dx['Newid'] = dx['subject_id'].astype(int)
dx = dx[dx['Newid'].isin(ids)].copy()
dx['seq_num'] = dx['seq_num'].astype(int)

v10, v9 = dx['icd_version'] == '10', dx['icd_version'] == '9'
code = dx['icd_code']
malig10 = v10 & code.str.match(r'^C')
malig9  = v9 & code.str.match(r'^(1[4-9]\d|20[0-8])')
second  = (v10 & code.str.match(r'^(C7[7-9]|C80|C7B)')) | (v9 & code.str.match(r'^19[6-9]'))
heme9   = v9 & code.str.match(r'^20[3-8]')

has_malig = set(dx.loc[malig10 | malig9, 'Newid'])
has_heme9 = set(dx.loc[heme9, 'Newid'])

prim = dx[(malig10 | malig9) & ~second].sort_values(['Newid', 'icd_version', 'seq_num'],
                                                   ascending=[True, True, True])
# ICD-10 優先 ('10' < '9' 字串排序)，再取 seq_num 最小
prim = prim.drop_duplicates('Newid', keep='first')
prim['cancer_category'] = [icd10_category(c) if v == '10' else icd9_category(c)
                           for c, v in zip(prim['icd_code'], prim['icd_version'])]
prim = prim.set_index('Newid')

rows = []
for pid in sorted(ids):
    if pid not in has_malig:
        keep, reason = 0, 'no_malignancy_code'
    elif pid in has_heme9:
        keep, reason = 0, 'icd9_leukemia_myeloma'
    else:
        keep, reason = 1, ''
    if pid in prim.index:
        cat, site = prim.at[pid, 'cancer_category'], prim.at[pid, 'icd_code']
    else:
        cat, site = 'Unknown', 'Unknown'   # 只有轉移碼
    rows.append({'Newid': pid, 'keep': keep, 'reason': reason,
                 'cancer_category': cat, 'primary_site': site})
cohort = pd.DataFrame(rows)
out = os.path.join(MIMIC_DIR, 'mimic_cohort_fixed.csv')
cohort.to_csv(out, index=False)
print(f"  keep: {cohort['keep'].sum():,} | 排除: {cohort['reason'].value_counts().to_dict()}")
print(f"  → {out}")

# ─── 2. Blood culture ─────────────────────────────────────────────────────────
print("\n[2] Blood culture (microbiologyevents)...")
mb = pd.read_csv(os.path.join(RAW, 'hosp', 'microbiologyevents.csv.gz'),
                 usecols=['subject_id', 'charttime', 'chartdate', 'spec_type_desc', 'org_name'])
mb = mb[mb['subject_id'].isin(ids) & mb['spec_type_desc'].isin(BC_SPECS)].copy()
mb['time'] = pd.to_datetime(mb['charttime'].fillna(mb['chartdate']))
bc = mb[['subject_id', 'time']].drop_duplicates().assign(kind='BC')
pos = mb['org_name'].notna() & ~mb['org_name'].str.contains('CANCELLED', na=False)
bc_pos = mb.loc[pos, ['subject_id', 'time']].drop_duplicates().assign(kind='BC_POS')
print(f"  BC: {len(bc):,} | BC_POS: {len(bc_pos):,}")

# ─── 3. CRP / Procalcitonin ───────────────────────────────────────────────────
print("\n[3] CRP / Procalcitonin (labevents)...")
li = pd.read_csv(os.path.join(RAW, 'hosp', 'd_labitems.csv.gz'), usecols=['itemid', 'label'])
inflab_items = set(li.loc[li['label'].str.contains(INFLAB_REGEX, case=False, na=False), 'itemid'])
print(f"  itemids: {sorted(inflab_items)}")
lb = []
for ch in pd.read_csv(os.path.join(RAW, 'hosp', 'labevents.csv.gz'),
                      usecols=['subject_id', 'itemid', 'charttime'],
                      chunksize=10_000_000):
    lb.append(ch[ch['itemid'].isin(inflab_items) & ch['subject_id'].isin(ids)])
lb = pd.concat(lb)
lb['time'] = pd.to_datetime(lb['charttime'])
inflab = lb[['subject_id', 'time']].drop_duplicates().assign(kind='INFLAB')
print(f"  INFLAB: {len(inflab):,}")

# ─── 4. ICU 體溫 ──────────────────────────────────────────────────────────────
print("\n[4] ICU 體溫 (chartevents 3.5GB, 約 10–15 分鐘)...")
tp = []
for ch in pd.read_csv(os.path.join(RAW, 'icu', 'chartevents.csv.gz'),
                      usecols=['subject_id', 'charttime', 'itemid', 'valuenum'],
                      chunksize=10_000_000):
    tp.append(ch[ch['itemid'].isin(list(TEMP_ITEMS)) & ch['subject_id'].isin(ids)])
tp = pd.concat(tp)
tp['temp_c'] = tp['valuenum'].where(tp['itemid'] == 223762, (tp['valuenum'] - 32) * 5 / 9)
tp = tp[(tp['temp_c'] > 30) & (tp['temp_c'] < 45)]          # 排除明顯登錄錯誤
tp['time'] = pd.to_datetime(tp['charttime'])
fever = tp.loc[tp['temp_c'] >= FEVER_C, ['subject_id', 'time']].drop_duplicates().assign(kind='FEVER')
print(f"  ICU 體溫: {len(tp):,} 筆 | ≥{FEVER_C}°C: {len(fever):,}")

ev = (pd.concat([bc, bc_pos, inflab, fever])
      .rename(columns={'subject_id': 'Newid'})
      .sort_values(['Newid', 'time']))
out = os.path.join(MIMIC_DIR, 'mimic_fn_events.csv')
ev.to_csv(out, index=False)
print(f"  → {out}  ({len(ev):,} 筆)")

# ─── 5. ICD 編碼的 FN 住院 ─────────────────────────────────────────────────────
print("\n[5] ICD FN 住院...")
hadm = lambda m: set(dx.loc[m, 'hadm_id'])
fn_hadm = ((hadm(v10 & code.str.startswith('D70')) & hadm(v10 & (code == 'R5081'))) |
           (hadm(v9 & code.str.startswith('2880')) & hadm(v9 & code.str.startswith('7806'))))
adm = pd.read_csv(os.path.join(RAW, 'hosp', 'admissions.csv.gz'), dtype={'hadm_id': str},
                  usecols=['subject_id', 'hadm_id', 'admittime', 'dischtime'])
adm = adm[adm['hadm_id'].isin(fn_hadm)].rename(columns={'subject_id': 'Newid'})
out = os.path.join(MIMIC_DIR, 'mimic_fn_icd_adm.csv')
adm[['Newid', 'admittime', 'dischtime']].to_csv(out, index=False)
print(f"  FN 住院: {len(adm):,} 次 ({adm['Newid'].nunique():,} 位) → {out}")

print("\n✓ 完成。下一步: python mimic_eval_lgb_fn.py")
