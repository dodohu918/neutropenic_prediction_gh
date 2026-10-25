"""
mimic_extract.py — 從 BigQuery 抽取 MIMIC-IV 資料，轉成 pipeline 格式

輸出四個 CSV 到 mimic_data/ 資料夾：
  mimic_lab.csv      → 等同 filtered_test_result.csv  (Newid, NAME, SIGNIN, RESULT)
  mimic_chemo.csv    → 等同 chemo_result.csv           (Newid, time, item, dose, unit)
  mimic_patients.csv → 等同 patient_cancer_subgroups.csv
  mimic_gcsf.csv     → G-CSF 給藥 (Newid, time, item)；給 mimic_gcsf.py 排除受 G-CSF 影響的 anchors

用法:
  python mimic_extract.py

認證: 第一次跑會開瀏覽器要求登入 Google 帳號 (用連結 PhysioNet 的那個帳號)。
"""

import os
import pandas as pd
import pandas_gbq

# ─── 必填：你的 Google Cloud project ID ───────────────────────────────────────
# BigQuery console 左上角 project 選擇器裡可以看到
PROJECT_ID = 'mimic-scoping'
# ─────────────────────────────────────────────────────────────────────────────

MIMIC_DS = 'physionet-data.mimiciv_3_1_hosp'
OUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mimic_data')
os.makedirs(OUT_DIR, exist_ok=True)

def run_query(sql, desc):
    print(f"  [{desc}] 執行中...", flush=True)
    df = pandas_gbq.read_gbq(sql, project_id=PROJECT_ID)
    print(f"  [{desc}] 完成：{len(df):,} 筆")
    return df


# ─── 共用 CTE：定義 solid tumor chemo 病人 ───────────────────────────────────
# 同 scoping query：化療 + 排白血病/骨髓瘤 (C90-C95)
SOLID_PTS_CTE = f"""
  chemo_pts AS (
    SELECT DISTINCT subject_id
    FROM `{MIMIC_DS}.emar`
    WHERE REGEXP_CONTAINS(LOWER(medication),
        r'cisplatin|carboplatin|docetaxel|paclitaxel|cyclophosphamide|doxorubicin|'
        r'fluorouracil|oxaliplatin|irinotecan|gemcitabine|vincristine|etoposide|'
        r'methotrexate|cytarabine|epirubicin|ifosfamide|pemetrexed|topotecan')
      AND charttime IS NOT NULL
  ),
  heme_pts AS (
    SELECT DISTINCT subject_id
    FROM `{MIMIC_DS}.diagnoses_icd`
    WHERE icd_version = 10
      AND SUBSTR(icd_code, 1, 3) IN ('C90','C91','C92','C93','C94','C95')
  ),
  solid_pts AS (
    SELECT subject_id FROM chemo_pts
    WHERE subject_id NOT IN (SELECT subject_id FROM heme_pts)
  )
"""

CHEMO_REGEX = (
    r'cisplatin|carboplatin|docetaxel|paclitaxel|cyclophosphamide|doxorubicin|'
    r'fluorouracil|oxaliplatin|irinotecan|gemcitabine|vincristine|etoposide|'
    r'methotrexate|cytarabine|epirubicin|ifosfamide|pemetrexed|topotecan'
)

# ─── 1. Lab events → mimic_lab.csv ───────────────────────────────────────────
# itemid 對應你的 marker NAME
ITEMID_MAP = {
    52075: 'Absolute Neutrophil Count',   # Hematology ANC (主要)
    53159: 'Absolute Neutrophil Count',   # Chemistry ANC (備用)
    51256: 'Neutrophil',
    51244: 'Lymphocyte',
    51254: 'Monocyte',
    50861: 'GPT (ALT)',
    50878: 'GOT (AST)',
}
ITEMIDS = list(ITEMID_MAP.keys())

lab_sql = f"""
WITH {SOLID_PTS_CTE}
SELECT
  l.subject_id              AS Newid,
  l.itemid,
  CAST(l.charttime AS STRING) AS SIGNIN,
  l.valuenum                AS RESULT
FROM `{MIMIC_DS}.labevents` l
WHERE l.subject_id IN (SELECT subject_id FROM solid_pts)
  AND l.itemid IN ({','.join(str(i) for i in ITEMIDS)})
  AND l.valuenum IS NOT NULL
  AND l.valuenum > 0
ORDER BY l.subject_id, l.charttime
"""

lab_df = run_query(lab_sql, 'lab events')

# itemid → NAME
lab_df['NAME'] = lab_df['itemid'].map(ITEMID_MAP)

# ANC 異常值過濾：>50 K/µL 不可能 (max 觀察到 880，顯然資料錯誤)
anc_mask = lab_df['itemid'].isin([52075, 53159])
lab_df   = lab_df[~(anc_mask & (lab_df['RESULT'] > 50))]

# 去重：同病人同時間若同時有 itemid 52075 + 53159，保留 52075 (Hematology 為主)
lab_df = (lab_df
          .sort_values(['Newid', 'SIGNIN', 'itemid'])
          .drop_duplicates(subset=['Newid', 'NAME', 'SIGNIN'], keep='first')
          [['Newid', 'NAME', 'SIGNIN', 'RESULT']])

out_lab = os.path.join(OUT_DIR, 'mimic_lab.csv')
lab_df.to_csv(out_lab, index=False)
print(f"  → 儲存 {out_lab}  ({len(lab_df):,} 筆)\n")


# ─── 2. Chemo emar → mimic_chemo.csv ─────────────────────────────────────────
# emar_detail 的欄位名在 MIMIC-IV 3.1 不穩定，直接略過 dose。
# pipeline_common.convert_dose_to_mg 遇到 NULL dose 會回傳 (0.0, 1)，
# 即 admin_cnt=1 仍被計入，init_n_admin / cum_n_admin 等特徵不受影響。
chemo_sql = f"""
WITH {SOLID_PTS_CTE}
SELECT
  e.subject_id                AS Newid,
  CAST(e.charttime AS STRING) AS time,
  e.medication                AS item,
  CAST(NULL AS FLOAT64)       AS dose,
  CAST(NULL AS STRING)        AS unit
FROM `{MIMIC_DS}.emar` e
WHERE e.subject_id IN (SELECT subject_id FROM solid_pts)
  AND REGEXP_CONTAINS(LOWER(e.medication), r'{CHEMO_REGEX}')
  AND e.charttime IS NOT NULL
ORDER BY Newid, time
"""

chemo_df = run_query(chemo_sql, 'chemo emar')
out_chemo = os.path.join(OUT_DIR, 'mimic_chemo.csv')
chemo_df.to_csv(out_chemo, index=False)
print(f"  → 儲存 {out_chemo}  ({len(chemo_df):,} 筆)\n")


# ─── 2b. G-CSF emar → mimic_gcsf.csv ─────────────────────────────────────────
# 不當 feature 用, 只用來排除受 G-CSF 影響的 anchors (見 mimic_gcsf.py)
GCSF_REGEX = (
    r'filgrastim|lenograstim|eflapegrastim|neupogen|neulasta|zarxio|granix|'
    r'nivestym|releuko|udenyca|fulphila|ziextenzo|rolvedon'
)
gcsf_sql = f"""
WITH {SOLID_PTS_CTE}
SELECT
  e.subject_id                AS Newid,
  CAST(e.charttime AS STRING) AS time,
  e.medication                AS item
FROM `{MIMIC_DS}.emar` e
WHERE e.subject_id IN (SELECT subject_id FROM solid_pts)
  AND REGEXP_CONTAINS(LOWER(e.medication), r'{GCSF_REGEX}')
  AND e.charttime IS NOT NULL
ORDER BY Newid, time
"""

gcsf_df = run_query(gcsf_sql, 'G-CSF emar')
out_gcsf = os.path.join(OUT_DIR, 'mimic_gcsf.csv')
gcsf_df.to_csv(out_gcsf, index=False)
print(f"  → 儲存 {out_gcsf}  ({len(gcsf_df):,} 筆)\n")


# ─── 3. Patient demographics + cancer category → mimic_patients.csv ──────────
patient_sql = f"""
WITH {SOLID_PTS_CTE},
cancer_diag AS (
  -- 每位病人取 seq_num 最小的癌症診斷
  SELECT
    subject_id,
    icd_code,
    CASE
      WHEN SUBSTR(icd_code,1,3) BETWEEN 'C00' AND 'C14'
        OR SUBSTR(icd_code,1,3) IN ('C30','C31','C32') THEN 'Head and Neck'
      WHEN SUBSTR(icd_code,1,3) IN ('C33','C34')        THEN 'Lung'
      WHEN SUBSTR(icd_code,1,3) = 'C50'                 THEN 'Breast'
      WHEN SUBSTR(icd_code,1,3) IN ('C22','C23','C24')  THEN 'Liver/Biliary'
      WHEN SUBSTR(icd_code,1,3) BETWEEN 'C18' AND 'C21' THEN 'Colorectal'
      WHEN SUBSTR(icd_code,1,3) BETWEEN 'C15' AND 'C17'
        OR SUBSTR(icd_code,1,3) IN ('C25','C26')        THEN 'GI (Other)'
      WHEN SUBSTR(icd_code,1,3) BETWEEN 'C51' AND 'C58' THEN 'Gynecologic'
      WHEN SUBSTR(icd_code,1,3) BETWEEN 'C60' AND 'C68' THEN 'GU'
      WHEN SUBSTR(icd_code,1,3) IN ('C43','C44')        THEN 'Skin'
      WHEN SUBSTR(icd_code,1,3) BETWEEN 'C70' AND 'C72' THEN 'CNS'
      WHEN SUBSTR(icd_code,1,3) BETWEEN 'C81' AND 'C88' THEN 'Hematologic'
      ELSE 'Other'
    END AS cancer_category,
    ROW_NUMBER() OVER (
      PARTITION BY subject_id ORDER BY seq_num ASC
    ) AS rn
  FROM `{MIMIC_DS}.diagnoses_icd`
  WHERE subject_id IN (SELECT subject_id FROM solid_pts)
    AND icd_version = 10
    AND SUBSTR(icd_code, 1, 1) = 'C'
),
first_adm AS (
  SELECT subject_id, CAST(MIN(admittime) AS STRING) AS dx_date
  FROM `{MIMIC_DS}.admissions`
  WHERE subject_id IN (SELECT subject_id FROM solid_pts)
  GROUP BY subject_id
)
SELECT
  p.subject_id                               AS Newid,
  COALESCE(cd.cancer_category, 'Unknown')    AS cancer_category,
  COALESCE(cd.icd_code, 'Unknown')           AS primary_site,
  fa.dx_date,
  p.gender                                   AS sex,
  p.anchor_age                               AS age_at_dx
FROM `{MIMIC_DS}.patients` p
JOIN solid_pts sp ON p.subject_id = sp.subject_id
LEFT JOIN cancer_diag cd ON p.subject_id = cd.subject_id AND cd.rn = 1
LEFT JOIN first_adm fa ON p.subject_id = fa.subject_id
"""

patient_df = run_query(patient_sql, 'patient demographics')
out_pt = os.path.join(OUT_DIR, 'mimic_patients.csv')
patient_df.to_csv(out_pt, index=False)
print(f"  → 儲存 {out_pt}  ({len(patient_df):,} 筆)\n")


# ─── 完成 ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print("✓ 抽取完成！")
print(f"  輸出目錄: {OUT_DIR}")
print(f"  mimic_lab.csv      : {len(lab_df):>7,} 筆")
print(f"  mimic_chemo.csv    : {len(chemo_df):>7,} 筆")
print(f"  mimic_patients.csv : {len(patient_df):>7,} 筆")
print(f"  mimic_gcsf.csv     : {len(gcsf_df):>7,} 筆")
print()
print("下一步: 跑 mimic_adapter.py 把這三個 CSV 轉成 frozen model 能吃的格式")
