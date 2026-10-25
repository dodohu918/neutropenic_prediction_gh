# Event-Anchored Prediction of Chemotherapy-Associated Neutropenia

Code accompanying the study *Event-Anchored Prediction of Chemotherapy-Associated Neutropenia from Longitudinal Electronic Health Records: External Validation and Uncertainty Quantification*.

At each chemotherapy cycle start (the "anchor"), models use a 28-day look-back window of laboratory and treatment history to predict severe neutropenia (ANC < 500) within the next 7, 14, 21, or 28 days. Models are developed on a Taiwanese hospital cohort (CCH) and externally validated on MIMIC-IV.

> **No data are included in this repository.** The development cohort is private hospital data and cannot be shared. MIMIC-IV is available to credentialed users from [PhysioNet](https://physionet.org/content/mimiciv/) under its data use agreement. Prediction files, logs, figures, and derived CSVs are excluded for the same reason.

## Repository layout

```
CCH/                               # Development cohort: event-anchored pipeline
├── pipeline_common.py             # Shared preprocessing / feature / cohort logic
├── eventanchor_common.py          # Builds the event-anchored dataset
├── event_anchored_eval.py         # Primary LightGBM evaluation
├── tune_lightgbm_eventanchored.py, tune_all_eventanchored.py   # Hyperparameter tuning
├── mlp_vs_lgb_eventanchored.py    # LightGBM / XGBoost / LR / MLP comparison
├── fttransformer_eventanchored.py, tabnet_eventanchored.py, tabr_eventanchored.py
├── sundial_eventanchored.py       # Sundial time-series foundation-model baseline
├── uq_eventanchored.py, uq_selective_risk_coverage.py, plot_risk_coverage.py   # Uncertainty quantification
├── shap_analysis.py               # SHAP feature importance
├── subgroup_eventanchored.py, subgroup_cancer_lgb.py   # Cancer-subtype subgroup analyses
├── cohort_flowchart.py            # Cohort selection flowchart
├── cch_fn_proxy.py                # Febrile-neutropenia proxy (blood culture + CRP/PCT)
└── best_params_*.json, tune_all_ea_thr0.5_seed42.json  # Tuning results (see below)
MIMIC/                             # External validation on MIMIC-IV
├── mimic_extract.py               # Extracts MIMIC-IV (BigQuery) into the pipeline's input format
├── mimic_fn_labels.py             # Cohort correction + febrile-neutropenia proxy labels
├── mimic_gcsf.py                  # Excludes anchors contaminated by G-CSF (14 days before T through the label event)
├── mimic_eval*.py                 # CCH-trained models evaluated on MIMIC-IV
└── (self-contained copies of the shared pipeline modules)
```

The `.json` files are saved hyperparameter-tuning results: `best_params_*.json` from `tune_lightgbm_eventanchored.py`, and `tune_all_ea_thr0.5_seed42.json` from `tune_all_eventanchored.py`. `MIMIC/mimic_eval_tuned.py` loads the tuned LightGBM/XGBoost parameters from the latter.

## Running

The scripts will not run as-is because the data are not included; they are provided so the methodology can be inspected and adapted. `pipeline_common.py` expects the CCH data at the repo root (`懶人包/filtered_test_result.csv`, `chemo_result.csv`; `cch_fn_proxy.py` also reads `懶人包/blood_culture.csv` with columns `Newid, SIGNIN`) and the cohort annotation files (`patient_cancer_subgroups.csv`, `lymphoma_marrow_adjudication.csv`) inside `CCH/`.

For MIMIC-IV, obtain access, then run `mimic_extract.py` (queries MIMIC-IV v3.1 on BigQuery; set `PROJECT_ID` to your own Google Cloud project) and `mimic_fn_labels.py` (reads a local copy at `MIMIC/physionet.org/files/mimiciv/3.1`). The `mimic_eval_*.py` scripts import from `CCH/` and train on the CCH cohort before evaluating on MIMIC-IV.

Most code comments are written in Traditional Chinese.
