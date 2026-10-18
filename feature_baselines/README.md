# Feature baselines

The supervised reference rows of the results table for all five tasks: TF-IDF + logistic
regression, structured logistic regression and XGBoost, fit on the same train/test split as
CLIP-H and scored at the same operating point. No API calls.

The models, column rules and scoring are those of `clip_h/mortality/mimic_eval/baselines_ci.py`,
which the mortality and discharge runner notebooks also call (cell "Evaluation: feature
baselines"). For those two tasks the script reads the same feature frame as the runners
(`<split>_cohort.parquet`), so it reproduces their baseline rows exactly.

## 1. Process

```
../data/<task>/          train.jsonl / test.jsonl (profile + label), and the feature columns:
                         train_features.csv / test_features.csv (readmission, hosp_los3, icu_los4) or
                         train_cohort.parquet / test_cohort.parquet (mortality, discharge)
      │
      │  tfidf_lr        TF-IDF (20,000 features, 1–2 word phrases) of the profile text → logistic regression
      │  structured_lr   feature columns: numeric (median-imputed, scaled) + one-hot categorical → logistic regression
      │  xgboost         the same feature columns → XGBoost
      │  scoring         top-k cutoff (k = test positives = 500), AUROC and F1 with 95% bootstrap intervals
      ▼
results/<task>/          summary.csv, <baseline>/stats.json
```

## 2. Folder structure

```
feature_baselines/
├── run_baselines.py     what you run: settings at the top, fixed steps below
├── results/             OUTPUT (not in git): results/<task>/
├── requirements.txt
└── README.md
```

## 3. Input files needed

`data/<task>/` at the repository root, written by `mimic_cleaning/run/run_<task>.py`:

| Task | Files |
|---|---|
| `readmission`, `hosp_los3`, `icu_los4` | `train.jsonl`, `test.jsonl`, `train_features.csv`, `test_features.csv` |
| `mortality`, `discharge` | `train.jsonl`, `test.jsonl`, `train_cohort.parquet`, `test_cohort.parquet` |

The files of a split are row-aligned; the script stops if the row counts differ, or if the labels
in the `.jsonl` and the parquet file disagree.

## 4. How to run

```bash
pip install -r requirements.txt
python run_baselines.py readmission     # or hosp_los3, icu_los4, mortality, discharge
```

Prints AUROC and F1 with their intervals and writes `results/<task>/summary.csv`.

## 5. Column rules

| Rule | Meaning |
|---|---|
| Leakage (`LEAK_RULES`) | readmission drops only identifiers and the label (it is predicted at discharge); `hosp_los3` and `icu_los4` also drop every `los*` and `discharge*` column; `mortality` and `discharge` also drop every `death*`, `dod*`, `discharge*` and `los*` column (their cohorts already contain only admission-time features, so this is a second safeguard) |
| Free text (`FREE_TEXT`) | medications, diagnoses, procedures, cultures, vitals, blood pressure and chief complaint go to TF-IDF only |
| Numeric | lab trajectories (reduced to their last value), columns stored as numbers however sparse, and text columns where at least half the values parse as numbers |
| Categorical | `CATEG_NAMES` and other text columns with at most 50 distinct values; higher-cardinality text is dropped |

Column roles and encoders are learned from train only and applied to test unchanged.
