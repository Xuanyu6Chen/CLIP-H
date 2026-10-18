# Runs the mortality cleaning from the raw MIMIC-IV v3.1 tables to the train/test files and profiles: settings at the top, fixed steps below. Functions come from ../src_2.
"""In-hospital mortality, predicted at admission.

    python run/run_mortality.py

Reads the raw MIMIC-IV v3.1 tables in ../mimic-iv-3.1 and writes its files to
../../data/mortality. This file has two parts:

    SETTINGS   everything that can be changed: sample sizes and class mix, test share,
               seed, profile sections, overwrite
    STEPS      fixed: tables -> cohort -> checks -> patient-level split -> balanced draw
               -> profiles -> files -> manifest

The cleaning functions in ../src_2 are the ones the clip_h/mortality runner notebooks used
before they moved here, so the files are the same patients and profiles as the reported runs.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src_2"))

from mimic_general import MORTALITY_FEATURE_COLUMNS, MORTALITY_FORBIDDEN_COLUMNS, load_mimic_tables, validate_cohort
from mimic_clean_mortality import build_cohort_mortality
from sample_split import (feature_signature, patient_level_split, save_cohort, save_split,
                          subsample_cohort, write_manifest)
from profiles import make_profile_mortality, write_profiles, write_profiles_with_labels

# =============================================================================
# SETTINGS: everything that can be changed is in this block
# =============================================================================
MIMIC_VERSION = "3.1"      # raw tables read from ../mimic-iv-<version>/
SEED = 42                  # fixes the split and the draw
N_PATIENTS_TRAIN = 5000    # training admissions after class balancing (None = all)
N_PATIENTS_TEST = 1000     # test admissions after class balancing (None = all)
SUBSAMPLE = "stratified"   # seeded, label-stratified draw ('head' takes the first N rows)
TRAIN_BALANCE = 0.5        # positive share of train after undersampling the majority class
TEST_BALANCE = 0.5         # positive share of test (a balanced 500/500 evaluation set)
TEST_FRACTION = 0.15       # share of patients sent to test before the draw
RICH_PROFILE = True        # the medication, culture and service sections of the profile
OVERWRITE = False          # False = stop instead of replacing files already in ../../data/mortality

# The runner notebooks (clip_h/mortality) check these settings against manifest.json and stop
# when they differ from their own, because the settings are part of the run folder name.

# =============================================================================
# STEPS: fixed. Change them only to change the method.
# =============================================================================
TASK = "mortality"
MIMIC_DIR = HERE.parent / f"mimic-iv-{MIMIC_VERSION}"   # hosp/ and icu/
OUT = HERE.parent.parent / "data" / TASK   # the files are written here: <repo>/data/<task>/
if not (MIMIC_DIR / "hosp").is_dir():
    raise FileNotFoundError(f"No {MIMIC_DIR / 'hosp'}. Put MIMIC-IV v{MIMIC_VERSION}'s hosp/ and icu/ folders in "
                            f"{MIMIC_DIR} (the files needed are listed in the README).")

# ---- 1  tables (cached as parquet in mimic-iv-<version>/_clip_h_cache/ on first read)
tables = load_mimic_tables(str(MIMIC_DIR / "hosp"), task=TASK)

# ---- 2  cohort, leakage guard, checks
cohort = build_cohort_mortality(tables, seed=SEED, rich=RICH_PROFILE)
leaked = set(MORTALITY_FORBIDDEN_COLUMNS) & set(cohort.columns)
assert not leaked, f"Leakage guard: excluded {TASK} columns present in cohort: {sorted(leaked)}"
validate_cohort(cohort, TASK, rich=RICH_PROFILE)   # stops on a missing label, excluded columns or all-empty columns
save_cohort(cohort, OUT, OVERWRITE)
print(f"[{TASK}] cohort {len(cohort):,} admissions | {cohort.shape[1]} columns | positive rate {cohort['y'].mean():.3f}")

# ---- 3  patient-level split, then the class-balanced draw of each half
train_df, test_df = patient_level_split(cohort, test_size=TEST_FRACTION, seed=SEED, label_col="y")
assert set(train_df["subject_id"]) & set(test_df["subject_id"]) == set(), "a patient is in both train and test"
train_df = subsample_cohort(train_df, N_PATIENTS_TRAIN, seed=SEED, label_col="y", method=SUBSAMPLE, pos_frac=TRAIN_BALANCE)
test_df = subsample_cohort(test_df, N_PATIENTS_TEST, seed=SEED, label_col="y", method=SUBSAMPLE, pos_frac=TEST_BALANCE)
print(f"[{TASK}] train {len(train_df):,} (positive rate {train_df['y'].mean():.3f}) | "
      f"test {len(test_df):,} (positive rate {test_df['y'].mean():.3f}) | patient-disjoint")

# ---- 4  profiles
train_profiles = [make_profile_mortality(r) for r in train_df.to_dict("records")]
test_profiles = [make_profile_mortality(r) for r in test_df.to_dict("records")]
y_train = train_df["y"].to_numpy().astype(int)
y_test = test_df["y"].to_numpy().astype(int)
assert len(set(y_train)) >= 2 and len(set(y_test)) >= 2, (
    "Train or test labels are single-class. Increase N_PATIENTS_* or use the full cohort (None).")

# ---- 5  files: rows, features, labels, profiles, profiles with labels
files = save_split(OUT, TASK, train_df, test_df, MORTALITY_FEATURE_COLUMNS, OVERWRITE)
for split, df, profiles, y in (("train", train_df, train_profiles, y_train), ("test", test_df, test_profiles, y_test)):
    write_profiles(profiles, OUT / f"{split}_profile.jsonl", OVERWRITE)
    write_profiles_with_labels(profiles, y, df["hadm_id"].to_numpy(), df["subject_id"].to_numpy(),
                               OUT / f"{split}.jsonl", OVERWRITE)

# ---- 6  manifest
write_manifest(OUT, {
    "task": TASK,
    "mimic_version": MIMIC_VERSION,
    "settings": {"mimic_version": MIMIC_VERSION, "seed": SEED, "n_patients_train": N_PATIENTS_TRAIN, "n_patients_test": N_PATIENTS_TEST,
                 "subsample": SUBSAMPLE, "train_balance": TRAIN_BALANCE, "test_balance": TEST_BALANCE,
                 "test_fraction": TEST_FRACTION, "rich_profile": RICH_PROFILE},
    "featsig": feature_signature(MORTALITY_FEATURE_COLUMNS, MORTALITY_FORBIDDEN_COLUMNS),
    "feature_columns": MORTALITY_FEATURE_COLUMNS,
    "forbidden_columns": MORTALITY_FORBIDDEN_COLUMNS,
    "cohort_rows": len(cohort),
    "label_counts_cohort": {str(k): int(v) for k, v in cohort["y"].value_counts().items()},
    "split": {"train_rows": len(train_df), "test_rows": len(test_df),
              "train_label_counts": {str(k): int(v) for k, v in train_df["y"].value_counts().items()},
              "test_label_counts": {str(k): int(v) for k, v in test_df["y"].value_counts().items()}},
    "files": files,
}, OVERWRITE)
print(f"[{TASK}] written to {OUT}")
