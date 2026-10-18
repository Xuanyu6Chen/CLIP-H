# Runs the hospital-LOS cleaning from the raw MIMIC-IV tables to the train/test files and profiles: settings at the top, fixed steps below. Functions come from ../src.
"""Hospital length of stay > 3 days, predicted from the first 24 hours.

    python run/run_hosp_los.py

Reads the raw MIMIC-IV tables in ../mimic-iv-3.1 and writes 10 files to
../../data/hosp_los3. This file has two parts:

    SETTINGS   everything that can be changed: window, label rule, sample size
               and class mix, seed, overwrite, feature columns, profile wording
    STEPS      fixed: cohort -> label -> draw -> split -> profiles
               -> profiles with labels -> manifest
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from mimic_general import (check_dnr, clean_chartevents, clean_diagnoses, clean_icustays,
                           clean_labevents, clean_microbiology, clean_omr, clean_outputevents,
                           clean_prescriptions, clean_procedureevents, clean_services,
                           clean_transfers, join_on_spine, los_label, read_table, save_cohort)
from mimic_clean_hosp_los import hosp_los_spine, prior_admissions
from sample_split import (NEVER_FEATURES, check_split, draw_sample, sample_counts, save_split,
                          select_features, split_train_test, write_manifest)
from profiles import write_profiles, write_profiles_with_labels

# =============================================================================
# SETTINGS: everything that can be changed is in this block
# =============================================================================
WINDOW_HOURS = 24        # hours after admission that events may come from
THRESHOLD_HOURS = 72     # label = 1 when the stay is longer than this (3 days)
MIN_LOS_HOURS = 30       # stays shorter than this are dropped (24h observed + 6h gap)
SAMPLE_SIZE = 6000       # patients drawn in total
POSITIVE_SHARE = 0.5     # share of them with label 1 (0.5 = balanced); "prevalence" = the rate found in the cohort
TEST_SIZE = 1000         # patients of the draw sent to test; the rest is train
SEED = 42                # fixes which patients are drawn and how they are split
OVERWRITE = False        # False = stop instead of replacing files already in ../../data/hosp_los3

# The feature columns, in file order. Each must be a column of cohort.csv; the
# run stops if one is missing or is on NEVER (below).
FEATURES = [
    # patient and admission
    "gender", "age", "race", "marital_status", "insurance", "language",
    "admission_type", "admission_location",
    # baseline before admission
    "height_in", "weight_lbs", "bmi", "blood_pressure",
    # labs, first 24h
    "creatinine", "sodium", "potassium", "lactate", "hemoglobin", "wbc", "platelets",
    "bun", "bicarbonate", "glucose", "albumin", "bilirubin", "inr", "troponin",
    "abnormal_lab_count",
    # cultures and medications, first 24h
    "n_cultures", "culture_sites", "medications",
    # service, transfers, ICU within 24h, vitals
    "service", "n_transfers_24h", "was_in_icu_24h", "vitals",
    # organ support, first 24h
    "mechanical_ventilation_flag", "dialysis_flag", "urine_output_ml_24h", "oliguria_flag",
    # admissions before this one
    "n_prior_admissions", "n_prior_admissions_1y",
]

# Wording of the patient profile. Blocks print in this order. A sentence is
# written only when every {column} in it has a value. Keys: section, lead, join,
# parts, show_if, empty_text (explained at the top of ../src/profiles.py).
PROFILE = {
    "fixed": {
        "section": "PATIENT BASELINE",
        "lead": "- ",
        "join": "\n- ",
        "parts": [
            "Age: {age} years.",
            "Sex: {gender}.",
            "Race: {race}.",
            "Marital status: {marital_status}.",
            "Insurance: {insurance}.",
            "Primary language: {language}.",
        ],
    },
    "baseline": {
        "section": "PATIENT BASELINE",
        "lead": "- Baseline ",
        "join": ", ",
        "parts": [
            "height {height_in} in",
            "weight {weight_lbs} lb",
            "BMI {bmi}",
            "blood pressure {blood_pressure}",
        ],
    },
    "service_transfers": {
        "section": "ADMISSION - FIRST 24H",
        "lead": "- ",
        "join": "\n- ",
        "parts": [
            "Admission type: {admission_type}.",
            "Admission location: {admission_location}.",
            "Admitting service: {service}.",
            "Unit transfers in first 24h: {n_transfers_24h}.",
        ],
    },
    "prior_utilization": {
        "section": "ADMISSION - FIRST 24H",
        "lead": "- ",
        "parts": [
            "Prior hospital admissions: {n_prior_admissions} total, {n_prior_admissions_1y} in the past year.",
        ],
    },
    "meds": {
        "section": "EARLY COURSE - FIRST 24H",
        "lead": "- ",
        "parts": [
            "Medications started, first 24h: {medications}.",
        ],
    },
    "icu_support": {
        "section": "EARLY COURSE - FIRST 24H",
        "lead": "- ",
        "join": "\n- ",
        "parts": [
            "Mechanical ventilation in first 24h: {mechanical_ventilation_flag}; dialysis: {dialysis_flag}.",
            "Urine output, first 24h: {urine_output_ml_24h} mL (oliguria: {oliguria_flag}).",
        ],
    },
    "vitals": {
        "section": "EARLY COURSE - FIRST 24H",
        "lead": "- ",
        "parts": [
            "Vital signs, first 24h (ranges): {vitals}.",
        ],
    },
    "labs": {
        "section": "LABS - FIRST 24H, value@hour, * = abnormal",
        "lead": "- ",
        "join": "\n- ",
        "parts": [
            "Creatinine {creatinine}",
            "BUN {bun}",
            "Sodium {sodium}",
            "Potassium {potassium}",
            "Bicarbonate {bicarbonate}",
            "Glucose {glucose}",
            "Lactate {lactate}",
            "Hemoglobin {hemoglobin}",
            "WBC {wbc}",
            "Platelets {platelets}",
            "INR {inr}",
            "Albumin {albumin}",
            "Bilirubin {bilirubin}",
            "Troponin {troponin}",
            "Abnormal results: {abnormal_lab_count}",
        ],
    },
    "micro": {
        "section": "INFECTION - FIRST 24H",
        "show_if": "n_cultures",
        "empty_text": "- No cultures ordered in first 24h.",
        "lead": "- ",
        "parts": [
            "{n_cultures} culture(s) ordered - {culture_sites}.",
        ],
    },
    "icu_status_24h": {
        "section": "ICU",
        "show_if": "was_in_icu_24h",
        "empty_text": "- No ICU admission in first 24h.",
        "lead": "- ",
        "parts": [
            "ICU admission within first 24h: yes.",
        ],
    },
}

# =============================================================================
# STEPS: fixed. Change them only to change the method.
# =============================================================================
TASK = "hosp_los3"
LABEL = "los_gt3"
# Whole-stay facts: each is a consequence of a long stay, so never a feature here.
# That includes the coded diagnoses, which are assigned over the whole stay.
NEVER = NEVER_FEATURES + [LABEL, "los_in_icu", "was_in_icu", "discharge_location",
                          "diagnoses", "diagnosis_count"]
OUT = HERE.parent.parent / "data" / TASK  # the 10 files are written here: <repo>/data/<task>/

# ---- 1-3  cohort: spine -> clean tables -> join -> cohort.csv
spine = hosp_los_spine(read_table("admissions"), read_table("patients"))
cohort = join_on_spine(spine, [
    clean_omr(spine),
    clean_diagnoses(spine),
    clean_labevents(spine, WINDOW_HOURS),
    clean_microbiology(spine, WINDOW_HOURS),
    clean_prescriptions(spine, WINDOW_HOURS),
    clean_services(spine),
    clean_transfers(spine),
    clean_icustays(spine),
    clean_chartevents(spine, WINDOW_HOURS),
    prior_admissions(spine),
    clean_outputevents(spine, WINDOW_HOURS),
    clean_procedureevents(spine, WINDOW_HOURS),
])
cohort = save_cohort(cohort, OUT / "cohort.csv", OVERWRITE)

# ---- 4  label
labelled = los_label(cohort, LABEL, THRESHOLD_HOURS, MIN_LOS_HOURS)

# ---- 5-6  draw the sample, feature columns
n_positive, n_negative = sample_counts(labelled, LABEL, SAMPLE_SIZE, POSITIVE_SHARE)
sample = draw_sample(labelled, LABEL, n_positive, n_negative, SEED)
X = select_features(sample, FEATURES, NEVER)
y = sample[[LABEL]].reset_index(drop=True)

# ---- 7-8  split, check, save
X_train, X_test, y_train, y_test, train_rows, test_rows = split_train_test(X, y, TEST_SIZE, SEED)
split = check_split(X_train, X_test, y_train, y_test, sample["subject_id"],
                    train_rows, test_rows, NEVER, TEST_SIZE)
check_dnr(X_train, "train features")
check_dnr(X_test, "test features")
files = save_split(OUT, X_train, X_test, y_train, y_test, OVERWRITE)

# ---- 9-10  profiles, DNR check
train_profiles = write_profiles(OUT / "train_features.csv", PROFILE, OUT / "train_profile.jsonl", OVERWRITE)
test_profiles = write_profiles(OUT / "test_features.csv", PROFILE, OUT / "test_profile.jsonl", OVERWRITE)
check_dnr(train_profiles, "train profiles")
check_dnr(test_profiles, "test profiles")

# ---- 11  profile + label of each patient on one line: the file the method folders read
write_profiles_with_labels(train_profiles, y_train[LABEL], OUT / "train.jsonl", OVERWRITE)
write_profiles_with_labels(test_profiles, y_test[LABEL], OUT / "test.jsonl", OVERWRITE)

# ---- 12  manifest
write_manifest(OUT, {
    "task": TASK,
    "cohort_file": OUT / "cohort.csv",
    "cohort_rows": len(cohort),
    "rows_after_label_rule": len(labelled),
    "label_counts_before_draw": {str(k): int(v) for k, v in labelled[LABEL].value_counts().items()},
    "drawn": {"positive": n_positive, "negative": n_negative},
    "settings": {"window_hours": WINDOW_HOURS, "threshold_hours": THRESHOLD_HOURS,
                 "min_los_hours": MIN_LOS_HOURS, "sample_size": SAMPLE_SIZE, "positive_share": POSITIVE_SHARE,
                 "test_size": TEST_SIZE, "seed": SEED},
    "feature_columns": FEATURES,
    "label_column": LABEL,
    "split": split,
    "files": files,
    "profiles": {"train": len(train_profiles), "test": len(test_profiles)},
}, OVERWRITE)
print(f"[{TASK}] cohort {len(cohort):,} -> labelled {len(labelled):,} -> drawn {len(sample):,} "
      f"-> train {len(X_train):,} / test {len(X_test):,}; written to {OUT}")
