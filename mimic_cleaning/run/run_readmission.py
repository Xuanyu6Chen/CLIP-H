# Runs the readmission cleaning from the raw MIMIC-IV tables to the train/test files and profiles: settings at the top, fixed steps below. Functions come from ../src.
"""30-day readmission, predicted at discharge.

    python run/run_readmission.py

Reads the raw MIMIC-IV tables in ../mimic-iv-3.1 and writes 10 files to
../../data/readmission. This file has two parts:

    SETTINGS   everything that can be changed: window, label rule, sample size
               and class mix, seed, overwrite, feature columns, profile wording
    STEPS      fixed: cohort -> DNR removal -> label -> draw -> split
               -> profiles -> profiles with labels -> manifest
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from mimic_general import (check_dnr, clean_chartevents, clean_diagnoses, clean_icustays,
                           clean_labevents, clean_microbiology, clean_omr, clean_prescriptions,
                           clean_services, clean_transfers, join_on_spine, read_table, save_cohort)
from mimic_clean_readmission import readmission_label, readmission_spine, remove_dnr
from sample_split import (NEVER_FEATURES, check_split, draw_sample, sample_counts, save_split,
                          select_features, split_train_test, write_manifest)
from profiles import write_profiles, write_profiles_with_labels

# =============================================================================
# SETTINGS: everything that can be changed is in this block
# =============================================================================
WINDOW_HOURS = None      # hours after admission that events may come from; None = the whole stay, up to discharge
READMIT_DAYS = 30        # label = 1 when the next admission starts within this many days of discharge
SAMPLE_SIZE = 6000       # patients drawn in total
POSITIVE_SHARE = 0.5     # share of them with label 1 (0.5 = balanced); "prevalence" = the rate found in the cohort
TEST_SIZE = 1000         # patients of the draw sent to test; the rest is train
SEED = 42                # fixes which patients are drawn and how they are split
OVERWRITE = False        # False = stop instead of replacing files already in ../../data/readmission

# The feature columns, in file order. Each must be a column of cohort.csv; the
# run stops if one is missing or is on NEVER (below).
FEATURES = [
    # patient and admission
    "gender", "age", "race", "marital_status", "insurance", "language",
    "admission_type", "admission_location",
    # baseline before admission
    "height_in", "weight_lbs", "bmi", "blood_pressure",
    # coded diagnoses, whole stay
    "diagnoses", "diagnosis_count",
    # labs, whole stay
    "creatinine", "sodium", "potassium", "lactate", "hemoglobin", "wbc", "platelets",
    "bun", "bicarbonate", "glucose", "albumin", "bilirubin", "inr", "troponin",
    "abnormal_lab_count",
    # cultures and medications, whole stay
    "n_cultures", "culture_sites", "medications",
    # service and transfers (always first 24h), ICU, vitals
    "service", "n_transfers_24h", "was_in_icu", "was_in_icu_24h", "los_in_icu", "vitals",
    # stay summary
    "length_of_stay", "discharge_location",
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
        "section": "ADMISSION",
        "lead": "- ",
        "join": "\n- ",
        "parts": [
            "Admission type: {admission_type}.",
            "Admission location: {admission_location}.",
            "Admitting service: {service}.",
            "Unit transfers in first 24h: {n_transfers_24h}.",
        ],
    },
    "meds": {
        "section": "TREATMENT - WHOLE STAY",
        "lead": "- ",
        "parts": [
            "Medications started during the stay: {medications}.",
        ],
    },
    "vitals": {
        "section": "TREATMENT - WHOLE STAY",
        "lead": "- ",
        "parts": [
            "Vital signs during the stay (ranges): {vitals}.",
        ],
    },
    "labs": {
        "section": "LABS - WHOLE STAY, value@hour, * = abnormal",
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
        "section": "INFECTION - WHOLE STAY",
        "show_if": "n_cultures",
        "empty_text": "- No cultures ordered during the stay.",
        "lead": "- ",
        "parts": [
            "{n_cultures} culture(s) ordered - {culture_sites}.",
        ],
    },
    "icu_status": {
        "section": "ICU",
        "show_if": "was_in_icu",
        "empty_text": "- No ICU admission.",
        "lead": "- ",
        "join": "\n- ",
        "parts": [
            "ICU admission: yes.",
            "ICU length of stay: {los_in_icu} days.",
        ],
    },
    "diagnoses": {
        "section": "HOSPITAL COURSE - WHOLE STAY",
        "lead": "- ",
        "parts": [
            "Coded diagnoses ({diagnosis_count}): {diagnoses}.",
        ],
    },
    "stay_summary": {
        "section": "HOSPITAL COURSE - WHOLE STAY",
        "lead": "- ",
        "join": "\n- ",
        "parts": [
            "Index length of stay: {length_of_stay} days.",
            "Discharge disposition: {discharge_location}.",
        ],
    },
}

# =============================================================================
# STEPS: fixed. Change them only to change the method.
# =============================================================================
TASK = "readmission"
LABEL = "readmission"
NEVER = NEVER_FEATURES + [LABEL]          # columns that can never be a feature
OUT = HERE.parent.parent / "data" / TASK  # the 10 files are written here: <repo>/data/<task>/

# ---- 1-3  cohort: spine -> clean tables -> join -> cohort.csv
spine = readmission_spine(read_table("admissions"), read_table("patients"), READMIT_DAYS)
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
])
cohort = save_cohort(cohort, OUT / "cohort.csv", OVERWRITE)

# ---- DNR wording out of the diagnoses (the only column it appears in)
cohort["diagnoses"], dnr_cells = remove_dnr(cohort["diagnoses"])

# ---- 4  label
labelled = readmission_label(cohort, READMIT_DAYS)

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
    "dnr_cells_replaced": dnr_cells,
    "rows_after_label_rule": len(labelled),
    "label_counts_before_draw": {str(k): int(v) for k, v in labelled[LABEL].value_counts().items()},
    "drawn": {"positive": n_positive, "negative": n_negative},
    "settings": {"window_hours": WINDOW_HOURS, "readmit_days": READMIT_DAYS,
                 "sample_size": SAMPLE_SIZE, "positive_share": POSITIVE_SHARE, "test_size": TEST_SIZE, "seed": SEED},
    "feature_columns": FEATURES,
    "label_column": LABEL,
    "split": split,
    "files": files,
    "profiles": {"train": len(train_profiles), "test": len(test_profiles)},
}, OVERWRITE)
print(f"[{TASK}] cohort {len(cohort):,} -> labelled {len(labelled):,} -> drawn {len(sample):,} "
      f"-> train {len(X_train):,} / test {len(X_test):,}; written to {OUT}")
