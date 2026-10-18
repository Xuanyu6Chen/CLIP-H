# MIMIC-IV cleaning

This folder cleans raw MIMIC-IV hospital records for the five prediction tasks.
For each task one script reads the raw tables and writes a cohort table,
train/test feature and label files, and a text profile of each patient.

All five tasks read MIMIC-IV **version 3.1** from `mimic-iv-3.1/`. They come in two
groups with their own code, because they were built separately and the reported
results depend on each exactly as it is:

- `readmission`, `hosp_los3`, `icu_los4`: functions in `src/`, one row per
  patient. Sections 1-7 describe them.
- `mortality`, `discharge`: functions in `src_2/`, one row per admission.
  Section 8 describes them.

| Task | Question | Run script | Output folder |
|---|---|---|---|
| `readmission` | Is the patient admitted again within 30 days of discharge? | `run/run_readmission.py` | `../data/readmission/` |
| `hosp_los3` | Does the hospital stay last longer than 3 days? | `run/run_hosp_los.py` | `../data/hosp_los3/` |
| `icu_los4` | Does the ICU stay last longer than 4 days? | `run/run_icu_los.py` | `../data/icu_los4/` |
| `mortality` | Does the patient die during the admission? | `run/run_mortality.py` | `../data/mortality/` |
| `discharge` | Is the patient discharged home without services? | `run/run_discharge.py` | `../data/discharge/` |

The raw data is not included. Everything else needed is in this folder.

## 1. Process: raw data → cohort → profiles

```
mimic-iv-3.1/                the raw MIMIC-IV tables (14 .csv.gz files, section 2)
      │
      │  steps 1-3    pick one admission per patient (the spine), clean each raw
      │               table into a few columns per admission, join them
      ▼
cohort.csv                   one row per patient, every cleaned column
      │
      │  step 4       add the label, drop the rows the label rule excludes
      │  steps 5-6    draw the sample, keep only the feature columns
      │  steps 7-8    train/test split, checks
      ▼
train_features.csv  test_features.csv  train_labels.csv  test_labels.csv
      │
      │  steps 9-10   fill the profile wording for every row
      ▼
train_profile.jsonl  test_profile.jsonl
      │
      │  step 11      put each profile and its label on one line
      ▼
train.jsonl  test.jsonl      what the methods read
      │
      │  step 12
      ▼
manifest.json                settings and counts of the run
```

Words used in this file:

- **Spine**: the table with one row per patient, holding the admission (or ICU
  stay) the task predicts on. Every other table is joined onto it.
- **Clock**: the time the stay starts. Hours are counted from it.
- **Window**: the hours after the clock start from which events may be used.

## 2. Folder structure

```
mimic_cleaning/
├── mimic-iv-3.1/                    INPUT: put the raw MIMIC-IV v3.1 tables here (list below and 8.2). Read only
├── run/                             what you run: one script per task, settings at its top
│   ├── run_readmission.py
│   ├── run_hosp_los.py
│   ├── run_icu_los.py
│   ├── run_mortality.py             (section 8)
│   ├── run_discharge.py             (section 8)
│   └── preprocess_labevents.py      optional, mortality and discharge: builds their lab cache faster
├── src/                             functions only, nothing runs here
│   ├── mimic_general.py             used by 2+ tasks: read tables, time window, 11 table cleaners, join, save cohort, LOS label, DNR check
│   ├── mimic_clean_readmission.py   readmission only: spine, DNR removal, 30-day label
│   ├── mimic_clean_hosp_los.py      hospital LOS only: spine, prior admissions
│   ├── mimic_clean_icu_los.py       ICU LOS only: spine
│   ├── sample_split.py              sample draw, feature columns, split, checks, save, manifest
│   └── profiles.py                  profile writer
├── src_2/                           mortality and discharge functions (section 8)
│   ├── mimic_general.py             used by both: cleaning helpers, column lists, read tables, feature builders, cohort check
│   ├── mimic_clean_mortality.py     mortality only: label, cohort
│   ├── mimic_clean_discharge.py     discharge only: label, admission-origin exclusion, cohort
│   ├── sample_split.py              patient-level split, balanced draw, file writers, manifest
│   └── profiles.py                  profile template and writers
├── requirements.txt                 pandas, numpy, scikit-learn, pyarrow, tqdm
└── README.md

../data/                             OUTPUT (repo root, not in git): written by the run scripts, one folder per task
├── readmission/
├── hosp_los3/
└── icu_los4/                        each task folder gets the same 10 files:
    ├── cohort.csv                   one row per patient, every cleaned column
    ├── train_features.csv           the feature columns only
    ├── test_features.csv
    ├── train_labels.csv             one column, the label
    ├── test_labels.csv
    ├── train_profile.jsonl          one line per row of the features: {"idx", "profile"}
    ├── test_profile.jsonl
    ├── train.jsonl                  profile and label together, one line per patient: {"id", "profile", "label"}
    ├── test.jsonl
    └── manifest.json                settings and counts of the run
```

Row `i` of a features file, its labels file, its profile file (`idx` = `i`) and
`train.jsonl` / `test.jsonl` (`id` = `i`) is the same patient. The feature files
hold no patient IDs.

Where a function goes: used by two or more tasks → `mimic_general.py`; used by
one task → that task's own file.

### Raw files needed

MIMIC-IV needs credentialed access (PhysioNet), so `mimic-iv-3.1/` is empty
here. It has the name of the folder that PhysioNet's MIMIC-IV v3.1 zip file
unpacks to: put MIMIC-IV's own `hosp/` and `icu/` folders in it (or unpack the
zip here). Only these 14 files are read (about 7 GB):

| File in `mimic-iv-3.1/` | Size | Used for | Tasks |
|---|---|---|---|
| `hosp/patients.csv.gz` | 2.8 MB | gender, age | all |
| `hosp/admissions.csv.gz` | 20 MB | the admissions to choose from, admission details, prior admissions | all |
| `hosp/omr.csv.gz` | 44 MB | height, weight, BMI, blood pressure | all |
| `hosp/diagnoses_icd.csv.gz` | 34 MB | coded diagnoses | all |
| `hosp/d_icd_diagnoses.csv.gz` | 0.9 MB | ICD code → title | all |
| `hosp/labevents.csv.gz` | 2.6 GB | 14 labs | all |
| `hosp/microbiologyevents.csv.gz` | 118 MB | cultures | all |
| `hosp/prescriptions.csv.gz` | 606 MB | medications | all |
| `hosp/services.csv.gz` | 8.6 MB | admitting service | all |
| `hosp/transfers.csv.gz` | 46 MB | unit transfers | all |
| `icu/icustays.csv.gz` | 3.3 MB | ICU stays, ICU flags | all |
| `icu/chartevents.csv.gz` | 3.5 GB | vital signs | all |
| `icu/outputevents.csv.gz` | 49 MB | urine output | `hosp_los3`, `icu_los4` |
| `icu/procedureevents.csv.gz` | 24 MB | ventilation, dialysis | `hosp_los3`, `icu_los4` |

## 3. How to run

```bash
pip install -r requirements.txt
python run/run_readmission.py
python run/run_hosp_los.py
python run/run_icu_los.py
python run/run_mortality.py     # section 8
python run/run_discharge.py     # section 8
```

- Tested with Python 3.15.0, pandas 3.0.6 and numpy 2.5.3.
- Each script is independent and can be run from any working directory.
- A run reads the raw tables, takes several minutes (most of it reading
  `labevents` and `chartevents`), and writes the 10 files to its folder in
  `../data/`.
- **A run never replaces a file.** If a file it is about to write is already
  there, it stops. The first file written is `cohort.csv`, after the cohort is
  built, so empty the task's output folder before a run, or set
  `OVERWRITE = True` at the top of the script.
- If a raw file is missing, the run stops at once and names it.

## 4. Run script structure

Each script is one file with two parts: the settings, then the steps. The three
scripts differ only in the settings, the spine function, and which table
cleaners they call.

```
run_readmission.py · run_hosp_los.py · run_icu_los.py
│
├── SETTINGS  (everything that can be changed)
│   ├── WINDOW_HOURS, label rule, SAMPLE_SIZE, POSITIVE_SHARE, TEST_SIZE, SEED, OVERWRITE
│   ├── FEATURES        the exact feature columns, in order
│   └── PROFILE         the wording of the patient profile
│
└── STEPS  (fixed)
    ├── TASK, LABEL, NEVER (columns that can never be a feature), OUT
    ├── 1-3   cohort    spine → table cleaners → join_on_spine → save_cohort      → cohort.csv
    ├──       DNR       remove_dnr on `diagnoses`                                 (run_readmission.py only)
    ├── 4     label     readmission_label  or  los_label
    ├── 5-6   sample    sample_counts → draw_sample → select_features
    ├── 7-8   split     split_train_test → check_split → check_dnr → save_split   → 4 csv files
    ├── 9-10  profiles  write_profiles → check_dnr                                → train_profile.jsonl, test_profile.jsonl
    ├── 11    + labels  write_profiles_with_labels                                → train.jsonl, test.jsonl
    └── 12    manifest  write_manifest                                            → manifest.json
```

Step numbers match the comments in the scripts and the process in section 1.

## 5. Inputs and outputs of every file

| File | Reads | Writes |
|---|---|---|
| `run/run_readmission.py` | 12 raw files in `mimic-iv-3.1/` | 10 files in `../data/readmission/` |
| `run/run_hosp_los.py` | 14 raw files in `mimic-iv-3.1/` | 10 files in `../data/hosp_los3/` |
| `run/run_icu_los.py` | 14 raw files in `mimic-iv-3.1/` | 10 files in `../data/icu_los4/` |
| `run/run_mortality.py`, `run/run_discharge.py` | 10 raw files in `mimic-iv-3.1/` | 12 files in `../data/<task>/` (section 8) |
| `src/mimic_general.py` | the raw files, each `clean_*` function its own (table in 7.2); `cohort.csv`, read back right after it is written | `cohort.csv` |
| `src/mimic_clean_readmission.py` | nothing from disk; the run script passes `admissions` and `patients` in | nothing |
| `src/mimic_clean_hosp_los.py` | `hosp/admissions.csv.gz` (in `prior_admissions`) | nothing |
| `src/mimic_clean_icu_los.py` | nothing from disk; the run script passes `admissions`, `patients`, `icustays` in | nothing |
| `src/sample_split.py` | nothing from disk | `train_features.csv`, `test_features.csv`, `train_labels.csv`, `test_labels.csv`, `manifest.json` |
| `src/profiles.py` | `train_features.csv`, `test_features.csv` | `train_profile.jsonl`, `test_profile.jsonl`, `train.jsonl`, `test.jsonl` |

The run scripts do the reading and writing through the `src/` functions; the
first three rows are the totals per task. Nothing in this folder reads
`../data/`. Every method in the repository (`clip_h/`, `direct_llm_baseline/`,
`direct_llm_hypotheses/`) reads its task's files from `data/<task>/`.

## 6. What you can change, what is fixed

**Can be changed**: the SETTINGS block at the top of each run script.

| Setting | Meaning | In |
|---|---|---|
| `WINDOW_HOURS` | hours after the clock start that events may come from; `None` = the whole stay, up to discharge | all three |
| `READMIT_DAYS` | label = 1 when the next admission starts within this many days of discharge | `run_readmission.py` |
| `THRESHOLD_HOURS` | label = 1 when the stay is longer than this | `run_hosp_los.py`, `run_icu_los.py` |
| `MIN_LOS_HOURS` | stays shorter than this are dropped | `run_hosp_los.py`, `run_icu_los.py` |
| `SAMPLE_SIZE` | patients drawn in total | all three |
| `POSITIVE_SHARE` | share of the sample with label 1: a number between 0 and 1 (`0.5` = balanced), or `"prevalence"` to keep the rate found in the cohort | all three |
| `TEST_SIZE` | patients of the draw sent to test; the rest is train | all three |
| `SEED` | fixes which patients are drawn and how they are split | all three |
| `OVERWRITE` | `False` = stop instead of replacing a file that is already there | all three |
| `FEATURES` | the feature columns, in file order | all three |
| `PROFILE` | the wording of the patient profile | all three |

**Fixed**: the STEPS block of each run script and everything in `src/`. That
covers the spine rules, the table rules in 7.2, the item ID lists (labs,
vitals, urine, ventilation, dialysis), `ZERO_FILL`, `DNR_PATTERNS`, `NEVER` and
`NEVER_FEATURES`, the folder names `mimic-iv-3.1/` and `../data/`, and the
random calls in `sample_split.py` (an equivalent-looking call picks different
patients).

Things to know before changing a setting:

- Any change to the numbers changes which patients are drawn, so every result
  built on the files changes with it.
- A feature must be a column of `cohort.csv` (or `length_of_stay`, which the
  readmission label step adds) and must not be on `NEVER`, or the run stops.
- A feature with no sentence in `PROFILE` is in the feature file but not in the
  profile.
- `service`, `n_transfers_24h` and `was_in_icu_24h` stay at 24h whatever
  `WINDOW_HOURS` is. Column names and profile sentences that say "24h" do not
  change by themselves.

`PROFILE` is a dictionary of blocks, printed in order. Each block may use these
keys and no others:

| Key | Meaning |
|---|---|
| `section` | heading the block sits under; printed once, as `[SECTION]`. Blocks that share a section must sit next to each other |
| `lead` | text before the block |
| `join` | text between its sentences |
| `parts` | the sentences, with `{column}` slots. A sentence is printed only when every slot has a value. A slot that is not a feature column makes its sentence never print, with no error |
| `show_if` | a column that must be non-zero for `parts` to print; when it is 0 the block prints `empty_text`, when it is empty the block is skipped |
| `empty_text` | printed when no sentence could be written |

Values are printed as stored, except numbers: `3.0` prints as `3`, others keep
at most 2 decimals.

## 7. Cleaning rules

### 7.1 Per task

| Rule | `readmission` | `hosp_los3` | `icu_los4` |
|---|---|---|---|
| Predicted at | discharge | 24h after admission | 24h after ICU entry |
| Spine: one row per patient | first admission that ended alive, paired with the admission after it | last admission | last ICU stay |
| Left out of the spine | patients with no live discharge; next admission starting before the discharge | nobody | ICU stays with no in or out time |
| Clock: start → end | `admittime` → `dischtime` | `admittime` → `dischtime` | ICU `intime` → `outtime` (saved as `admittime`, `dischtime`) |
| Window (`WINDOW_HOURS`) | whole stay, start to discharge (`None`) | first 24h | first 24h from ICU entry, also for the hospital tables |
| Table cleaners called | 9 | 12 | 11 |
| DNR wording | replaced by `[removed]` in `diagnoses`; the row stays | `diagnoses` is not a feature | `diagnoses` is not a feature |
| Dropped at the label step | ended in death or negative gap (the spine already excludes both) | stays under 30h (`MIN_LOS_HOURS`) | stays under 30h (`MIN_LOS_HOURS`) |
| Label = 1 when | `readmission`: next admission 0 to 30 whole days after discharge (`READMIT_DAYS`) | `los_gt3`: stay > 72h (`THRESHOLD_HOURS`) | `icu_los_gt4`: stay > 96h (`THRESHOLD_HOURS`) |
| Rows in `cohort.csv` | 218,079 | 223,452 | 65,355 |

- "Ended alive" = `hospital_expire_flag` 0, no `deathtime`, `discharge_location` not "died".
- Readmission keeps patients who died within 30 days without a readmission
  (label 0); they are only marked in `death_within_30d_no_readmission`.
- Readmission's `cohort.csv` is the cohort as built and still has the DNR
  wording in `diagnoses`. It is replaced by `[removed]` right after that file is
  saved, so the features, the profiles and `train.jsonl` / `test.jsonl` have none.
- 30h floor = 24h observed + 6h gap. A stay ending at hour 14 has no labs at
  hour 20, so its thin record would give away the label.

### 7.2 Table cleaners (`src/mimic_general.py`, one per raw table)

"Task" window = the window in 7.1. Used by: R readmission, H hospital LOS, I ICU LOS.

| Raw table | Function | Columns | Rule | Window | Used by |
|---|---|---|---|---|---|
| `omr` | `clean_omr` | `height_in`, `weight_lbs`, `bmi`, `blood_pressure` | value dated nearest on or before the clock start; if none, nearest after. Joined on the patient | none (outpatient) | R H I |
| `diagnoses_icd`, `d_icd_diagnoses` | `clean_diagnoses` | `diagnoses`, `diagnosis_count` | ICD titles in `seq_num` order, no repeats, joined with `; ` | whole stay | R H I |
| `labevents` | `clean_labevents` | 14 labs, `abnormal_lab_count` | listed item IDs only. Each lab is a sequence `value*@Nh; ...` by hour, `*` = flagged abnormal; readings with no number skipped | task | R H I |
| `microbiologyevents` | `clean_microbiology` | `n_cultures`, `culture_sites` | distinct specimens; specimen types, no repeats | task | R H I |
| `prescriptions` | `clean_prescriptions` | `medications` | `drug_type` MAIN only; names of drugs started in the window, no repeats | task | R H I |
| `services` | `clean_services` | `service` | latest service in the first 24h, as a readable name | always 24h | R H I |
| `transfers` | `clean_transfers` | `n_transfers_24h` | care-unit changes; discharge events, `UNKNOWN` and `Discharge Lounge` dropped, repeats collapsed | always 24h | R H I |
| `icustays` | `clean_icustays` | `was_in_icu`, `los_in_icu`, `was_in_icu_24h` | any ICU stay; total ICU days; ICU entry within 24h of the clock start | whole stay, whole stay, 24h | R H I |
| `chartevents` | `clean_chartevents` | `vitals` | HR, SBP, SpO2, RR, Temp (°F converted to °C) as `name min–max` | task | R H I |
| `outputevents` | `clean_outputevents` | `urine_output_ml_24h`, `oliguria_flag` | urine item IDs summed; flag when under 400 mL | task | H I |
| `procedureevents` | `clean_procedureevents` | `mechanical_ventilation_flag`, `dialysis_flag` | any ventilation / dialysis item started in the window | task | H I |
| `admissions` | `prior_admissions` (`src/mimic_clean_hosp_los.py`) | `n_prior_admissions`, `n_prior_admissions_1y` | admissions started strictly before this one: all, and in the past 365 days | before admission | H |

- Join: every cleaner is left-joined onto the spine by `hadm_id`. No events →
  empty cell; count and flag columns → 0 (`ZERO_FILL`). For the ICU-only tables
  0 therefore also means "not recorded": a patient outside the ICU has
  `urine_output_ml_24h` 0 and `oliguria_flag` 0.
- `service` and `n_transfers_24h` stay at 24h in every task: the profile calls
  it the admitting service, and under a whole-stay window "latest" would become
  the discharge service.

### 7.3 Feature columns

Only the columns listed in a script's `FEATURES` become features.

| Columns | `readmission` | `hosp_los3` | `icu_los4` |
|---|---|---|---|
| `gender`, `age`, `race`, `marital_status`, `insurance`, `language`, `admission_type`, `admission_location` | yes | yes | yes |
| `height_in`, `weight_lbs`, `bmi`, `blood_pressure` | yes | yes | yes |
| 14 labs, `abnormal_lab_count` | whole stay | 24h | 24h |
| `n_cultures`, `culture_sites`, `medications` | whole stay | 24h | 24h |
| `vitals` | whole stay | 24h | 24h |
| `service`, `n_transfers_24h` | yes | yes | yes |
| `was_in_icu_24h` | yes | yes | never: every patient is in the ICU |
| `diagnoses`, `diagnosis_count` | yes | never: coded over the whole stay | never |
| `was_in_icu`, `los_in_icu` | yes | never: whole-stay fact | never: ICU days are the label |
| `discharge_location` | yes | never | never |
| `length_of_stay` | yes | not built | not built |
| `mechanical_ventilation_flag`, `dialysis_flag`, `urine_output_ml_24h`, `oliguria_flag` | not built | 24h | 24h |
| `n_prior_admissions`, `n_prior_admissions_1y` | not built | yes | not built |
| **Total** | **40** | **40** | **37** |

Never a feature in any task (`NEVER_FEATURES`, `src/sample_split.py`): `id`,
`subject_id`, `hadm_id`, `stay_id`, `admittime`, `dischtime`, `edregtime`,
`edouttime`, `deathtime`, `hospital_expire_flag`, `next_admit_time`,
`readmission_day`, `has_readmission`, `_los_hours`, and the label column.

### 7.4 Draw, split, checks, profiles (same in every task)

- **Draw**: `SAMPLE_SIZE` patients, one row per patient, seeded with `SEED`.
  `POSITIVE_SHARE` of them have label 1. With `"prevalence"` that share is
  counted from the labelled cohort (positive patients / all patients), so the
  sample keeps the cohort's own rate.
- **Split**: exactly `TEST_SIZE` patients go to test, the rest to train, seeded
  with `SEED`. Each class is split separately and gives the test set its
  proportional share, so both halves keep the class mix.
- **Checks that stop the run**: fewer patients in a class than the draw needs; a
  feature column missing from the cohort or on `NEVER`; a patient on both sides
  of the split; train and test columns differ; a test size that
  leaves a class with no test or no train rows; DNR / code-status wording left
  in the features or the profiles.
- **Profiles**: written from the saved feature files with `PROFILE`, so nothing
  kept out of the features can reach a profile. `train.jsonl` and `test.jsonl`
  hold the same profiles, each with its patient's label.

## 8. Mortality and discharge (`src_2/`)

These two tasks were built inside the `clip_h/mortality` and `clip_h/discharge`
runner notebooks. Their cleaning functions moved here unchanged, so the files are
the same patients and profiles as the reported runs: on a test extract, the run
scripts reproduce the notebooks' cohort, train/test rows, profiles and exported
files byte for byte, and the runner notebooks keep their run-folder names.

The cleaning differs from sections 1-7 and is kept as it is:

| | `readmission`, `hosp_los3`, `icu_los4` | `mortality`, `discharge` |
|---|---|---|
| One row per | patient | admission |
| Order | draw the sample, then split it | split patients 85 / 15, then draw a class-balanced sample from each half |
| Labs | every value in the window, with hour and abnormal flag | the first value in the first 24 hours (else the earliest) |
| Profile | blocks under `[SECTION]` headings | one paragraph from a fixed template |
| Functions | `src/` | `src_2/` |

### 8.1 Process

```
mimic-iv-3.1/                hosp/ and icu/ tables (list below)
      │
      │  1  load_mimic_tables     read the tables; cached as parquet in mimic-iv-3.1/_clip_h_cache/
      │  2  build_cohort_<task>   one row per adult admission, the features and the label y;
      │                           leakage guard and validate_cohort stop on an excluded column
      ▼
cohort.parquet
      │
      │  3  patient_level_split   15% of patients to test, no patient on both sides
      │     subsample_cohort      seeded, class-balanced draw: 5,000 train and 1,000 test
      │  4  make_profile_<task>   one paragraph per admission
      ▼
train_cohort.parquet  test_cohort.parquet  train/test_features.csv  train/test_labels.csv
train/test_profile.jsonl  train.jsonl  test.jsonl  manifest.json
```

### 8.2 Raw files needed

MIMIC-IV **version 3.1**, in `mimic-iv-3.1/` (the same folder as the other three
tasks): `hosp/` `patients`, `admissions`, `omr`, `services`, `diagnoses_icd`,
`d_icd_diagnoses`, `labevents`, `prescriptions`, `microbiologyevents`, and
`icu/icustays` (all `.csv.gz`). The MIMIC-IV-ED module is not used.

The first run reads `labevents` in filtered chunks and caches the eight lab items
in `mimic-iv-3.1/_clip_h_cache/`. `python run/preprocess_labevents.py` builds the
same cache faster beforehand (optional).

### 8.3 Output files

| File in `../data/<task>/` | Holds | Read by |
|---|---|---|
| `cohort.parquet` | every admission of the cohort, every column | nobody; for inspection |
| `train_cohort.parquet`, `test_cohort.parquet` | the drawn rows: `subject_id`, `hadm_id`, label `y` and features, types and index kept | the `clip_h/<task>` runners |
| `train_features.csv`, `test_features.csv` | the feature columns only, no IDs | nobody; the same file as the other tasks |
| `train_labels.csv`, `test_labels.csv` | one column named after the task | the runners, the comparators |
| `train_profile.jsonl`, `test_profile.jsonl` | `{"idx", "profile"}` per row | the runners, the comparators |
| `train.jsonl`, `test.jsonl` | `{"id", "label", "profile", "subject_id"}` per row; `id` is the `hadm_id` | the comparators |
| `manifest.json` | the settings, the feature and excluded column lists, their signature, counts | the runners (checked against Step 1) |

Unlike the other tasks, these files hold MIMIC-IV IDs (`subject_id`, `hadm_id`):
the runners need them for the patient-level bootstrap and the prediction files.
`data/` is not in git.

### 8.4 Settings

At the top of `run_mortality.py` and `run_discharge.py`: `SEED` (42),
`N_PATIENTS_TRAIN` / `N_PATIENTS_TEST` (5,000 / 1,000), `SUBSAMPLE`
(`'stratified'`), `TRAIN_BALANCE` / `TEST_BALANCE` (0.5 / 0.5), `TEST_FRACTION`
(0.15), `RICH_PROFILE`, `CLEAN_COHORT` (discharge), `OVERWRITE`, and `MIMIC_VERSION`
(`"3.1"`). The runner
notebooks hold the same settings in their Step 1 (they are part of the run-folder
name) and stop when `manifest.json` says the data was written with other values.

The cohort rules are in 8.5 and in the docstrings of `src_2/`.

### 8.5 Cohort rules

| Rule | `mortality` | `discharge` |
|---|---|---|
| Predicted at | admission | admission |
| One row per | adult admission (age 18 or older; ages capped at 91, as MIMIC-IV de-identifies them) | eligible adult admission |
| Label = 1 when | the patient dies during the admission: a recorded `deathtime`, or `discharge_location` DIED | the patient is discharged **Home**: a private residence without services |
| Label = 0 when | the patient survives to discharge | any other eligible destination: home health care, skilled nursing, rehabilitation, long-term acute care, psychiatric care, assisted living, other supported or facility-based care |
| Left out | nobody: survival to discharge is a clean negative | admissions ending in death, hospice, discharge against medical advice, transfer to another acute-care hospital or a missing disposition; with `CLEAN_COHORT = True` also admissions from a skilled nursing facility (their baseline residence is not home) |
| Never a feature (`<TASK>_FORBIDDEN_COLUMNS`) | death time, date of death, discharge location, length of stay, and whole-stay ICU, ventilation, dialysis, medication, culture and service quantities | discharge location, date of death, the diagnosis, DRG and procedure codes assigned during the stay, length of stay, and the same whole-stay quantities |

Same in both tasks:

- **Features** are admission-time only (`<TASK>_FEATURE_COLUMNS`). Laboratory values
  are the first measurement in the first 24 hours of the admission, or the earliest
  available measurement when none falls in that window. Comorbidity (a Charlson
  index) uses the diagnosis codes of prior admissions only. The service is the
  admitting service.
- **Cleaning**: age capping, physiological range checks (out-of-range values become
  missing), unit handling, height and weight conversion, blood-pressure parsing and
  race normalization.
- **Checks that stop the run**: an excluded column in the cohort, a label with one
  class, an all-empty column (`validate_cohort`), a patient on both sides of the split.
- **Split and draw**: 15% of patients go to test (`patient_level_split`), then a seeded,
  class-balanced sample is drawn from each half (`subsample_cohort`): 2,500 and 500
  per class at the paper settings.
- **Profiles**: one paragraph per admission from a fixed template (`profiles.py`); a
  missing field reads `not recorded`. `RICH_PROFILE = True` adds the medication,
  culture and service sections. No LLM is involved.
