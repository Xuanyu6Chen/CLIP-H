# Mortality and discharge: shared cleaning functions: numeric and race cleaning, the feature and forbidden column lists, reading the raw MIMIC-IV tables, the admission-time feature builders, and the cohort check.
"""Functions used by both the mortality and the discharge task.

Moved unchanged from ``clip_h/mortality/mimic_eval/clip_h_lib.py`` (the same file as in
``clip_h/discharge``). Nothing here runs on its own; ``../run/run_mortality.py`` and
``../run/run_discharge.py`` call these functions in order. The raw tables are read from
``../mimic-iv-<version>`` (``MIMIC_VERSION`` in the run scripts).
"""
from __future__ import annotations

import time as _time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================================ #
# Physiological ranges & numeric cleaning
# ============================================================================ #
# Out-of-range values are treated as data-entry errors and set to missing (NaN).
PHYSIOLOGICAL_RANGES: Dict[str, Tuple[float, float]] = {
    "age": (18.0, 91.0),          # MIMIC caps anchor_age at 91 (=91+)
    "heart_rate": (20.0, 250.0),
    "pulse": (20.0, 250.0),
    "temperature_c": (30.0, 45.0),
    "resp_rate": (4.0, 80.0),
    "o2sat": (50.0, 100.0),
    "sbp": (40.0, 300.0),
    "dbp": (20.0, 200.0),
    "bmi": (10.0, 80.0),
    "weight_kg": (20.0, 400.0),
    "height_cm": (100.0, 230.0),
    # Common labs (loose plausibility bounds in standard US units)
    "albumin": (0.5, 7.0),         # g/dL
    "creatinine": (0.1, 25.0),     # mg/dL
    "hemoglobin": (2.0, 25.0),     # g/dL
    "sodium": (90.0, 200.0),       # mEq/L
    "bun": (1.0, 250.0),           # mg/dL
    "glucose": (10.0, 2000.0),     # mg/dL
    "wbc": (0.0, 200.0),           # K/uL
    "platelets": (0.0, 2000.0),    # K/uL
}

_MISSING_TOKENS = {"", "unknown", "nan", "none", "null", "?", "n/a", "na"}

def is_missing(value) -> bool:
    """True if a scalar should be treated as missing."""
    if value is None:
        return True
    try:
        if isinstance(value, float) and np.isnan(value):
            return True
    except TypeError:
        pass
    return str(value).strip().lower() in _MISSING_TOKENS

# Map the long, inconsistent MIMIC race/ethnicity strings into stable buckets.
def normalize_race(value) -> str:
    """Collapse MIMIC race strings into consistent buckets."""
    if is_missing(value):
        return "unknown"
    s = str(value).strip().lower()
    if "white" in s or "portuguese" in s:
        return "white"
    if "black" in s or "african" in s:
        return "black"
    if "hispanic" in s or "latino" in s or "south american" in s:
        return "hispanic"
    if "asian" in s:
        return "asian"
    if "native" in s or "alaska" in s or "american indian" in s:
        return "native_american"
    if "pacific" in s or "hawaiian" in s:
        return "pacific_islander"
    if any(k in s for k in ("unknown", "unable", "declined", "not specified", "patient declined")):
        return "unknown"
    if "other" in s or "multiple" in s:
        return "other"
    return "other"

# ============================================================================ #
# Comorbidity: Charlson index from ICD-10 diagnosis codes
# ============================================================================ #

# Charlson conditions mapped to weights by ICD-10 prefix (a compact subset of the standard
# Quan mapping).
_CHARLSON_ICD10: List[Tuple[Tuple[str, ...], int]] = [
    (("I21", "I22", "I252"), 1),                                    # myocardial infarction
    (("I50", "I110", "I130", "I132"), 1),                           # CHF
    (("I70", "I71", "I731", "I738", "I739", "I771", "K551"), 1),    # peripheral vascular
    (tuple(f"I6{n}" for n in range(0, 10)) + ("G45", "G46"), 1),    # cerebrovascular
    (("F00", "F01", "F02", "F03", "G30"), 1),                       # dementia
    (("J40", "J41", "J42", "J43", "J44", "J45", "J46", "J47"), 1),  # chronic pulmonary
    (("M05", "M06", "M32", "M33", "M34", "M353"), 1),               # rheumatic
    (("K25", "K26", "K27", "K28"), 1),                              # peptic ulcer
    (("B18", "K70", "K713", "K717", "K73", "K74", "K760"), 1),      # mild liver
    # Diabetes (E10-E14) is scored separately in _comorbidity_features because the
    # weight depends on the 4th character (chronic complication = 2, otherwise = 1), which a
    # flat prefix list cannot express (it would miss the common uncomplicated code E11.9).
    (("G81", "G82", "G041", "G114", "G801", "G802"), 2),           # hemiplegia
    (("N18", "N19", "I12", "I13", "N052", "N250"), 2),             # renal
    (("C0", "C1", "C2", "C3", "C40", "C41", "C43", "C45", "C46", "C47", "C48",
      "C49", "C5", "C6", "C70", "C71", "C72", "C73", "C74", "C75", "C76",
      "C81", "C82", "C83", "C84", "C85", "C88", "C90", "C91", "C92"), 2),  # malignancy
    (("K704", "K711", "K721", "K729", "K765", "K766", "K767", "I85"), 3),  # mod/severe liver
    (("C77", "C78", "C79", "C80"), 6),                              # metastatic solid tumour
    (("B20", "B21", "B22", "B24"), 6),                              # HIV/AIDS
]

def _norm_code(code) -> str:
    if is_missing(code):
        return ""
    return str(code).strip().upper().replace(".", "")

def clean_numeric_v(series: "pd.Series", field: str = None,
                    low=None, high=None) -> "pd.Series":
    """Coerce a Series to numeric and set implausible values to NaN (vectorised).

    The plausible range is ``PHYSIOLOGICAL_RANGES[field]`` when ``field`` is a known key,
    otherwise ``[low, high]`` (either bound may be None).
    """
    vals = pd.to_numeric(series, errors="coerce")
    if field is not None and field in PHYSIOLOGICAL_RANGES:
        lo, hi = PHYSIOLOGICAL_RANGES[field]
    else:
        lo, hi = low, high
    if lo is not None:
        vals = vals.where(vals >= lo)
    if hi is not None:
        vals = vals.where(vals <= hi)
    return vals

# ============================================================================ #
# Feature column manifests (leakage-safety encoded structurally)
# ============================================================================ #
# Whole-stay and post-hoc quantities: each aggregates over the entire stay, so it is known only
# during or after the admission and must not be a feature for the admission-time tasks
# (discharge, mortality). length_of_stay_days and icu_los_days are known only at discharge;
# icu_flag, ventilation_flag and dialysis_flag are whole-stay care events; n_medications,
# polypharmacy_flag, n_cultures, positive_culture_flag and n_services accumulate over the stay.
WHOLE_STAY_COLUMNS: List[str] = [
    "length_of_stay_days", "icu_los_days", "icu_flag", "ventilation_flag", "dialysis_flag",
    "n_medications", "polypharmacy_flag", "n_cultures", "positive_culture_flag", "n_services",
]

# Discharge: admission-time signals only. No current-admission ICD/DRG/procedures and no
# whole-stay quantities (WHOLE_STAY_COLUMNS): the prediction is made from what is known early in
# the admission (demographics, admission context, first labs/vitals, admitting service, prior
# history), so every whole-stay quantity is excluded here and listed as forbidden below.
DISCHARGE_FEATURE_COLUMNS: List[str] = [
    "subject_id", "hadm_id", "age", "gender", "race", "marital_status",
    "insurance", "language",
    "admission_type", "admission_location", "arrival_transport", "triage_acuity",
    "triage_temperature", "triage_heartrate", "triage_resprate", "triage_o2sat",
    "triage_sbp", "triage_dbp", "triage_pain", "chief_complaint",
    "lab_albumin_first", "lab_creatinine_first", "lab_hemoglobin_first",
    "lab_sodium_first", "lab_bun_first", "lab_glucose_first", "lab_wbc_first",
    "lab_platelets_first",
    "bmi", "systolic_bp", "diastolic_bp", "height_cm", "weight_kg",
    "curr_service",   # the admitting service (first by transfertime), an admission-time decision
    "n_admissions_prior_365d", "n_ed_visits_prior_180d", "charlson_prior",
]

# Columns that would leak the discharge outcome or are not known at admission; validate_cohort
# asserts they are absent. Includes every whole-stay quantity (WHOLE_STAY_COLUMNS).
DISCHARGE_FORBIDDEN_COLUMNS: List[str] = [
    "diagnosis", "icd_code", "drg_code", "drg_severity", "drg_mortality",
    "procedures_icd", "discharge_location", "dod",
] + WHOLE_STAY_COLUMNS

# ---------------------------------------------------------------------------- #
# In-hospital mortality task.  Label: y = 1 iff the patient died during the index
# admission (deathtime present, or discharge_location == DIED). Predicted from
# admission-time signals only. Every death-revealing or discharge-side quantity is
# forbidden (dod, deathtime, discharge_location, hospital_expire_flag), as is every
# whole-stay quantity; length_of_stay_days in particular is known only post hoc and
# co-varies sharply with in-hospital death (a leakage risk).
# ---------------------------------------------------------------------------- #
MORTALITY_FEATURE_COLUMNS: List[str] = [c for c in DISCHARGE_FEATURE_COLUMNS
                                        if c not in ("length_of_stay_days", "icu_los_days")]

MORTALITY_FORBIDDEN_COLUMNS: List[str] = list(dict.fromkeys([
    "dod", "deathtime", "discharge_location", "length_of_stay_days", "hospital_expire_flag",
] + WHOLE_STAY_COLUMNS))

# MIMIC-IV hosp `labevents` itemids for the first-24h labs in the cohort features.
LAB_ITEMIDS: Dict[str, int] = {
    "albumin": 50862, "creatinine": 50912, "hemoglobin": 51222, "sodium": 50983,
    "bun": 51006, "glucose": 50931, "wbc": 51301, "platelets": 51265,
}

# ============================================================================ #
# Loading (chunked, parquet-cached)
# ============================================================================ #
def _read_filtered_csv_gz(path: str, usecols: List[str],
                          itemid_filter: Optional[set] = None,
                          chunksize: int = 2_000_000) -> pd.DataFrame:
    """Read a (possibly multi-GB) gzip CSV in chunks, keeping only needed cols/rows."""
    frames = []
    for chunk in pd.read_csv(path, compression="gzip", usecols=usecols, chunksize=chunksize):
        if itemid_filter is not None and "itemid" in chunk.columns:
            chunk = chunk[chunk["itemid"].isin(itemid_filter)]
        frames.append(chunk)
    if not frames:
        return pd.DataFrame(columns=usecols)
    return pd.concat(frames, ignore_index=True)

def _cached_table(name: str, csv_path: str, usecols: List[str], cache: Path) -> pd.DataFrame:
    """Load a MIMIC gzip CSV and cache to parquet; read from cache on all subsequent calls."""
    p = cache / f"{name}.parquet"
    t0 = _time.time()
    if p.exists():
        df = pd.read_parquet(p)
        print(f"    [cache]  {name:<22s}  {len(df):>9,} rows   {_time.time()-t0:.1f}s", flush=True)
        return df
    print(f"    [csv]    {name:<22s}  reading {Path(csv_path).name} ...", end="", flush=True)
    df = pd.read_csv(csv_path, compression="gzip", usecols=usecols)
    df.to_parquet(p, index=False)
    print(f"  {len(df):>9,} rows   {_time.time()-t0:.1f}s  [cached]", flush=True)
    return df

def _cached_table_optional(name: str, csv_path: str, usecols: List[str],
                            cache: Path) -> pd.DataFrame:
    """Like _cached_table but returns an empty DataFrame when the file is absent."""
    p = cache / f"{name}.parquet"
    t0 = _time.time()
    if p.exists():
        df = pd.read_parquet(p)
        print(f"    [cache]  {name:<22s}  {len(df):>9,} rows   {_time.time()-t0:.1f}s", flush=True)
        return df
    try:
        print(f"    [csv]    {name:<22s}  reading {Path(csv_path).name} ...", end="", flush=True)
        df = pd.read_csv(csv_path, compression="gzip", usecols=usecols)
        df.to_parquet(p, index=False)
        print(f"  {len(df):>9,} rows   {_time.time()-t0:.1f}s  [cached]", flush=True)
        return df
    except FileNotFoundError:
        print("  not found, skipped", flush=True)
        return pd.DataFrame()

def load_mimic_tables(
    mimic_path: str,
    task: str,
    cache_dir: Optional[str] = None,
    lab_itemids: Optional[Dict[str, int]] = None,
    include_ed: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Load the MIMIC-IV tables needed for a task as a dict of DataFrames.

    Every table is cached to parquet on first read, so reruns are fast. ``labevents`` and
    ``prescriptions`` are read in chunks; ``labevents`` is filtered to ``lab_itemids`` and its
    cache file is keyed by that itemid set. ``mimic_path`` should point at the ``hosp/``
    folder; ``icu/`` and ``ed/`` are resolved as siblings. The cache defaults to
    ``_clip_h_cache`` next to ``hosp/``.

    The MIMIC-IV-ED tables (``ed_stays``, ``triage``) are read only when ``include_ed`` is
    True (default False). ``prescriptions`` and ``microbio`` are read for the ``discharge``
    and ``mortality`` tasks.

    Returns a dict with keys among: patients, admissions, omr, services, diagnoses, d_icd,
    icustays, ed_stays, triage, labevents, prescriptions, microbio.
    """
    lab_itemids = lab_itemids or LAB_ITEMIDS
    hosp = Path(mimic_path)
    root = hosp.parent
    ed = root / "ed"
    icu = root / "icu"
    cache = Path(cache_dir) if cache_dir else (root / "_clip_h_cache")
    cache.mkdir(parents=True, exist_ok=True)

    def g(p):
        return str(p)

    _tload = _time.time()
    print(f"  Loading tables  (cache: {cache})", flush=True)
    tables: Dict[str, pd.DataFrame] = {}

    # Small tables: cached to parquet on first load and read from the cache thereafter.
    tables["patients"]  = _cached_table("patients",  g(hosp / "patients.csv.gz"),  cache=cache,
        usecols=["subject_id", "gender", "anchor_age", "anchor_year", "dod"])
    tables["admissions"] = _cached_table("admissions", g(hosp / "admissions.csv.gz"), cache=cache,
        usecols=["subject_id", "hadm_id", "admittime", "dischtime", "deathtime",
                 "admission_type", "admission_location", "discharge_location",
                 "insurance", "language", "marital_status", "race",
                 "edregtime", "edouttime"])
    tables["omr"]      = _cached_table("omr",      g(hosp / "omr.csv.gz"),      cache=cache,
        usecols=["subject_id", "chartdate", "result_name", "result_value"])
    tables["services"] = _cached_table("services", g(hosp / "services.csv.gz"), cache=cache,
        usecols=["subject_id", "hadm_id", "transfertime", "curr_service"])
    tables["diagnoses"] = _cached_table("diagnoses", g(hosp / "diagnoses_icd.csv.gz"), cache=cache,
        usecols=["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"])
    tables["d_icd"]    = _cached_table("d_icd",    g(hosp / "d_icd_diagnoses.csv.gz"), cache=cache,
        usecols=["icd_code", "icd_version", "long_title"])

    # ICU (optional in some extracts).
    tables["icustays"] = _cached_table_optional("icustays", g(icu / "icustays.csv.gz"), cache=cache,
        usecols=["subject_id", "hadm_id", "stay_id", "first_careunit", "los"])

    # MIMIC-IV-ED tables, read only when include_ed is True.
    if include_ed:
        tables["ed_stays"] = _cached_table_optional("ed_stays", g(ed / "edstays.csv.gz"), cache=cache,
            usecols=["subject_id", "hadm_id", "stay_id", "intime", "arrival_transport"])
        tables["triage"]   = _cached_table_optional("triage",   g(ed / "triage.csv.gz"),   cache=cache,
            usecols=["subject_id", "stay_id", "temperature", "heartrate", "resprate",
                     "o2sat", "sbp", "dbp", "pain", "acuity", "chiefcomplaint"])

    # labevents: filtered to the requested itemids and parquet-cached. preprocess_labevents.py
    # can build this cache ahead of time.
    lab_cache = cache / f"labevents_{'_'.join(str(v) for v in sorted(lab_itemids.values()))}.parquet"
    _t = _time.time()
    if lab_cache.exists():
        print(f"    [cache]  {'labevents':<22s}", end="", flush=True)
        tables["labevents"] = pd.read_parquet(lab_cache)
        print(f"  {len(tables['labevents']):>9,} rows   {_time.time()-_t:.1f}s", flush=True)
    else:
        print(f"    [csv]    {'labevents':<22s}  reading labevents.csv.gz (large; run preprocess_labevents.py for speed) ...", flush=True)
        labs = _read_filtered_csv_gz(
            g(hosp / "labevents.csv.gz"),
            usecols=["subject_id", "hadm_id", "itemid", "charttime", "valuenum", "valueuom"],
            itemid_filter=set(lab_itemids.values()),
        )
        labs.to_parquet(lab_cache, index=False)
        tables["labevents"] = labs
        print(f"    [csv]    {'labevents':<22s}  {len(labs):>9,} rows   {_time.time()-_t:.1f}s  [cached]", flush=True)

    # Prescriptions (medication burden) and microbiology (infection workup) for the discharge
    # and mortality cohorts. The counts the cohort builders derive from them accumulate over
    # the stay (WHOLE_STAY_COLUMNS), so the admission-time feature selection drops them.
    if task in ("discharge", "mortality"):
        rx_cache = cache / "prescriptions_drug.parquet"
        _t = _time.time()
        if rx_cache.exists():
            print(f"    [cache]  {'prescriptions':<22s}", end="", flush=True)
            tables["prescriptions"] = pd.read_parquet(rx_cache)
            print(f"  {len(tables['prescriptions']):>9,} rows   {_time.time()-_t:.1f}s", flush=True)
        else:
            print(f"    [csv]    {'prescriptions':<22s}  reading prescriptions.csv.gz ...", flush=True)
            rx = _read_filtered_csv_gz(
                g(hosp / "prescriptions.csv.gz"),
                usecols=["subject_id", "hadm_id", "drug"],
            )
            rx.to_parquet(rx_cache, index=False)
            tables["prescriptions"] = rx
            print(f"    [csv]    {'prescriptions':<22s}  {len(rx):>9,} rows   {_time.time()-_t:.1f}s  [cached]", flush=True)
        tables["microbio"] = _cached_table_optional("microbio",
            g(hosp / "microbiologyevents.csv.gz"), cache=cache,
            usecols=["subject_id", "hadm_id", "org_name"])
    print(f"  All tables loaded in {_time.time()-_tload:.1f}s  |  keys: {list(tables.keys())}", flush=True)
    return tables

# ============================================================================ #
# Cohort assembly (leakage-safe). The helpers operate on a dict of DataFrames so
# the join logic runs unchanged on small synthetic tables.
# ============================================================================ #
def _prior_counts(admissions: pd.DataFrame, windows=(90, 365)) -> pd.DataFrame:
    """For each admission, count prior admissions within each day-window.

    Uses searchsorted over each patient's sorted admission times, O(A log A) rather than O(A²).
    """
    df = admissions[["subject_id", "hadm_id", "admittime"]].copy()
    df["admittime"] = pd.to_datetime(df["admittime"], errors="coerce")
    df = df.dropna(subset=["admittime"])
    df = df.sort_values(["subject_id", "admittime"]).reset_index(drop=True)

    counts = {f"n_admissions_prior_{w}d": np.zeros(len(df), dtype=int) for w in windows}

    for _subj, grp in df.groupby("subject_id", sort=False):
        if len(grp) < 2:
            continue
        idx = grp.index.to_numpy()
        t_ns = grp["admittime"].values.astype("datetime64[ns]").astype(np.int64)
        for w in windows:
            w_ns = int(w) * 86_400 * int(1e9)
            col = f"n_admissions_prior_{w}d"
            for i in range(1, len(t_ns)):
                lo = int(np.searchsorted(t_ns[:i], t_ns[i] - w_ns, side="left"))
                counts[col][idx[i]] = i - lo

    res = df[["subject_id", "hadm_id"]].copy()
    for w in windows:
        res[f"n_admissions_prior_{w}d"] = counts[f"n_admissions_prior_{w}d"]
    return res

def _ed_prior_counts(admissions: pd.DataFrame, ed_stays: pd.DataFrame, window=180) -> pd.DataFrame:
    """Count prior ED visits within `window` days before each admission.

    Vectorised via a join on subject_id and a boolean window mask.
    """
    res = admissions[["subject_id", "hadm_id"]].copy()
    col = f"n_ed_visits_prior_{window}d"
    res[col] = 0
    if ed_stays is None or ed_stays.empty or "intime" not in ed_stays.columns:
        return res
    adm = admissions[["subject_id", "hadm_id", "admittime"]].copy()
    adm["admittime"] = pd.to_datetime(adm["admittime"], errors="coerce")
    ed = ed_stays[["subject_id", "intime"]].copy()
    ed["intime"] = pd.to_datetime(ed["intime"], errors="coerce")
    ed = ed.dropna(subset=["intime"])
    # Join on subject_id then filter by window
    joined = adm.merge(ed, on="subject_id")
    delta = pd.Timedelta(days=window)
    mask = (joined["intime"] < joined["admittime"]) & \
           (joined["intime"] >= joined["admittime"] - delta)
    cnt = joined[mask].groupby("hadm_id").size().reset_index(name=col)
    res = res.merge(cnt, on="hadm_id", how="left")
    res[col] = res[col].fillna(0).astype(int)
    return res

def _labs_value(labevents: pd.DataFrame, admissions: pd.DataFrame, which: str,
                lab_itemids: Dict[str, int], first_hours: int = 24) -> pd.DataFrame:
    """Per-admission first-``first_hours`` ('first') or last ('last') value for each lab.

    Returns one row per hadm_id with columns lab_<name>_<which>. Values outside
    ``PHYSIOLOGICAL_RANGES`` are set to NaN. Fully vectorised (no Python loop over admissions).
    """
    suffix = "first" if which == "first" else "last"
    # Lab columns are not pre-filled here: missing ones are added after the merge to avoid
    # suffix collisions.
    res = admissions[["hadm_id"]].drop_duplicates().copy()
    if labevents is None or labevents.empty:
        for name in lab_itemids:
            res[f"lab_{name}_{suffix}"] = np.nan
        return res

    lab = labevents[["hadm_id", "itemid", "charttime", "valuenum"]].copy()
    lab["charttime"] = pd.to_datetime(lab["charttime"], errors="coerce")
    lab = lab.dropna(subset=["charttime", "valuenum"])

    adm_t = (admissions[["hadm_id", "admittime", "dischtime"]]
             .drop_duplicates("hadm_id").copy())
    adm_t["admittime"] = pd.to_datetime(adm_t["admittime"], errors="coerce")
    adm_t["dischtime"] = pd.to_datetime(adm_t["dischtime"], errors="coerce")

    lab = lab.merge(adm_t, on="hadm_id", how="inner")

    if which == "first":
        lab["cutoff"] = lab["admittime"] + pd.Timedelta(hours=first_hours)
        in_window = lab[lab["charttime"] <= lab["cutoff"]]
        agg_win = (in_window.sort_values("charttime")
                   .groupby(["hadm_id", "itemid"])["valuenum"].first().reset_index())
        # When no value falls in the first `first_hours`, use the earliest value of the admission.
        agg_all = (lab.sort_values("charttime")
                   .groupby(["hadm_id", "itemid"])["valuenum"].first().reset_index())
        agg = agg_all.merge(agg_win, on=["hadm_id", "itemid"], how="left",
                            suffixes=("_all", "_win"))
        agg["valuenum"] = agg["valuenum_win"].fillna(agg["valuenum_all"])
        agg = agg[["hadm_id", "itemid", "valuenum"]]
    else:
        bounded = lab[lab["charttime"] <= lab["dischtime"].fillna(
            pd.Timestamp("2099-01-01"))]
        agg = (bounded.sort_values("charttime")
               .groupby(["hadm_id", "itemid"])["valuenum"].last().reset_index())

    id_to_name = {v: k for k, v in lab_itemids.items()}
    agg["_name"] = agg["itemid"].map(id_to_name)
    agg = agg.dropna(subset=["_name"])
    for name in lab_itemids:                               # range-clean each lab
        mask = agg["_name"] == name
        if mask.any():
            agg.loc[mask, "valuenum"] = clean_numeric_v(
                agg.loc[mask, "valuenum"], field=name).values
    agg["_col"] = "lab_" + agg["_name"] + "_" + suffix

    pivot = (agg.pivot_table(index="hadm_id", columns="_col",
                             values="valuenum", aggfunc="first")
             .reset_index())
    pivot.columns.name = None

    res = res.merge(pivot, on="hadm_id", how="left")
    for name in lab_itemids:                               # ensure all cols present
        col = f"lab_{name}_{suffix}"
        if col not in res.columns:
            res[col] = np.nan
    return res

def _omr_vitals(omr: pd.DataFrame, subjects: pd.Series) -> pd.DataFrame:
    """Most-recent OMR BMI / BP / height / weight per subject, cleaned.

    Vectorised (no iterrows loop).
    """
    _VITAL_COLS = ["bmi", "systolic_bp", "diastolic_bp", "height_cm", "weight_kg"]
    base = pd.DataFrame({"subject_id": pd.unique(subjects)})
    if omr is None or omr.empty:
        for c in _VITAL_COLS:
            base[c] = np.nan
        return base

    o = omr.copy()
    o["chartdate"] = pd.to_datetime(o["chartdate"], errors="coerce")
    o = o.sort_values("chartdate")
    wide = (o.groupby(["subject_id", "result_name"])["result_value"]
            .last().unstack().reset_index())
    wide.columns.name = None          # drop MultiIndex name artifact from unstack

    if "BMI (kg/m2)" in wide.columns:
        bmi = pd.to_numeric(wide["BMI (kg/m2)"], errors="coerce")
        wide["bmi"] = bmi.where((bmi >= 10) & (bmi <= 80))

    if "Blood Pressure" in wide.columns:
        split = wide["Blood Pressure"].astype(str).str.extract(
            r"(\d+)\s*/\s*(\d+)", expand=True)
        sbp = pd.to_numeric(split[0], errors="coerce")
        dbp = pd.to_numeric(split[1], errors="coerce")
        wide["systolic_bp"]  = sbp.where((sbp >= 40) & (sbp <= 300))
        wide["diastolic_bp"] = dbp.where((dbp >= 20) & (dbp <= 200))

    if "Height (Inches)" in wide.columns:
        h = pd.to_numeric(wide["Height (Inches)"], errors="coerce")
        wide["height_cm"] = (h.where((h >= 39) & (h <= 91)) * 2.54).round(1)

    if "Weight (Lbs)" in wide.columns:
        w = pd.to_numeric(wide["Weight (Lbs)"], errors="coerce")
        wide["weight_kg"] = (w.where((w >= 44) & (w <= 880)) * 0.453592).round(1)

    keep = ["subject_id"] + [c for c in _VITAL_COLS if c in wide.columns]
    result = base.merge(wide[keep], on="subject_id", how="left")
    for c in _VITAL_COLS:             # ensure all expected cols present (NaN if absent)
        if c not in result.columns:
            result[c] = np.nan
    return result

def _icu_features(icustays: pd.DataFrame, admissions: pd.DataFrame) -> pd.DataFrame:
    res = admissions[["hadm_id"]].copy()
    res["icu_flag"] = 0
    res["icu_los_days"] = np.nan
    if icustays is None or icustays.empty:
        return res
    agg = icustays.groupby("hadm_id")["los"].sum().reset_index()
    agg["icu_flag"] = 1
    agg = agg.rename(columns={"los": "icu_los_days"})
    res = res.drop(columns=["icu_flag", "icu_los_days"]).merge(agg, on="hadm_id", how="left")
    res["icu_flag"] = res["icu_flag"].fillna(0).astype(int)
    return res

def _service_feature(services: pd.DataFrame, admissions: pd.DataFrame) -> pd.DataFrame:
    res = admissions[["hadm_id"]].copy()
    res["curr_service"] = np.nan
    if services is None or services.empty:
        return res
    s = services.copy()
    s["transfertime"] = pd.to_datetime(s["transfertime"], errors="coerce")
    # The admitting service is the first service by transfer time (an admission-time decision).
    # The last service would reflect intra-stay transfers not known at admission (whole-stay leakage).
    first = (s.sort_values("transfertime").groupby("hadm_id")["curr_service"].first().reset_index())
    return res.drop(columns=["curr_service"]).merge(first, on="hadm_id", how="left")

def _comorbidity_features(diagnoses: pd.DataFrame, admissions: pd.DataFrame) -> pd.DataFrame:
    """Prior-admission Charlson comorbidity index per admission (``charlson_prior``).

    Codes come from the patient's prior admissions only, never the current one, so the
    score is known at admission time and leakage-safe.
    """
    adm = admissions[["subject_id", "hadm_id", "admittime"]].copy()
    adm["admittime"] = pd.to_datetime(adm["admittime"], errors="coerce")

    # Normalise all codes once, avoiding repeated _norm_code calls per admission.
    dx = diagnoses[["hadm_id", "icd_code", "icd_version"]].copy()
    dx["code_norm"] = dx["icd_code"].fillna("").apply(_norm_code)

    # Single-pass accumulation: adm is sorted by (subject_id, admittime);
    # each row receives Charlson from codes seen in *previous* rows only.
    dx_adm = dx.merge(adm[["hadm_id", "admittime", "subject_id"]],
                      on="hadm_id", how="left")
    adm_sorted = adm.sort_values(["subject_id", "admittime"]).reset_index(drop=True)
    codes_by_hadm: Dict = (
        dx_adm.groupby("hadm_id")["code_norm"].apply(list).to_dict()
    )

    out_hadm: List[int] = []
    out_ch:   List[int] = []
    prev_subj = None
    accumulated: List[str] = []

    for row in adm_sorted.itertuples(index=False):
        if row.subject_id != prev_subj:
            accumulated = []
            prev_subj = row.subject_id
        # Charlson score from the accumulated (already normalised) prior codes.
        codes = accumulated
        if not codes:
            score = 0
        else:
            score = 0
            for prefixes, weight in _CHARLSON_ICD10:
                if any(c.startswith(prefixes) for c in codes):
                    score += weight
            dm_codes = [c for c in codes if c[:3] in {"E10","E11","E12","E13","E14"}]
            if dm_codes:
                comp = any(len(c) >= 4 and c[3] in "23457" for c in dm_codes)
                score += 2 if comp else 1
        out_hadm.append(row.hadm_id)
        out_ch.append(score)
        accumulated.extend(codes_by_hadm.get(row.hadm_id, []))

    return pd.DataFrame({"hadm_id": out_hadm, "charlson_prior": out_ch})

def _assemble_admission_features(tables: Dict[str, pd.DataFrame], rich: bool = True) -> pd.DataFrame:
    """Build the admission-time feature frame used by the mortality cohort.

    Mirrors the feature-engineering steps of ``build_cohort_discharge`` (same leakage-safe
    helpers) but applies no labelling or disposition exclusion. It keeps
    ``length_of_stay_days``, ``deathtime`` and ``discharge_location`` so the caller can
    derive its label from them; the caller then drops them by column selection.
    """
    _t0 = _time.time()
    adm = tables["admissions"].copy()
    pts = tables["patients"]
    adm = adm.merge(pts[["subject_id", "gender", "anchor_age", "anchor_year"]],
                    on="subject_id", how="left")
    adm["admittime"] = pd.to_datetime(adm["admittime"], errors="coerce")
    adm["dischtime"] = pd.to_datetime(adm["dischtime"], errors="coerce")
    adm["age"] = (pd.to_numeric(adm["anchor_age"], errors="coerce")
                  + adm["admittime"].dt.year.subtract(
                      pd.to_numeric(adm["anchor_year"], errors="coerce"))).clip(upper=91).where(
                  lambda s: s >= 18)
    adm["length_of_stay_days"] = (adm["dischtime"] - adm["admittime"]).dt.days
    adm["race"] = adm["race"].map(normalize_race)

    adm = adm.merge(_prior_counts(adm, windows=(365,)), on=["subject_id", "hadm_id"], how="left")
    adm = adm.merge(_ed_prior_counts(adm, tables.get("ed_stays"), window=180),
                    on=["subject_id", "hadm_id"], how="left")
    adm = adm.merge(_comorbidity_features(tables["diagnoses"], adm),
                    on="hadm_id", how="left")
    adm = adm.merge(_labs_value(tables.get("labevents"), adm, "first", LAB_ITEMIDS),
                    on="hadm_id", how="left")
    adm = adm.merge(_omr_vitals(tables.get("omr"), adm["subject_id"]), on="subject_id", how="left")
    adm = adm.merge(_icu_features(tables.get("icustays"), adm), on="hadm_id", how="left")
    adm = adm.merge(_service_feature(tables.get("services"), adm), on="hadm_id", how="left")

    if tables.get("ed_stays") is not None and not tables["ed_stays"].empty \
            and tables.get("triage") is not None and not tables["triage"].empty:
        ed = tables["ed_stays"].merge(tables["triage"], on=["subject_id", "stay_id"], how="left")
        ed = ed.dropna(subset=["hadm_id"]).drop_duplicates("hadm_id")
        ren = {"temperature": "triage_temperature", "heartrate": "triage_heartrate",
               "resprate": "triage_resprate", "o2sat": "triage_o2sat", "sbp": "triage_sbp",
               "dbp": "triage_dbp", "pain": "triage_pain", "acuity": "triage_acuity",
               "chiefcomplaint": "chief_complaint"}
        ed = ed.rename(columns=ren)
        keep = ["hadm_id", "arrival_transport"] + list(ren.values())
        adm = adm.merge(ed[[c for c in keep if c in ed.columns]], on="hadm_id", how="left")

    for c in ["ventilation_flag", "dialysis_flag"]:
        if c not in adm.columns:
            adm[c] = 0

    if rich:
        rx = tables.get("prescriptions")
        if rx is not None and not rx.empty and "drug" in rx.columns:
            nmed = (rx.groupby("hadm_id")["drug"].nunique().reset_index()
                    .rename(columns={"drug": "n_medications"}))
            adm = adm.merge(nmed, on="hadm_id", how="left")
            adm["n_medications"] = adm["n_medications"].fillna(0)
            adm["polypharmacy_flag"] = (adm["n_medications"] >= 10).astype(int)
        micro = tables.get("microbio")
        if micro is not None and not micro.empty:
            ncult = micro.groupby("hadm_id").size().reset_index(name="n_cultures")
            adm = adm.merge(ncult, on="hadm_id", how="left")
            adm["n_cultures"] = adm["n_cultures"].fillna(0)
            if "org_name" in micro.columns:
                pos = (micro.dropna(subset=["org_name"]).groupby("hadm_id").size()
                       .reset_index(name="_p"))
                adm = adm.merge(pos, on="hadm_id", how="left")
                adm["positive_culture_flag"] = (adm.pop("_p").fillna(0) > 0).astype(int)
            else:
                adm["positive_culture_flag"] = 0
        svc = tables.get("services")
        if svc is not None and not svc.empty and "curr_service" in svc.columns:
            nsvc = (svc.groupby("hadm_id")["curr_service"].nunique().reset_index()
                    .rename(columns={"curr_service": "n_services"}))
            adm = adm.merge(nsvc, on="hadm_id", how="left")
            adm["n_services"] = adm["n_services"].fillna(0)
    print(f"  admission features assembled: {len(adm):,} rows  ({_time.time()-_t0:.1f}s)", flush=True)
    return adm

def validate_cohort(df: pd.DataFrame, task: str, rich: bool = False,
                    min_rows: int = 1) -> bool:
    """Tripwire after the cohort build: turn silent ETL errors into a loud, early failure.

    Checks that the cohort has at least ``min_rows`` rows, a two-class label column ``y``,
    none of the task's forbidden columns (outcome-coupled or whole-stay) and no all-NaN
    columns. Raises ValueError listing every problem; returns True otherwise. ``rich`` does
    not change the checks."""
    errs = []
    if df is None or len(df) < min_rows:
        errs.append(f"cohort has {0 if df is None else len(df)} rows (< {min_rows})")
    if df is not None and "y" not in df.columns:
        errs.append("missing label column 'y'")
    elif df is not None and df["y"].nunique() < 2:
        errs.append(f"label has <2 classes: {df['y'].value_counts().to_dict()}")
    if df is not None:
        # Leakage tripwire: the task's forbidden columns (outcome-coupled and whole-stay) must be
        # absent from the built cohort.
        forbidden = {
            "discharge": DISCHARGE_FORBIDDEN_COLUMNS,
            "mortality": MORTALITY_FORBIDDEN_COLUMNS,
        }.get(task, [])
        leaked = [c for c in forbidden if c in df.columns]
        if leaked:
            errs.append(f"forbidden {task} columns present in cohort (leakage): {sorted(leaked)}")
    if df is not None:
        allnan = [c for c in df.columns if df[c].isna().all()]
        if allnan:
            errs.append(f"all-NaN columns: {allnan}")
    if errs:
        raise ValueError("Cohort validation failed:\n  - " + "\n  - ".join(errs))
    return True
