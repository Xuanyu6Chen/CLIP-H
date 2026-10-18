# General MIMIC-IV cleaning functions used by two or more tasks: read raw tables, keep a time window, squash events into one cell per admission.
"""General cleaning functions.

Every function here takes a *spine* (one row per chosen admission, with the
clock start in `admittime`) and returns columns for those admissions. Choosing
the spine is each task's own job (see mimic_clean_<task>.py).

Nothing in this file runs on its own; the run scripts in ../run call these
functions in order. The raw tables are read from ../mimic-iv-3.1.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

# =============================================================================
# Fixed lists
# =============================================================================

# Where the raw tables are read from: the mimic-iv-3.1 folder beside run/ and
# src/. It must hold MIMIC-IV's own hosp/ and icu/ folders.
MIMIC_DIR = Path(__file__).resolve().parents[1] / "mimic-iv-3.1"

# Table name -> raw file under MIMIC_DIR.
RAW_TABLES: dict[str, str] = {
    "patients": "hosp/patients.csv.gz",
    "admissions": "hosp/admissions.csv.gz",
    "omr": "hosp/omr.csv.gz",
    "diagnoses": "hosp/diagnoses_icd.csv.gz",
    "diagnosis_labels": "hosp/d_icd_diagnoses.csv.gz",
    "labevents": "hosp/labevents.csv.gz",
    "microbiology": "hosp/microbiologyevents.csv.gz",
    "prescriptions": "hosp/prescriptions.csv.gz",
    "services": "hosp/services.csv.gz",
    "transfers": "hosp/transfers.csv.gz",
    "icustays": "icu/icustays.csv.gz",
    "chartevents": "icu/chartevents.csv.gz",
    "outputevents": "icu/outputevents.csv.gz",
    "procedureevents": "icu/procedureevents.csv.gz",
}

# Lab name -> the item IDs that count as it. Only these rows are kept from labevents.
LAB_ITEMIDS: dict[str, list[int]] = {
    "creatinine": [50912],
    "sodium": [50983],
    "potassium": [50971, 50822],
    "lactate": [50813],
    "hemoglobin": [51222, 50811],
    "wbc": [51301],
    "platelets": [51265],
    "bun": [51006],
    "bicarbonate": [50882, 50803],
    "glucose": [50931, 50809],
    "albumin": [50862],
    "bilirubin": [50885],
    "inr": [51237],
    "troponin": [51002, 51003],
}

# Vital sign -> chartevents item IDs. Temperature comes in Fahrenheit and
# Celsius; clean_chartevents converts F to C and merges both into "Temp".
VITAL_ITEMIDS: dict[str, list[int]] = {
    "HR": [220045],
    "SBP": [220050, 220179],
    "SpO2": [220277],
    "RR": [220210],
    "Temp_F": [223761],
    "Temp_C": [223762],
}

# outputevents item IDs summed as urine output (mL), and the low-urine cutoff.
URINE_ITEMIDS: list[int] = [
    226559, 226560, 226561, 226584, 226563, 226564,
    226565, 226567, 226557, 226558, 227489,
]
OLIGURIA_ML = 400.0

# procedureevents item IDs: ventilation (invasive + non-invasive) and dialysis.
VENT_ITEMIDS: list[int] = [225792, 225794]
DIALYSIS_ITEMIDS: list[int] = [225441, 225802, 225803, 225805, 225809, 225955]

# Service code -> readable name.
SERVICE_MAP: dict[str, str] = {
    "MED": "Medicine", "CMED": "Cardiac Medicine", "SURG": "General Surgery",
    "CSURG": "Cardiac Surgery", "OMED": "Orthopedic Medicine", "NMED": "Neurology",
    "ORTHO": "Orthopedics", "NSURG": "Neurosurgery", "VSURG": "Vascular Surgery",
    "TSURG": "Thoracic Surgery", "PSURG": "Plastic Surgery", "OBS": "Obstetrics",
    "GYN": "Gynecology", "GU": "Urology", "PSYCH": "Psychiatry", "TRAUM": "Trauma",
    "ENT": "ENT", "DENT": "Dental", "EYE": "Ophthalmology",
}

# OMR result name -> output column. These are outpatient (baseline) values.
OMR_NAMES: dict[str, str] = {
    "Height (Inches)": "height_in",
    "Weight (Lbs)": "weight_lbs",
    "BMI (kg/m2)": "bmi",
    "Blood Pressure": "blood_pressure",
}

# How save_cohort reads cohort.csv back: the OMR columns as text, exactly as
# recorded ("120/80", "182.10"). Left to guess, pandas turns "66" into "66.0"
# in some rows only, which changes the feature files.
COHORT_DTYPES: dict[str, type] = {column: str for column in OMR_NAMES.values()}

# Count and flag columns where "no event recorded" means 0, not missing.
# was_in_icu_24h MUST be here: left empty for never-ICU patients, the ICU
# profile line would appear for late-ICU patients only and leak the whole-stay
# "touched the ICU" fact.
ZERO_FILL: tuple[str, ...] = (
    "n_cultures", "n_transfers_24h", "was_in_icu", "was_in_icu_24h",
    "abnormal_lab_count", "los_in_icu", "urine_output_ml_24h", "oliguria_flag",
    "mechanical_ventilation_flag", "dialysis_flag",
)

# DNR / code-status wording is too close to an outcome proxy, so it may not
# appear in features or profiles. Applied in this order.
DNR_PATTERNS: list[str] = [
    r"\bDNR\b",
    r"\bdo\s+not\s+resuscitate\b",
    r"\bdo[-\s]*not[-\s]*resuscitate\b",
    r"\bcode\s+status\s*[:=-]?\s*DNR\b",
    r"\bDNR\s*/\s*DNI\b",
    r"\bDNI\s*/\s*DNR\b",
    r"\bRNS\s+status\b",
    r"\bresuscitation\s+status\b",
]
ANY_DNR = re.compile("|".join(f"(?:{p})" for p in DNR_PATTERNS), flags=re.IGNORECASE)


# =============================================================================
# Read
# =============================================================================

def read_table(name: str, usecols: list[str] | None = None) -> pd.DataFrame:
    """Read one small raw table whole (only the columns in `usecols`, if given)."""
    path = MIMIC_DIR / RAW_TABLES[name]
    if not path.exists():
        raise FileNotFoundError(f"Missing raw MIMIC-IV file: {path}. Put MIMIC-IV's hosp/ and "
                                f"icu/ folders in {MIMIC_DIR} (the files needed are listed in the README).")
    return pd.read_csv(path, usecols=usecols)


def read_events(name: str, hadm_ids: Iterable[int], usecols: list[str],
                itemids: Iterable[int] | None = None) -> pd.DataFrame:
    """Read one huge raw table in 1M-row chunks, keeping only rows for the given
    admissions (and item IDs, if given). Keeps memory bounded for the 17 GB
    labevents and the chartevents / prescriptions tables.
    """
    path = MIMIC_DIR / RAW_TABLES[name]
    cohort = set(hadm_ids)
    allow = set(itemids) if itemids is not None else None
    kept: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=1_000_000):
        chunk = chunk[chunk["hadm_id"].isin(cohort)]
        if allow is not None:
            chunk = chunk[chunk["itemid"].isin(allow)]
        if not chunk.empty:
            kept.append(chunk)
    return pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(columns=usecols)


def add_patient_info(df: pd.DataFrame, patients: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Attach gender (spelled out) and age at `time_col` to each row.

    Rows whose patient is not in `patients` are dropped. `time_col` must already
    be a datetime column.
    """
    columns = [c for c in ["subject_id", "gender", "anchor_age", "anchor_year", "dod"]
               if c in patients.columns]
    out = df.merge(patients[columns], on="subject_id", how="inner")
    out["age"] = out["anchor_age"] + (out[time_col].dt.year - out["anchor_year"])
    out["gender"] = out["gender"].map({"F": "Female", "M": "Male"}).fillna(out["gender"])
    return out


# =============================================================================
# Window and squash: many event rows -> one cell per admission
# =============================================================================

def in_window(events: pd.DataFrame, spine: pd.DataFrame, time_col: str,
              window_hours: float | None) -> pd.DataFrame:
    """Keep only the events inside the observation window.

    Adds `hr` (hours since the spine's admittime) and keeps 0 <= hr <=
    window_hours. `window_hours=None` means the whole stay: the window ends at
    that admission's dischtime, NOT unbounded, because events recorded after
    discharge sit outside even a predict-at-discharge task's view.
    """
    columns = ["hadm_id", "admittime"] + (["dischtime"] if window_hours is None else [])
    anchor = spine[columns].copy()
    anchor["admittime"] = pd.to_datetime(anchor["admittime"], errors="coerce")
    merged = events.merge(anchor, on="hadm_id", how="inner")
    times = pd.to_datetime(merged[time_col], errors="coerce")
    merged["hr"] = (times - merged["admittime"]).dt.total_seconds() / 3600.0
    if window_hours is None:
        dischtime = pd.to_datetime(merged["dischtime"], errors="coerce")
        return merged[(merged["hr"] >= 0) & (times <= dischtime)].drop(columns="dischtime").copy()
    return merged[(merged["hr"] >= 0) & (merged["hr"] <= window_hours)].copy()


def value_sequence(events: pd.DataFrame, item_col: str, value_col: str,
                   flag_col: str, hr_col: str) -> pd.DataFrame:
    """One column per item, each cell "v*@Nh; v@Nh; ..." ordered by hour.
    `*` marks readings flagged "abnormal". Readings without a value are skipped.
    """
    ev = events.sort_values([item_col, hr_col])
    groups = ev.groupby(["hadm_id", item_col]).size().index   # every (admission, item) pair
    if len(groups) == 0:
        return pd.DataFrame(index=pd.Index([], name="hadm_id"))
    ev = ev[ev[value_col].notna()]
    abnormal = (ev[flag_col].astype(str).str.lower() == "abnormal").fillna(False).astype(bool)
    readings = [f"{value:g}{'*' if star else ''}@{hour:.0f}h"
                for value, star, hour in zip(ev[value_col].astype(float).tolist(),
                                             abnormal.tolist(), ev[hr_col].tolist())]
    seq = (ev[["hadm_id", item_col]].assign(seq=readings)
           .groupby(["hadm_id", item_col])["seq"].agg("; ".join)
           .reindex(groups).reset_index(name="seq"))
    return seq.pivot(index="hadm_id", columns=item_col, values="seq")


def minmax_range(events: pd.DataFrame, item_col: str, value_col: str,
                 order: list[str] | None = None) -> pd.Series:
    """One cell per admission of "item min–max" ranges, e.g. "HR 59–97; SBP 90–140".
    `order` sets the item order.
    """
    g = events.groupby(["hadm_id", item_col])[value_col].agg(["min", "max"]).reset_index()

    def one(item: str, mn: float, mx: float) -> str:
        mn, mx = round(mn, 1), round(mx, 1)
        body = f"{mn:g}" if mn == mx else f"{mn:g}–{mx:g}"
        return f"{item} {body}"

    g["rng"] = [one(it, mn, mx) for it, mn, mx in zip(g[item_col], g["min"], g["max"])]
    order_map = {v: i for i, v in enumerate(order)} if order else {}
    g["ord"] = g[item_col].map(lambda x: order_map.get(x, 99))
    g = g.sort_values(["hadm_id", "ord"])
    return g.groupby("hadm_id")["rng"].apply(lambda s: "; ".join(s)).rename("vitals")


def joined_list(events: pd.DataFrame, group_col: str, value_col: str,
                sep: str = "; ") -> pd.Series:
    """One cell per group: the values joined in order, without repeats."""
    def join_unique(s: pd.Series) -> Any:
        seen = list(dict.fromkeys(str(x).strip() for x in s if pd.notna(x) and str(x).strip()))
        return sep.join(seen) if seen else pd.NA

    return events.groupby(group_col)[value_col].apply(join_unique)


def nearest_value(events: pd.DataFrame, spine: pd.DataFrame, subject_col: str,
                  time_col: str, name_col: str, value_col: str,
                  name_map: dict[str, str]) -> pd.DataFrame:
    """For each admission and mapped name, the value dated nearest on-or-before
    admittime (fallback: nearest after). For tables without hadm_id, joined on
    the patient.
    """
    anchor = spine[["subject_id", "hadm_id", "admittime"]].copy()
    anchor["admittime"] = pd.to_datetime(anchor["admittime"], errors="coerce")
    ev = events[events[name_col].isin(name_map)].copy()
    ev[time_col] = pd.to_datetime(ev[time_col], errors="coerce")
    ev["col"] = ev[name_col].map(name_map)
    merged = ev.merge(anchor, left_on=subject_col, right_on="subject_id", how="inner")
    merged["delta"] = (merged["admittime"] - merged[time_col]).dt.total_seconds()
    merged["before"] = merged["delta"] >= 0          # prefer on/before admit
    merged["absdelta"] = merged["delta"].abs()
    merged = merged.sort_values(
        ["hadm_id", "col", "before", "absdelta"],
        ascending=[True, True, False, True],
    )
    picked = merged.groupby(["hadm_id", "col"]).first().reset_index()
    if picked.empty:
        return pd.DataFrame(index=pd.Index([], name="hadm_id"))
    return picked.pivot(index="hadm_id", columns="col", values=value_col)


# =============================================================================
# Table cleaners: one per raw table, each returns a few columns per admission
# (indexed by hadm_id). "Task window" = the window_hours the run script passes.
# =============================================================================

NO_ROWS = pd.DataFrame(index=pd.Index([], name="hadm_id"))


def clean_omr(spine: pd.DataFrame) -> pd.DataFrame:
    """height_in, weight_lbs, bmi, blood_pressure: nearest value before
    admission. OMR is outpatient and has no hadm_id, so there is no window."""
    omr = read_table("omr")
    omr = omr[omr["subject_id"].isin(set(spine["subject_id"]))].copy()
    return nearest_value(omr, spine, "subject_id", "chartdate",
                         "result_name", "result_value", OMR_NAMES)


def clean_diagnoses(spine: pd.DataFrame) -> pd.DataFrame:
    """diagnoses (the admission's ICD titles, in sequence order, joined into one
    cell) and diagnosis_count.

    The list is coded over the WHOLE stay, so only a predict-at-discharge task
    (readmission) may use it as a feature. The first-24h tasks keep the two
    columns in cohort.csv and list them under NEVER."""
    diagnoses = read_table("diagnoses")
    titles = read_table("diagnosis_labels")[["icd_code", "icd_version", "long_title"]]
    titles = titles.drop_duplicates(["icd_code", "icd_version"])
    dx = diagnoses[diagnoses["hadm_id"].isin(set(spine["hadm_id"]))].copy()
    dx = dx.merge(titles, on=["icd_code", "icd_version"], how="left").sort_values(["hadm_id", "seq_num"])
    return pd.DataFrame({
        "diagnoses": joined_list(dx, "hadm_id", "long_title"),
        "diagnosis_count": dx.groupby("hadm_id").size(),
    })


def clean_labevents(spine: pd.DataFrame, window_hours: float | None) -> pd.DataFrame:
    """14 lab columns as value sequences, plus abnormal_lab_count. Task window."""
    ev = read_events("labevents", spine["hadm_id"],
                     usecols=["hadm_id", "itemid", "charttime", "valuenum", "flag"],
                     itemids=[i for ids in LAB_ITEMIDS.values() for i in ids])
    if ev.empty:
        return NO_ROWS
    ev["analyte"] = ev["itemid"].map({i: name for name, ids in LAB_ITEMIDS.items() for i in ids})
    ev = in_window(ev, spine, "charttime", window_hours)
    seq = value_sequence(ev, "analyte", "valuenum", "flag", "hr")
    abnormal = (ev[ev["flag"].astype(str).str.lower() == "abnormal"]
                .groupby("hadm_id").size().rename("abnormal_lab_count"))
    return seq.join(abnormal, how="outer")


def clean_microbiology(spine: pd.DataFrame, window_hours: float | None) -> pd.DataFrame:
    """n_cultures (distinct specimens) and culture_sites. Task window."""
    micro = read_table("microbiology", usecols=["hadm_id", "micro_specimen_id", "spec_type_desc", "charttime"])
    m = micro[micro["hadm_id"].isin(set(spine["hadm_id"]))].copy()
    m = m.drop_duplicates(["hadm_id", "micro_specimen_id", "spec_type_desc", "charttime"])
    m = in_window(m, spine, "charttime", window_hours)
    return pd.DataFrame({
        "n_cultures": m.groupby("hadm_id")["micro_specimen_id"].nunique(),
        "culture_sites": joined_list(m, "hadm_id", "spec_type_desc"),
    })


def clean_prescriptions(spine: pd.DataFrame, window_hours: float | None) -> pd.DataFrame:
    """medications: names of MAIN drugs started. Task window."""
    ev = read_events("prescriptions", spine["hadm_id"],
                     usecols=["hadm_id", "starttime", "drug", "drug_type"])
    if ev.empty:
        return NO_ROWS
    ev = ev[ev["drug_type"] == "MAIN"]
    ev = in_window(ev, spine, "starttime", window_hours)
    return joined_list(ev, "hadm_id", "drug").rename("medications").to_frame()


def clean_chartevents(spine: pd.DataFrame, window_hours: float | None) -> pd.DataFrame:
    """vitals: min–max range per vital sign (HR, SBP, SpO2, RR, Temp). ICU-only
    table, so empty for patients never in the ICU. Task window."""
    ev = read_events("chartevents", spine["hadm_id"],
                     usecols=["hadm_id", "itemid", "charttime", "valuenum"],
                     itemids=[i for ids in VITAL_ITEMIDS.values() for i in ids])
    if ev.empty:
        return NO_ROWS
    ev["vital"] = ev["itemid"].map({i: name for name, ids in VITAL_ITEMIDS.items() for i in ids})
    ev = in_window(ev, spine, "charttime", window_hours)
    is_f = ev["vital"] == "Temp_F"
    ev.loc[is_f, "valuenum"] = (ev.loc[is_f, "valuenum"] - 32.0) * 5.0 / 9.0
    ev["vital"] = ev["vital"].replace({"Temp_F": "Temp", "Temp_C": "Temp"})
    return minmax_range(ev, "vital", "valuenum",
                        order=["HR", "SBP", "SpO2", "RR", "Temp"]).to_frame()


def clean_outputevents(spine: pd.DataFrame, window_hours: float | None) -> pd.DataFrame:
    """urine_output_ml_24h (summed) and oliguria_flag (under OLIGURIA_ML).
    ICU-only table. Task window."""
    ev = read_events("outputevents", spine["hadm_id"],
                     usecols=["hadm_id", "charttime", "itemid", "value"],
                     itemids=URINE_ITEMIDS)
    if ev.empty:
        return NO_ROWS
    ev = in_window(ev, spine, "charttime", window_hours)
    ev["value"] = pd.to_numeric(ev["value"], errors="coerce")
    out = ev.groupby("hadm_id")["value"].sum().rename("urine_output_ml_24h").to_frame()
    out["oliguria_flag"] = (out["urine_output_ml_24h"] < OLIGURIA_ML).astype("int64")
    return out


def clean_procedureevents(spine: pd.DataFrame, window_hours: float | None) -> pd.DataFrame:
    """mechanical_ventilation_flag and dialysis_flag: any started in the window.
    ICU-only table. Task window."""
    ev = read_events("procedureevents", spine["hadm_id"],
                     usecols=["hadm_id", "starttime", "itemid"],
                     itemids=VENT_ITEMIDS + DIALYSIS_ITEMIDS)
    if ev.empty:
        return NO_ROWS
    ev = in_window(ev, spine, "starttime", window_hours)
    vent = ev[ev["itemid"].isin(VENT_ITEMIDS)].groupby("hadm_id").size()
    dial = ev[ev["itemid"].isin(DIALYSIS_ITEMIDS)].groupby("hadm_id").size()
    out = pd.DataFrame(index=pd.Index(sorted(set(vent.index) | set(dial.index)), name="hadm_id"))
    out["mechanical_ventilation_flag"] = out.index.isin(vent.index).astype("int64")
    out["dialysis_flag"] = out.index.isin(dial.index).astype("int64")
    return out


def clean_services(spine: pd.DataFrame) -> pd.DataFrame:
    """service: latest service within the first 24h, as a readable name.

    Always 24h, whatever the task window: the profile calls this the ADMITTING
    service, and under a whole-stay window "latest" would silently become the
    discharge service instead.
    """
    services = read_table("services")
    s = services[services["hadm_id"].isin(set(spine["hadm_id"]))].copy()
    s = in_window(s, spine, "transfertime", 24.0).sort_values(["hadm_id", "hr"])
    latest = s.groupby("hadm_id")["curr_service"].last()
    service = latest.map(lambda c: SERVICE_MAP.get(c, str(c).title()) if pd.notna(c) else c)
    return service.rename("service").to_frame()


def clean_transfers(spine: pd.DataFrame) -> pd.DataFrame:
    """n_transfers_24h: real care-unit changes in the first 24h (UNKNOWN,
    Discharge Lounge and discharge events dropped; repeats collapsed). Always
    24h: the column name and the profile sentence both say so."""
    transfers = read_table("transfers")
    t = transfers[transfers["hadm_id"].isin(set(spine["hadm_id"]))].copy()
    t = t[~t["eventtype"].astype(str).str.lower().eq("discharge")]
    t = t[~t["careunit"].isin(["UNKNOWN", "Discharge Lounge"])]
    t = in_window(t, spine, "intime", 24.0).sort_values(["hadm_id", "hr"])

    def count_moves(units: list[str]) -> int:
        deduped = [c for i, c in enumerate(units) if i == 0 or c != units[i - 1]]
        return max(len(deduped) - 1, 0)

    n = t.groupby("hadm_id")["careunit"].apply(lambda s: count_moves(list(s)))
    return n.rename("n_transfers_24h").to_frame()


def clean_icustays(spine: pd.DataFrame) -> pd.DataFrame:
    """was_in_icu and los_in_icu (whole stay), and was_in_icu_24h.

    The whole-stay pair is valid only for a task that predicts at discharge
    (readmission). The length-of-stay tasks must use was_in_icu_24h: an ICU
    transfer after the window is a consequence of the long stay, so the
    whole-stay flag leaks the label. In the readmission cohort the two differed
    for 19.4% of ICU-flagged admissions, a subgroup with a 93.8% positive rate.
    """
    icustays = read_table("icustays")
    icu = icustays[icustays["hadm_id"].isin(set(spine["hadm_id"]))].copy()
    anchors = spine[["hadm_id", "admittime"]].drop_duplicates("hadm_id")
    icu = icu.merge(anchors, on="hadm_id", how="left")
    intime = pd.to_datetime(icu["intime"], errors="coerce")
    admit = pd.to_datetime(icu["admittime"], errors="coerce")
    hours_from_admit = (intime - admit).dt.total_seconds() / 3600.0
    icu["_in_window"] = ((hours_from_admit >= 0) & (hours_from_admit <= 24.0)).astype(int)
    return icu.groupby("hadm_id").agg(
        was_in_icu=("stay_id", lambda s: 1),
        was_in_icu_24h=("_in_window", "max"),
        los_in_icu=("los", "sum"),
    )


# =============================================================================
# Join, save, label, check
# =============================================================================

def join_on_spine(spine: pd.DataFrame, frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Attach every cleaner's columns to the spine by hadm_id. An admission with
    no events gets an empty cell; counts and flags (ZERO_FILL) get 0."""
    cohort = spine.set_index("hadm_id")
    for frame in frames:
        if frame is None or frame.empty:
            continue
        cohort = cohort.join(frame, how="left")
    cohort = cohort.reset_index()
    for column in ZERO_FILL:
        if column in cohort.columns:
            cohort[column] = cohort[column].fillna(0)
    return cohort


def save_cohort(cohort: pd.DataFrame, path: Path, overwrite: bool) -> pd.DataFrame:
    """Write cohort.csv and return it as read back from that file, so every
    later step works from exactly what is on disk (a CSV does not keep column
    types). Stops if the file is already there and `overwrite` is false."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Set OVERWRITE = True in the run script to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(path, index=False)
    return pd.read_csv(path, dtype=COHORT_DTYPES)


def los_label(cohort: pd.DataFrame, label_col: str, threshold_hours: float,
              min_hours: float) -> pd.DataFrame:
    """Add the length-of-stay label: 1 when the stay lasted longer than
    `threshold_hours` (admittime to dischtime, the spine's clock).

    Hours, not floored days, so 72h means exactly "more than 3 days".

    Stays shorter than `min_hours` are dropped. Such a record is cut short BY
    the outcome: a stay ending at hour 14 cannot have labs at hour 20, so its
    sparseness would encode the label. 30h = 24 observed + a 6h gap
    (MIMIC-Extract, ACM CHIL 2020).
    """
    df = cohort.copy()
    start = pd.to_datetime(df["admittime"], errors="coerce")
    end = pd.to_datetime(df["dischtime"], errors="coerce")
    df["_los_hours"] = (end - start).dt.total_seconds() / 3600.0
    df = df[df["_los_hours"].notna()].copy()
    before = len(df)
    df = df[df["_los_hours"] >= min_hours].copy()
    print(f"[{label_col}] stays of at least {min_hours:g}h: kept {len(df):,}/{before:,}")
    df[label_col] = (df["_los_hours"] > threshold_hours).astype(int)
    print(f"[{label_col}] stay > {threshold_hours:g}h: prevalence {df[label_col].mean():.4f}")
    return df


def check_dnr(texts: pd.DataFrame | Iterable[Any], where: str) -> None:
    """Stop the run if any DNR / code-status wording is left in `texts`
    (a table of features, or a list of profile texts)."""
    values = texts.to_numpy(dtype=object).ravel() if isinstance(texts, pd.DataFrame) else texts
    found = sum(1 for value in values if isinstance(value, str) and ANY_DNR.search(value))
    if found:
        raise ValueError(f"DNR / code-status wording remains in {where}: {found} value(s)")
