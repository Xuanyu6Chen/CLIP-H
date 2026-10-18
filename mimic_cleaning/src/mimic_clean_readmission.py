# Readmission-only cleaning functions: its spine (first live discharge + next admission), DNR removal, and the 30-day label.
"""Functions only the readmission task uses.

The task predicts AT DISCHARGE whether the patient is admitted again within 30
days, so the whole stay is observable and coded diagnoses are legitimate here.
"""
from __future__ import annotations

import re

import pandas as pd

from mimic_general import ANY_DNR, DNR_PATTERNS, add_patient_info

SPINE_COLUMNS = [
    "subject_id", "hadm_id", "admittime", "dischtime", "edregtime", "edouttime",
    "deathtime", "discharge_location", "hospital_expire_flag", "gender", "age",
    "admission_type", "admission_location", "insurance", "language",
    "marital_status", "race",
    "next_admit_time", "readmission_day", "has_readmission",
    "death_within_30d_no_readmission",
]


def left_alive(admissions: pd.DataFrame) -> pd.Series:
    """True for each admission the patient left alive (no expire flag, no death
    time, discharge location not "died")."""
    expired = pd.to_numeric(admissions["hospital_expire_flag"], errors="coerce").fillna(0).astype(int) != 0
    has_deathtime = admissions["deathtime"].notna()
    location = admissions["discharge_location"].astype("string").str.strip().str.lower()
    return ~(expired | has_deathtime | (location == "died").fillna(False))


def readmission_spine(admissions: pd.DataFrame, patients: pd.DataFrame,
                      window_days: int = 30, drop_death_without_readmission: bool = False,
                      max_patients: int | None = None) -> pd.DataFrame:
    """One row per patient: their FIRST admission that ended alive, paired with
    the admission that follows it.

    Adds next_admit_time, readmission_day (whole days from discharge to the next
    admission), has_readmission (next admission within `window_days`) and
    death_within_30d_no_readmission (died within the window before any
    readmission). Patients with no live discharge, or whose next admission
    starts before the discharge, are left out.
    """
    adm = admissions.copy()
    adm["admittime"] = pd.to_datetime(adm["admittime"], errors="coerce")
    adm["dischtime"] = pd.to_datetime(adm["dischtime"], errors="coerce")
    adm = add_patient_info(adm, patients, "admittime")
    dod = pd.to_datetime(adm["dod"], errors="coerce") if "dod" in adm.columns else pd.Series(pd.NaT, index=adm.index)
    adm["dod"] = dod
    if max_patients is not None:
        adm = adm[adm["subject_id"].isin(set(adm["subject_id"].drop_duplicates().iloc[:max_patients]))].copy()

    # Order each patient's admissions by time, keeping patients in file order.
    adm["_patient_order"] = pd.factorize(adm["subject_id"])[0]
    adm = adm.sort_values(["_patient_order", "admittime"], kind="stable")
    adm["next_admit_time"] = adm.groupby("subject_id", sort=False)["admittime"].shift(-1)
    spine = adm[left_alive(adm)].groupby("subject_id", sort=False).head(1).copy()
    without_live_discharge = adm["subject_id"].nunique() - len(spine)

    # Timedelta.days on purpose: whole days, rounded down (a gap of 30 days 23
    # hours counts as 30). Changing it changes the label.
    days = (spine["next_admit_time"] - spine["dischtime"]).dt.days
    negative_gap = days < 0
    spine, days = spine[~negative_gap].copy(), days[~negative_gap]
    readmitted = days.notna() & (days >= 0) & (days <= window_days)

    death_gap = (spine["dod"].dt.normalize() - spine["dischtime"].dt.normalize()).dt.days
    readmitted_before_death = readmitted & (spine["next_admit_time"].dt.normalize() <= spine["dod"].dt.normalize())
    died_first = death_gap.between(0, window_days) & ~readmitted_before_death

    spine["readmission_day"] = days.astype("float64")   # float, so cohort.csv writes whole days as "5.0"
    spine["has_readmission"] = readmitted.astype(int)
    spine["death_within_30d_no_readmission"] = died_first.astype(int)
    if drop_death_without_readmission:
        spine = spine[~died_first]

    spine = spine[SPINE_COLUMNS].reset_index(drop=True)
    spine.attrs["stats"] = {
        "source_patients": int(adm["subject_id"].nunique()),
        "source_admissions": int(len(adm)),
        "patients_without_live_discharge": int(without_live_discharge),
        "patients_dropped_negative_gap": int(negative_gap.sum()),
        "patients_with_readmission": int(spine["has_readmission"].sum()),
    }
    return spine


def remove_dnr(text: pd.Series) -> tuple[pd.Series, int]:
    """Replace DNR / code-status wording with "[removed]". The admission stays;
    only the wording goes. Returns the cleaned column and how many cells changed."""
    patterns = [re.compile(p, flags=re.IGNORECASE) for p in DNR_PATTERNS]

    def clean(value: str) -> str:
        for pattern in patterns:
            value = pattern.sub("[removed]", value)
        value = re.sub(r"\s+", " ", value).strip()
        return re.sub(r"\s+([,.;:])", r"\1", value)

    hit = text.map(lambda value: isinstance(value, str) and ANY_DNR.search(value) is not None).astype(bool)
    cleaned = text.copy()
    cleaned[hit] = text[hit].map(clean)
    return cleaned, int(hit.sum())


def readmission_label(cohort: pd.DataFrame, max_days: int = 30) -> pd.DataFrame:
    """Add the label `readmission`: 1 when the next admission started within
    `max_days` of discharge. Also adds length_of_stay (whole days), which this
    task shows in its profile.

    The label is computed here from readmission_day, and has_readmission is set
    to the same value. Admissions that ended in death or have a negative gap are
    dropped.
    """
    df = cohort[left_alive(cohort)].copy()
    days = pd.to_numeric(df["readmission_day"], errors="coerce")
    df = df[~(days < 0)].copy()
    days = days.loc[df.index]
    df["readmission"] = ((days >= 0) & (days <= max_days)).astype(int)
    df["has_readmission"] = df["readmission"]
    stay = pd.to_datetime(df["dischtime"], errors="coerce") - pd.to_datetime(df["admittime"], errors="coerce")
    df["length_of_stay"] = stay.dt.days
    print(f"[readmission] next admission within {max_days} days: prevalence {df['readmission'].mean():.4f}")
    return df
