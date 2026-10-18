# Hospital-LOS-only cleaning functions: its spine (each patient's last admission) and the prior-admission counts.
"""Functions only the hospital length-of-stay task uses.

The task predicts from the FIRST 24 HOURS whether the stay runs past 3 days, so
nothing describing the rest of the stay may be used.
"""
from __future__ import annotations

import pandas as pd

from mimic_general import add_patient_info, read_table

SPINE_COLUMNS = [
    "subject_id", "hadm_id", "admittime", "dischtime", "edregtime", "edouttime",
    "deathtime", "discharge_location", "hospital_expire_flag", "gender", "age",
    "admission_type", "admission_location", "insurance", "language",
    "marital_status", "race",
]


def hosp_los_spine(admissions: pd.DataFrame, patients: pd.DataFrame,
                   max_patients: int | None = None) -> pd.DataFrame:
    """One row per patient: their LAST hospital admission. The clock starts at
    admittime."""
    adm = admissions.copy()
    adm["admittime"] = pd.to_datetime(adm["admittime"], errors="coerce")
    adm["dischtime"] = pd.to_datetime(adm["dischtime"], errors="coerce")
    adm = add_patient_info(adm, patients, "admittime")
    if max_patients is not None:
        adm = adm[adm["subject_id"].isin(set(adm["subject_id"].drop_duplicates().iloc[:max_patients]))].copy()

    # Order each patient's admissions by time, keeping patients in file order.
    adm["_patient_order"] = pd.factorize(adm["subject_id"])[0]
    adm = adm.sort_values(["_patient_order", "admittime"], kind="stable")
    spine = adm.groupby("subject_id", sort=False).tail(1)
    return spine[SPINE_COLUMNS].reset_index(drop=True)


def prior_admissions(spine: pd.DataFrame) -> pd.DataFrame:
    """n_prior_admissions and n_prior_admissions_1y: the patient's admissions
    that STARTED strictly before the spine admission (all, and within the past
    365 days). Known at admit time, so safe for a first-24h task."""
    adm = read_table("admissions")[["subject_id", "admittime"]].copy()
    adm["admittime"] = pd.to_datetime(adm["admittime"], errors="coerce")
    target = spine[["subject_id", "hadm_id", "admittime"]].rename(columns={"admittime": "target_time"})
    target["target_time"] = pd.to_datetime(target["target_time"], errors="coerce")
    pairs = adm.merge(target, on="subject_id", how="inner")
    prior = pairs[pairs["admittime"] < pairs["target_time"]]
    past_year = prior[prior["admittime"] >= prior["target_time"] - pd.Timedelta(days=365)]
    counts = pd.DataFrame({
        "n_prior_admissions": prior.groupby("hadm_id").size(),
        "n_prior_admissions_1y": past_year.groupby("hadm_id").size(),
    })
    return counts.reindex(pd.Index(spine["hadm_id"], name="hadm_id")).fillna(0).astype(int)
