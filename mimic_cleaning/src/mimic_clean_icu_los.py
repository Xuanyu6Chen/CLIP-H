# ICU-LOS-only cleaning function: its spine (each patient's last ICU stay, clock on ICU entry).
"""Functions only the ICU length-of-stay task uses.

The task predicts from the FIRST 24 HOURS OF THE ICU STAY whether that stay
runs past 4 days.
"""
from __future__ import annotations

import pandas as pd

from mimic_general import add_patient_info

SPINE_COLUMNS = [
    "subject_id", "hadm_id", "stay_id", "admittime", "dischtime", "edregtime",
    "edouttime", "deathtime", "discharge_location", "hospital_expire_flag",
    "gender", "age", "admission_type", "admission_location", "insurance",
    "language", "marital_status", "race",
]

ADMISSION_CONTEXT = [
    "hadm_id", "admission_type", "admission_location", "insurance", "language",
    "marital_status", "race", "discharge_location", "hospital_expire_flag",
    "edregtime", "edouttime", "deathtime",
]


def icu_los_spine(admissions: pd.DataFrame, patients: pd.DataFrame,
                  icustays: pd.DataFrame, max_patients: int | None = None) -> pd.DataFrame:
    """One row per patient: their LAST ICU stay.

    The ICU clock drives everything: admittime := ICU intime and dischtime :=
    ICU outtime. hadm_id is kept so the hospital tables (labs, medications,
    cultures) still join, but they are windowed on the ICU entry, i.e. the
    first 24h OF THE ICU STAY, which can be days after hospital admission.
    """
    icu = icustays.copy()
    icu["intime"] = pd.to_datetime(icu["intime"], errors="coerce")
    icu["outtime"] = pd.to_datetime(icu["outtime"], errors="coerce")
    icu = icu.dropna(subset=["intime", "outtime"])
    icu = icu.sort_values(["subject_id", "intime"]).groupby("subject_id", as_index=False).last()
    icu = icu.merge(admissions[[c for c in ADMISSION_CONTEXT if c in admissions.columns]],
                    on="hadm_id", how="left")
    icu = add_patient_info(icu, patients, "intime")
    if max_patients is not None:
        icu = icu[icu["subject_id"].isin(set(icu["subject_id"].drop_duplicates().iloc[:max_patients]))].copy()
    icu["admittime"] = icu["intime"]
    icu["dischtime"] = icu["outtime"]
    return icu[[c for c in SPINE_COLUMNS if c in icu.columns]].reset_index(drop=True)
