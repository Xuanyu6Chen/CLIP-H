# Discharge-only cleaning functions: the discharge-destination label, the admission-origin exclusion and the cohort builder.
"""Functions only the discharge-destination task uses.

Moved unchanged from ``clip_h_lib.py``. The task predicts at admission whether the
patient is discharged home without services.
"""
from __future__ import annotations

import time as _time
from typing import Dict, Sequence

import pandas as pd

from mimic_general import (
    DISCHARGE_FEATURE_COLUMNS,
    LAB_ITEMIDS,
    _comorbidity_features,
    _ed_prior_counts,
    _icu_features,
    _labs_value,
    _omr_vitals,
    _prior_counts,
    _service_feature,
    normalize_race,
)


# ---------------------------------------------------------------------------- #
# Discharge-disposition label mapping (built from discharge_location, after which the
# column is dropped). The clinical question is "independent home discharge" vs
# "needs ongoing support": Home = own residence with no services; Other = home
# with services (home health) or any facility/rehab. DIED/HOSPICE are end-of-life
# competing risks (not a disposition decision); AGAINST ADVICE is a patient-driven
# self-discharge (a different generative process); a missing discharge_location is a
# missing label, not a negative. ACUTE HOSPITAL is a transfer to another acute-care
# hospital (continued acute care, not a post-acute disposition) and is likewise
# excluded, following the Geneva post-acute-care score (Louis-Simonet et al.,
# J Gen Intern Med 2008; PMC2492858), which excludes in-hospital deaths and acute
# inter-hospital transfers from the home-vs-PAC cohort. All excluded groups are
# dropped, never labelled 0.
# ---------------------------------------------------------------------------- #
DISCHARGE_HOME_LOCATIONS = {"HOME"}

DISCHARGE_OTHER_LOCATIONS = {
    "HOME HEALTH CARE", "SKILLED NURSING FACILITY", "REHAB",
    "CHRONIC/LONG TERM ACUTE CARE", "PSYCH FACILITY",
    "OTHER FACILITY", "ASSISTED LIVING", "HEALTHCARE FACILITY",
}

DISCHARGE_EXCLUDE_LOCATIONS = {"DIED", "HOSPICE", "AGAINST ADVICE", "ACUTE HOSPITAL"}

# Sensitivity-analysis switch: treat HOME HEALTH CARE as Home rather than Other.
DISCHARGE_HOME_INCLUDES_HOMEHEALTH = False

# Admission-origin exclusion (clean_cohort). Patients transferred in from a
# skilled-nursing/residential facility were not living independently at baseline,
# so "discharge home" is not a well-posed target for them. Established discharge-
# disposition models either exclude such patients or build a separate model split on
# baseline residence (Geneva PAC score, PMC2492858; "Hospital-Day-1" SNF model,
# Dabilis et al., PMID 31133235; aspiration-pneumonia discharge models, PMC9955922,
# which fit one model for admitted-from-home and another for admitted-from-facility).
DISCHARGE_ADMIT_FROM_FACILITY_LOCATIONS = {"TRANSFER FROM SKILLED NURSING FACILITY"}

def discharge_admit_keep_mask(admission_location: "pd.Series | Sequence") -> pd.Series:
    """Boolean mask: False for admissions originating from a residential / skilled-
    nursing facility (baseline non-home residence). Pure function; see
    ``DISCHARGE_ADMIT_FROM_FACILITY_LOCATIONS`` for the literature rationale."""
    s = pd.Series(admission_location).astype("string").str.strip().str.upper()
    return ~s.isin(DISCHARGE_ADMIT_FROM_FACILITY_LOCATIONS)

def assign_discharge_label(
    discharge_location: "pd.Series | Sequence",
    include_homehealth: bool = DISCHARGE_HOME_INCLUDES_HOMEHEALTH,
) -> pd.DataFrame:
    """Map raw ``discharge_location`` to the discharge label, with explicit exclusions.

    Returns a DataFrame (indexed like the input when a Series) with:
      * ``y``           : Int64, 1 = Home, 0 = Other, <NA> for dropped rows
      * ``keep``        : bool, False for excluded/missing/unmapped rows
      * ``drop_reason`` : "" when kept, else "missing" / "<disposition>" / "unmapped"

    Pure function (no MIMIC data needed). See the disposition
    constants above for the Home/Other/Exclude mapping rationale.
    """
    s = pd.Series(discharge_location).astype("string").str.strip().str.upper()
    home = set(DISCHARGE_HOME_LOCATIONS)
    other = set(DISCHARGE_OTHER_LOCATIONS)
    if include_homehealth:
        home = home | {"HOME HEALTH CARE"}
        other = other - {"HOME HEALTH CARE"}

    is_missing = s.isna() | s.isin(["", "<NA>", "NAN", "NONE", "UNKNOWN"])
    is_excl = s.isin(DISCHARGE_EXCLUDE_LOCATIONS)
    is_home = s.isin(home)
    is_other = s.isin(other)

    y = pd.Series(pd.NA, index=s.index, dtype="Int64")
    y[is_home] = 1
    y[is_other] = 0

    reason = pd.Series("", index=s.index, dtype="object")
    reason[is_missing] = "missing"
    reason[is_excl] = s[is_excl].str.lower().astype(object)
    is_unmapped = ~(is_missing | is_excl | is_home | is_other)
    reason[is_unmapped] = "unmapped"

    keep = (is_home | is_other) & ~is_missing & ~is_excl
    return pd.DataFrame({"y": y, "keep": keep, "drop_reason": reason})

def build_cohort_discharge(tables: Dict[str, pd.DataFrame], seed: int = 42,
                           rich: bool = True, clean_cohort: bool = True) -> pd.DataFrame:
    """Assemble the leakage-safe discharge cohort (admission-time signals only).

    Label (``assign_discharge_label``): ``y`` = 1 for discharge Home, 0 for Other. The
    discharge fields are used only to build the label and are not exposed as features (no
    current-admission ICD/DRG).

    ``clean_cohort`` (default True) applies the literature-grounded admission-origin
    exclusion (drop patients transferred in from a skilled-nursing facility, who were
    not living independently at baseline). Set False for a sensitivity analysis on the
    full admitted-from-anywhere cohort. The disposition-side exclusions (death/hospice/
    AMA/acute-transfer/missing) are always applied via ``assign_discharge_label``.

    ``rich`` also computes medication, culture and service counts; these are whole-stay
    quantities, so the final selection of ``DISCHARGE_FEATURE_COLUMNS`` drops them.
    """
    _t0 = _time.time()
    _n = [0]
    def _step(msg: str, df: "pd.DataFrame | None" = None) -> None:
        _n[0] += 1
        rows = f"  {len(df):>9,} rows" if df is not None else ""
        print(f"    step {_n[0]:>2d}/{_DISCHARGE_STEPS}  {_time.time()-_t0:6.1f}s  {msg}{rows}", flush=True)

    _step("copy admissions + merge patients")
    adm = tables["admissions"].copy()
    pts = tables["patients"]
    adm = adm.merge(pts[["subject_id", "gender", "anchor_age", "anchor_year"]],
                    on="subject_id", how="left")

    _step("parse timestamps, compute age / LOS / race", adm)
    adm["admittime"] = pd.to_datetime(adm["admittime"], errors="coerce")
    adm["dischtime"] = pd.to_datetime(adm["dischtime"], errors="coerce")
    adm["age"] = (pd.to_numeric(adm["anchor_age"], errors="coerce")
                 + adm["admittime"].dt.year.subtract(
                     pd.to_numeric(adm["anchor_year"], errors="coerce"))).clip(upper=91).where(
                 lambda s: s >= 18)
    adm["length_of_stay_days"] = (adm["dischtime"] - adm["admittime"]).dt.days
    adm["race"] = adm["race"].map(normalize_race)

    _step("build label y (Home vs Other; exclude death/hospice/AMA/missing)", adm)
    _lab = assign_discharge_label(adm["discharge_location"],
                                  include_homehealth=DISCHARGE_HOME_INCLUDES_HOMEHEALTH)
    adm["y"] = _lab["y"]
    adm["_disch_keep"] = _lab["keep"].to_numpy()
    _n_home = int((_lab["y"] == 1).sum())
    _n_other = int((_lab["y"] == 0).sum())
    _dropped = _lab.loc[~_lab["keep"], "drop_reason"].value_counts().to_dict()
    print(f"      label: Home={_n_home:,} / Other={_n_other:,}; "
          f"dropped {int((~_lab['keep']).sum()):,} {_dropped}", flush=True)

    if clean_cohort and "admission_location" in adm.columns:
        _admit_keep = discharge_admit_keep_mask(adm["admission_location"]).to_numpy()
        _n_admit_excl = int((adm["_disch_keep"].astype(bool) & ~_admit_keep).sum())
        adm["_disch_keep"] = adm["_disch_keep"].astype(bool) & _admit_keep
        print(f"      clean_cohort: excluded {_n_admit_excl:,} admitted-from-facility "
              f"(baseline non-home residence)", flush=True)

    _step("prior admission counts (365d window)", adm)
    adm = adm.merge(_prior_counts(adm, windows=(365,)), on=["subject_id", "hadm_id"], how="left")

    _step("prior ED visit counts (180d window)", adm)
    adm = adm.merge(_ed_prior_counts(adm, tables.get("ed_stays"), window=180),
                    on=["subject_id", "hadm_id"], how="left")

    _step("Charlson comorbidities (prior admissions only)", adm)
    adm = adm.merge(_comorbidity_features(tables["diagnoses"], adm),
                    on="hadm_id", how="left")

    _step("lab aggregation: first-24h values (8 labs)", adm)
    adm = adm.merge(_labs_value(tables.get("labevents"), adm, "first", LAB_ITEMIDS),
                    on="hadm_id", how="left")

    _step("OMR vitals (BMI, BP, height, weight)", adm)
    adm = adm.merge(_omr_vitals(tables.get("omr"), adm["subject_id"]), on="subject_id", how="left")

    _step("ICU features (flag, LOS)", adm)
    adm = adm.merge(_icu_features(tables.get("icustays"), adm), on="hadm_id", how="left")

    _step("clinical service feature", adm)
    adm = adm.merge(_service_feature(tables.get("services"), adm), on="hadm_id", how="left")

    _step("ED triage vitals (optional)", adm)
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
        _step("rich clinical substrate (meds, cultures, services)", adm)
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
            pos = (micro.dropna(subset=["org_name"]).groupby("hadm_id").size()
                   .reset_index(name="_p")) if "org_name" in micro.columns else None
            adm = adm.merge(ncult, on="hadm_id", how="left")
            adm["n_cultures"] = adm["n_cultures"].fillna(0)
            if pos is not None:
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

    _step("drop missing age + excluded dispositions, select feature columns", adm)
    adm = adm[adm["age"].notna() & adm["_disch_keep"].astype(bool)].copy()
    adm["y"] = adm["y"].astype(int)
    keep = [c for c in DISCHARGE_FEATURE_COLUMNS if c in adm.columns] + ["y"]
    result = adm[keep].reset_index(drop=True)
    print(f"  Cohort ready: {len(result):,} rows x {len(result.columns)} cols  "
          f"(total {_time.time()-_t0:.1f}s)", flush=True)
    return result

_DISCHARGE_STEPS = 12
