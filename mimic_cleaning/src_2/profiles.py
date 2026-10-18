# Mortality and discharge: the patient profile text, one paragraph per admission from a fixed template.
"""Template profiles for mortality and discharge. No LLM is involved.

The profile functions are moved unchanged from ``clip_h_lib.py``; both tasks use the same
template. The two writers at the end are new and write the same lines the runner notebooks
used to export.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence

from mimic_general import is_missing


# ============================================================================ #
# Profile builders (deterministic text templates)
# ============================================================================ #
def _join_clause(label: str, bits: List[str]) -> Optional[str]:
    bits = [b for b in bits if b]
    if not bits:
        return None
    return f"{label}: " + ", ".join(bits) + "."

def make_profile_discharge(row) -> str:
    """Deterministic clinical paragraph for the discharge task (admission-time)."""
    r = row if isinstance(row, dict) else row.to_dict()

    def f(k):
        v = r.get(k)
        return None if is_missing(v) else str(v).strip()

    parts: List[str] = []
    demo = " ".join(x for x in [f("age") and f"{f('age')}-year-old", f("race"), f("gender")] if x)
    if demo:
        parts.append(f"The patient is a {demo}.")
    socio = _join_clause("Socio-demographics", [
        f("marital_status") and f"marital status {f('marital_status')}",
        f("insurance") and f"insurance {f('insurance')}",
        f("language") and f"language {f('language')}",
    ])
    if socio:
        parts.append(socio)
    adm = _join_clause("Admission", [
        f("admission_type") and f"type {str(f('admission_type')).lower()}",
        f("admission_location") and f"location {str(f('admission_location')).lower()}",
        f("arrival_transport") and f"arrival {str(f('arrival_transport')).lower()}",
        f("triage_acuity") and f"triage acuity {f('triage_acuity')}",
        f("chief_complaint") and f"chief complaint {f('chief_complaint')}",
    ])
    if adm:
        parts.append(adm)
    triage = _join_clause("Triage vitals", [
        f("triage_temperature") and f"temp {f('triage_temperature')}",
        f("triage_heartrate") and f"HR {f('triage_heartrate')}",
        f("triage_resprate") and f"RR {f('triage_resprate')}",
        f("triage_o2sat") and f"O2sat {f('triage_o2sat')}",
        f("triage_sbp") and f"SBP {f('triage_sbp')}",
        f("triage_dbp") and f"DBP {f('triage_dbp')}",
        f("triage_pain") and f"pain {f('triage_pain')}",
    ])
    if triage:
        parts.append(triage)
    labs = _join_clause("Early labs", [
        f("lab_albumin_first") and f"albumin {f('lab_albumin_first')}",
        f("lab_creatinine_first") and f"creatinine {f('lab_creatinine_first')}",
        f("lab_hemoglobin_first") and f"hemoglobin {f('lab_hemoglobin_first')}",
        f("lab_sodium_first") and f"sodium {f('lab_sodium_first')}",
        f("lab_bun_first") and f"BUN {f('lab_bun_first')}",
        f("lab_glucose_first") and f"glucose {f('lab_glucose_first')}",
        f("lab_wbc_first") and f"WBC {f('lab_wbc_first')}",
        f("lab_platelets_first") and f"platelets {f('lab_platelets_first')}",
    ])
    if labs:
        parts.append(labs)
    vitals = _join_clause("Vitals", [
        f("bmi") and f"BMI {f('bmi')}",
        (f("systolic_bp") and f("diastolic_bp")) and f"BP {f('systolic_bp')}/{f('diastolic_bp')}",
        f("height_cm") and f"height {f('height_cm')}cm",
        f("weight_kg") and f"weight {f('weight_kg')}kg",
    ])
    if vitals:
        parts.append(vitals)
    care = _join_clause("Care context", [
        (str(f("icu_flag")) in {"1", "1.0", "True", "true"}) and "ICU stay during admission",
        f("icu_los_days") and f"ICU LOS {f('icu_los_days')} days",
        (str(f("ventilation_flag")) in {"1", "1.0", "True", "true"}) and "mechanical ventilation",
        (str(f("dialysis_flag")) in {"1", "1.0", "True", "true"}) and "dialysis",
        f("curr_service") and f"service {f('curr_service')}",
    ])
    if care:
        parts.append(care)
    course = _join_clause("Clinical course", [
        f("n_medications") and f"{f('n_medications')} distinct medications",
        (str(f("polypharmacy_flag")) in {"1", "1.0", "True", "true"}) and "polypharmacy (>=10 meds)",
        f("n_cultures") and f"{f('n_cultures')} microbiology cultures ordered",
        (str(f("positive_culture_flag")) in {"1", "1.0", "True", "true"}) and "positive culture",
        f("n_services") and f"{f('n_services')} clinical services involved",
    ])
    if course:
        parts.append(course)
    hist = _join_clause("Prior history", [
        f("n_admissions_prior_365d") and f"{f('n_admissions_prior_365d')} admissions in prior year",
        f("n_ed_visits_prior_180d") and f"{f('n_ed_visits_prior_180d')} ED visits in prior 6 months",
        f("charlson_prior") and f"prior Charlson index {f('charlson_prior')}",
    ])
    if hist:
        parts.append(hist)
    if f("length_of_stay_days"):
        parts.append(f"Length of stay: {f('length_of_stay_days')} days.")
    return " ".join(parts) or "Patient record."

def make_profile_mortality(row) -> str:
    """Admission-time clinical paragraph for the in-hospital mortality task (same template
    as ``make_profile_discharge``)."""
    return make_profile_discharge(row)


# ---------------------------------------------------------------------------- #
# Writers (new): the profile files of data/<task>/
# ---------------------------------------------------------------------------- #
def write_profiles(profiles: Sequence[str], out_jsonl: Path, overwrite: bool) -> None:
    """One JSON line per row: ``{"idx", "profile"}``, in the row order of the split."""
    out_jsonl = Path(out_jsonl)
    if out_jsonl.exists() and not overwrite:
        raise FileExistsError(f"{out_jsonl} already exists. Set OVERWRITE = True in the run script to replace it.")
    with out_jsonl.open("w", encoding="utf-8") as f:
        f.writelines(json.dumps({"idx": i, "profile": str(p)}, ensure_ascii=False) + "\n"
                     for i, p in enumerate(profiles))


def write_profiles_with_labels(profiles: Sequence[str], labels: Sequence, ids: Sequence,
                               subject_ids: Optional[Sequence], out_jsonl: Path,
                               overwrite: bool) -> None:
    """One JSON line per patient: ``{"id", "label", "profile"}`` plus ``subject_id``.

    ``id`` is the admission's ``hadm_id``. This is the file the comparators read.
    """
    out_jsonl = Path(out_jsonl)
    if out_jsonl.exists() and not overwrite:
        raise FileExistsError(f"{out_jsonl} already exists. Set OVERWRITE = True in the run script to replace it.")
    with out_jsonl.open("w", encoding="utf-8") as f:
        for i, (p, y) in enumerate(zip(profiles, labels)):
            rec = {"id": (int(ids[i]) if str(ids[i]).lstrip("-").isdigit() else str(ids[i])),
                   "label": int(y), "profile": str(p)}
            if subject_ids is not None:
                rec["subject_id"] = int(subject_ids[i])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
