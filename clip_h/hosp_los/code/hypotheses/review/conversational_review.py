"""Phase 2C: the conversational review. Builds the prompt that is pasted into a chat model
and validates the JSON reply that comes back.

    build_concept_exemplars    raw values of the patients each candidate is true for
    los_raw_facts, icu_raw_facts, readmission_raw_facts
                               how one patient's raw values are printed, one function per task
                               (the notebook setting RAW_FACTS names the one to use)
    build_review_prompt        prompts/review.txt + the numbered candidates
    validate_review_response   checks the reply (keys, score ranges, votes, every candidate once)
    ingest_review_response     reads the reply file, applies the keep rule, writes the result files
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


# --- Readmission: raw values tagged with the HOSPITAL score ----------------------------------
# The HOSPITAL score for 30-day potentially avoidable readmission (Donzé et al.) is the
# reference knowledge of the readmission review. A patient's raw values are shown to the
# reviewer with a tag where they meet a HOSPITAL component; the 0/1 flags are never shown.
HOSPITAL_SCORE_URL = "https://www.mdcalc.com/calc/10051/hospital-score-readmissions"

# The HOSPITAL components that can be read from one feature column (oncology discharge and
# prior-year admissions cannot). A component is shown only when its column is in the task's
# features file; a column that is missing is skipped without a message. Each entry:
#   (column, label, criterion, threshold, unit, hospital_marker)
# criterion: "lt" meets when value < threshold; "ge" when value >= threshold;
#            "gt0" when count > 0; "nonelective" when admission_type is not elective.
HOSPITAL_FACT_FIELDS: tuple[tuple[str, str, str, float | None, str, str], ...] = (
    ("hemoglobin", "Hgb", "lt", 12.0, "g/dL", "H"),
    ("sodium", "Na", "lt", 135.0, "mEq/L", "S"),
    ("procedure_count", "procedures", "gt0", 0.0, "", "P"),
    ("admission_type", "admission", "nonelective", None, "", "I"),
    ("los_days", "LOS", "ge", 5.0, "d", "L"),
)
ELECTIVE_ADMISSION_TYPES = {"ELECTIVE", "SURGICAL SAME DAY ADMISSION"}


def _format_field(column: str, label: str, criterion: str, threshold: float | None,
                  unit: str, marker: str, value: Any) -> str | None:
    """Format one HOSPITAL-component raw value, tagging it when it meets the criterion."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    if criterion == "nonelective":
        text = str(value).strip()
        meets = text.upper() not in ELECTIVE_ADMISSION_TYPES
        body = f"{label} {text}"
    else:
        try:  # numeric criteria; a value that is not one number (a lab trajectory string) is skipped
            number = float(value)
        except (TypeError, ValueError):
            return None
        if criterion == "gt0":
            count = int(round(number))
            meets = count > 0
            body = f"{label} {count}"
        elif criterion == "ge":
            meets = number >= threshold
            body = f"{label} {number:g}{unit}"
        else:  # "lt"
            meets = number < threshold
            body = f"{label} {number:g} {unit}"
    return f"{body} [HOSPITAL {marker}+]" if meets else body


def format_raw_facts(row: Mapping[str, Any], withhold: Iterable[str] = ()) -> str:
    """HOSPITAL-component raw values for one patient (e.g. 'Hgb 8.8 [HOSPITAL H+]; Na 142').

    Thresholds follow the HOSPITAL readmission score (see HOSPITAL_SCORE_URL).
    `withhold` names columns to leave out, so that a reviewer cannot read a HOSPITAL component
    off the raw values of a candidate that is meant to be judged without it.
    """
    blocked = set(withhold)
    parts = [
        _format_field(column, label, criterion, threshold, unit, marker, row.get(column))
        for column, label, criterion, threshold, unit, marker in HOSPITAL_FACT_FIELDS
        if column not in blocked
    ]
    return "; ".join(part for part in parts if part)


def readmission_raw_facts(row: Mapping[str, Any], withhold: Iterable[str] = ()) -> str:
    """``format_raw_facts`` for the readmission profiles, whose labs cover the whole stay.

    Hemoglobin and sodium are trajectory strings there ('9.1*@5h; 8.8*@70h'); the last measured
    value (the one closest to discharge) is taken, so the two HOSPITAL lab components are shown.
    """
    row = dict(row)
    for column in ('hemoglobin', 'sodium'):
        value = row.get(column)
        if isinstance(value, str) and '@' in value:
            numbers = re.findall(r'(-?\d+\.?\d*)\*?@', value)
            row[column] = float(numbers[-1]) if numbers else None
    return format_raw_facts(row, withhold)


def build_concept_exemplars(
    candidates: pd.DataFrame,
    firing: np.ndarray,
    features: pd.DataFrame,
    *,
    k: int = 3,
    withhold: Iterable[str] = (),
    fact_fn=None,
) -> dict[int, list[str]]:
    """Per candidate, raw-value fact sheets for the top-k patients where the candidate FIRES.

    `firing` is an (n_patients, n_candidates) matrix -- column j is the binary annotation of
    candidates row j -- and `features` is the raw feature frame positionally aligned to `firing`
    rows. It uses each candidate's own annotation column, so it also works for a hypothesis that
    no longer belongs to one neuron (after Phase 2B).
    `fact_fn(row, withhold) -> str` prints one patient's raw values: `los_raw_facts` (hospital
    length of stay), `icu_raw_facts` (ICU length of stay) or `readmission_raw_facts`; the
    default is `format_raw_facts`.
    Keyed by the candidate's row index to line up with the numbered review prompt.
    """
    fact_fn = fact_fn or format_raw_facts
    firing = np.asarray(firing)
    exemplars: dict[int, list[str]] = {}
    for pos, (index, row) in enumerate(candidates.iterrows()):
        acts = firing[:, pos]
        order = np.argsort(acts)[::-1][:k]
        facts = [
            fact_fn(features.iloc[int(i)], withhold)
            for i in order if acts[int(i)] > 0
        ]
        exemplars[index] = [fact for fact in facts if fact]
    return exemplars


# --- Hospital and ICU length of stay: first-24h raw values -----------------------------------
# Flags are shown by name when set; values are shown as they are (labs are first-24h
# trajectory strings, not single numbers). The two tasks differ in the name of the ICU flag.
# A column that is not in the task's features file is skipped without a message.
LOS_FLAG_FIELDS = (('was_in_icu_24h', 'ICU-24h'), ('mechanical_ventilation_flag', 'mech-vent'),
                   ('dialysis_flag', 'dialysis'), ('oliguria_flag', 'oliguria'))
ICU_FLAG_FIELDS = (('was_in_icu', 'ICU'),) + LOS_FLAG_FIELDS[1:]
LOS_VALUE_FIELDS = (('admission_type', 'adm'), ('service', 'svc'), ('lactate', 'lactate'),
                    ('creatinine', 'creat'), ('hemoglobin', 'Hgb'), ('wbc', 'WBC'),
                    ('abnormal_lab_count', 'abn-labs'))


def los_raw_facts(row: Mapping[str, Any], withhold: Iterable[str] = ()) -> str:
    """First-24h raw values for one patient of the hospital length-of-stay task
    (e.g. 'ICU-24h; adm EW EMER.; lactate 2.1@3h; 1.4@15h'). ``withhold`` names columns to leave out.
    """
    return _first_day_facts(row, withhold, LOS_FLAG_FIELDS)


def icu_raw_facts(row: Mapping[str, Any], withhold: Iterable[str] = ()) -> str:
    """The same for the ICU length-of-stay task, whose flag list names ``was_in_icu`` for the ICU."""
    return _first_day_facts(row, withhold, ICU_FLAG_FIELDS)


def _first_day_facts(row: Mapping[str, Any], withhold: Iterable[str], flag_fields) -> str:
    blocked = set(withhold)
    parts = []
    for column, label in flag_fields:
        if column in blocked:
            continue
        value = row.get(column)
        if pd.notna(value) and str(value).strip().lower() not in ('', '0', '0.0', 'false', 'nan', 'no'):
            parts.append(label)
    for column, label in LOS_VALUE_FIELDS:
        if column in blocked:
            continue
        value = row.get(column)
        if value is None or (not isinstance(value, str) and pd.isna(value)):
            continue
        text = str(value).strip()
        if text and text.lower() != 'nan':
            parts.append(f"{label} {text}")
    return "; ".join(parts)


FORBIDDEN_HYPOTHESIS_TEXT = re.compile(
    r"\b(readmission|readmitted|future admissions?|discharge outcome|death|mortality|"
    r"DNR|DNI|RNS|code status|resuscitation status)\b",
    re.IGNORECASE,
)
VALID_DIRECTIONS = {"increase", "decrease", "unclear"}


def _candidate_line(index: int, row: pd.Series,
                    exemplars: Mapping[int, list[str]] | None = None) -> str:
    details: list[str] = []
    activation_frequency = row.get("activation_frequency")
    if pd.notna(activation_frequency):
        details.append(f"activation_frequency={float(activation_frequency):.3f}")
    fidelity_columns = [column for column in row.index if str(column).endswith("_fidelity_score")]
    if fidelity_columns and pd.notna(row[fidelity_columns[0]]):
        details.append(f"fidelity={float(row[fidelity_columns[0]]):.3f}")
    suffix = f"  [{'; '.join(details)}]" if details else ""
    line = f'{index}: "{str(row["interpretation"]).strip()}"{suffix}'
    rows = (exemplars or {}).get(index) or []
    if rows:
        line += "".join(f"\n    top-activating raw values: {fact}" for fact in rows)
    return line


def build_review_prompt(candidates: pd.DataFrame, instructions: str, n_reviewers: int = 3,
                        exemplars: Mapping[int, list[str]] | None = None) -> str:
    """The complete review prompt: ``instructions`` (the text of ``prompts/review.txt``)
    with its slots filled, followed by the numbered candidates and their exemplars."""
    if 'interpretation' not in candidates.columns or candidates.empty:
        raise ValueError('Candidates must contain a nonempty interpretation column')
    numbered = '\n'.join(_candidate_line(i, row, exemplars) for i, row in candidates.iterrows())
    body = instructions.replace('{hospital_url}', HOSPITAL_SCORE_URL).replace('{n_reviewers}', str(n_reviewers))
    return body + '\n\nCANDIDATES:\n' + numbered + '\n'


def _strict_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def validate_review_response(
    payload: Any,
    candidate_count: int,
    n_reviewers: int = 3,
) -> list[dict[str, Any]]:
    """Validate strict schema, reviewer math, and one-to-one source coverage."""
    if not isinstance(payload, dict) or set(payload) != {"hypotheses"}:
        raise ValueError('Response must be one object containing only "hypotheses"')
    hypotheses = payload["hypotheses"]
    if not isinstance(hypotheses, list) or not hypotheses:
        raise ValueError('"hypotheses" must be a nonempty list')

    required = {
        "hypothesis", "source_ids", "reviewer_votes", "clinical_validity",
        "leakage_risk", "expected_direction", "mechanism", "rationale",
    }
    seen_ids: set[int] = set()
    seen_text: set[str] = set()
    validated: list[dict[str, Any]] = []

    for index, item in enumerate(hypotheses):
        if not isinstance(item, dict) or set(item) != required:
            raise ValueError(f"Hypothesis {index} must contain exactly {sorted(required)}")
        text = str(item["hypothesis"]).strip()
        if not text or FORBIDDEN_HYPOTHESIS_TEXT.search(text):
            raise ValueError(f"Hypothesis {index} is empty or contains forbidden outcome text")
        normalized = re.sub(r"\W+", " ", text.lower()).strip()
        if normalized in seen_text:
            raise ValueError(f"Duplicate merged hypothesis text at item {index}")
        seen_text.add(normalized)

        source_ids = item["source_ids"]
        if not isinstance(source_ids, list):
            raise ValueError(f"Hypothesis {index} source_ids must be a list")
        # [] marks a hypothesis ADDED by the reviewers (from the reference knowledge, with no
        # SAE source). It is allowed and marked ``reviewer_added``; it goes through dedup, the
        # 2E gate and Phase 3 like any other candidate.
        current_ids: list[int] = []
        for value in source_ids:
            source_id = _strict_int(value, f"hypothesis {index} source id")
            if source_id < 0 or source_id >= candidate_count:
                raise ValueError(f"Hypothesis {index} contains out-of-range source id {source_id}")
            if source_id in seen_ids or source_id in current_ids:
                raise ValueError(f"Source id {source_id} is duplicated")
            current_ids.append(source_id)
        seen_ids.update(current_ids)

        votes = item["reviewer_votes"]
        if not isinstance(votes, dict) or set(votes) != {"valid", "invalid"}:
            raise ValueError(f"Hypothesis {index} reviewer_votes must contain valid and invalid")
        valid = _strict_int(votes["valid"], f"hypothesis {index} valid votes")
        invalid = _strict_int(votes["invalid"], f"hypothesis {index} invalid votes")
        if min(valid, invalid) < 0 or valid + invalid != n_reviewers:
            raise ValueError(f"Hypothesis {index} reviewer votes must sum to {n_reviewers}")

        clinical_validity = _strict_int(item["clinical_validity"], "clinical_validity")
        leakage_risk = _strict_int(item["leakage_risk"], "leakage_risk")
        if clinical_validity not in range(4) or leakage_risk not in range(3):
            raise ValueError(f"Hypothesis {index} contains an out-of-range score")
        direction = str(item["expected_direction"]).strip().lower()
        if direction not in VALID_DIRECTIONS:
            raise ValueError(f"Hypothesis {index} has invalid expected_direction")

        validated.append({
            "interpretation": text,
            "source_ids": current_ids,
            "reviewer_added": not current_ids,
            "valid_votes": valid,
            "invalid_votes": invalid,
            "clinical_validity": clinical_validity,
            "leakage_risk": leakage_risk,
            "expected_direction": direction,
            "mechanism": str(item["mechanism"]).strip(),
            "rationale": str(item["rationale"]).strip(),
        })

    n_added = sum(1 for row in validated if row["reviewer_added"])
    if n_added > 10:
        raise ValueError(f"{n_added} reviewer-added hypotheses — cap is 10; trim the additions")
    expected_ids = set(range(candidate_count))
    if seen_ids != expected_ids:
        missing = sorted(expected_ids - seen_ids)
        raise ValueError(f"source_ids do not cover every candidate exactly once; missing={missing}")
    return validated


def ingest_review_response(
    response_path: Path,
    candidates: pd.DataFrame,
    results_dir: Path,
    *,
    n_reviewers: int = 3,
    max_hypotheses: int = 20,
    min_clinical_validity: int = 1,
    leakage_drop: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Read the review reply, check it, apply the keep rule and write the result files.

    A reviewed hypothesis is kept with a majority of valid votes, clinical validity at or above
    ``min_clinical_validity`` and leakage risk below ``leakage_drop``; at most ``max_hypotheses``
    are kept (most valid votes first, then highest clinical validity). Writes
    ``hypotheses_conversational_merged.csv`` (every reviewed hypothesis with its ``keep`` flag),
    ``hypotheses.csv`` (the kept ones) and ``conversational_curation_summary.json`` to
    ``results_dir``. Returns ``(kept, all reviewed, summary)``.
    """
    raw = Path(response_path).read_text(encoding="utf-8").strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Conversational response is not strict JSON") from exc
    rows = validate_review_response(payload, len(candidates), n_reviewers)
    merged = pd.DataFrame(rows)
    majority = n_reviewers // 2 + 1
    merged["keep"] = (
        (merged["valid_votes"] >= majority)
        & (merged["clinical_validity"] >= min_clinical_validity)
        & (merged["leakage_risk"] < leakage_drop)
    )
    final = (
        merged[merged["keep"]]
        .sort_values(["valid_votes", "clinical_validity"], ascending=False)
        .head(max_hypotheses)
        .reset_index(drop=True)
    )
    if final.empty:
        raise ValueError("No hypotheses passed the conversational review rules")

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    serializable = merged.copy()
    serializable["source_ids"] = serializable["source_ids"].map(json.dumps)
    final_serializable = final.copy()
    final_serializable["source_ids"] = final_serializable["source_ids"].map(json.dumps)
    serializable.to_csv(results_dir / "hypotheses_conversational_merged.csv", index=False)
    final_serializable.to_csv(results_dir / "hypotheses.csv", index=False)
    summary = {
        "source": "conversational",
        "candidate_count": int(len(candidates)),
        "merged_concept_count": int(len(merged)),
        "kept_count": int(len(final)),
        "dropped_low_validity": int((merged["clinical_validity"] < min_clinical_validity).sum()),
        "dropped_leakage": int((merged["leakage_risk"] >= leakage_drop).sum()),
        "dropped_minority_vote": int((merged["valid_votes"] < majority).sum()),
        "n_reviewers": n_reviewers,
        "reviewer_majority": majority,
    }
    (results_dir / "conversational_curation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return final, merged, summary
