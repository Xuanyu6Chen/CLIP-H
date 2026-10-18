# Mortality-only cleaning functions: the in-hospital death label and the cohort builder.
"""Functions only the in-hospital mortality task uses.

Moved unchanged from ``clip_h_lib.py``. The task predicts at admission whether the
patient dies during that admission.
"""
from __future__ import annotations

from typing import Dict, Sequence

import pandas as pd

from mimic_general import MORTALITY_FEATURE_COLUMNS, _assemble_admission_features


def assign_mortality_label(deathtime: "pd.Series | Sequence",
                           discharge_location: "pd.Series | Sequence | None" = None) -> pd.DataFrame:
    """Map death signals to the in-hospital mortality label. Pure function.

    ``y`` = 1 iff the patient died during the admission, evidenced by a non-null
    ``deathtime`` or a ``discharge_location`` of DIED. Everyone else is 0 (survived to
    discharge). Unlike the discharge task there are no exclusions: survival-to-discharge
    is a well-posed negative, so ``keep`` is True for all rows.
    """
    dt = pd.Series(deathtime)
    died = dt.notna() & (dt.astype("string").str.strip().str.upper() != "")
    if discharge_location is not None:
        dl = pd.Series(discharge_location).astype("string").str.strip().str.upper()
        died = died | dl.eq("DIED")
        died.index = dt.index
    # NA-safe cast. A null death signal (NaT deathtime and/or missing/Arrow-backed
    # <NA> discharge_location) is "not observed to have died" -> survived (0). Nullable
    # comparisons propagate <NA> via Kleene logic (e.g. `<NA>.eq("DIED")` -> <NA>), so
    # fill before the integer cast; otherwise `.astype(int)` raises on NAType.
    y = died.fillna(False).astype(bool).astype(int).astype("Int64")
    keep = pd.Series(True, index=dt.index)
    reason = pd.Series("", index=dt.index, dtype="object")
    return pd.DataFrame({"y": y, "keep": keep, "drop_reason": reason})

def build_cohort_mortality(tables: Dict[str, pd.DataFrame], seed: int = 42,
                           rich: bool = True) -> pd.DataFrame:
    """In-hospital mortality cohort. Label = died during the index admission
    (``deathtime`` present or ``discharge_location`` == DIED); prediction from
    admission-time signals only. Death-revealing and discharge-side columns are
    excluded from features (``MORTALITY_FEATURE_COLUMNS``)."""
    adm = _assemble_admission_features(tables, rich=rich)
    lab = assign_mortality_label(adm.get("deathtime"), adm.get("discharge_location"))
    adm["y"] = lab["y"].to_numpy()
    adm = adm[adm["age"].notna()].copy()
    adm["y"] = adm["y"].astype(int)
    print(f"      mortality: died={int((adm['y']==1).sum()):,} / "
          f"survived={int((adm['y']==0).sum()):,}", flush=True)
    cols = [c for c in MORTALITY_FEATURE_COLUMNS if c in adm.columns] + ["y"]
    return adm[cols].reset_index(drop=True)
