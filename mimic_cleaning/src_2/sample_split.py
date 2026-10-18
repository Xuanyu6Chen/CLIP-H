# Mortality and discharge: patient-level train/test split and the seeded, class-balanced subsample.
"""From a labelled cohort to the train and test rows, and the files written for them.

``patient_level_split`` and ``subsample_cohort`` are moved unchanged from ``clip_h_lib.py``.
Their random calls decide WHICH patients land in each half; changing them, even to an
equivalent-looking call, picks different patients. The writers at the end are new: they save
what the runner notebooks used to build in memory.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================================ #
# Patient-level split (no subject_id in both train and test)
# ============================================================================ #
def patient_level_split(
    df: pd.DataFrame,
    test_size: float = 0.15,
    seed: int = 42,
    group_col: str = "subject_id",
    label_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows so that no ``group_col`` (patient) appears in both halves.

    Splitting is performed on the set of unique patients (optionally stratified by
    a per-patient label = max of ``label_col`` over the patient's rows), then rows
    are assigned by membership. Guarantees
    ``set(train[group_col]) & set(test[group_col]) == set()``.
    """
    from sklearn.model_selection import train_test_split

    patients = df[[group_col]].drop_duplicates().reset_index(drop=True)
    # Degenerate cohort: with <2 unique patients a split is undefined, so return all
    # rows as train and an empty test rather than letting sklearn raise.
    if len(patients) < 2:
        return df.reset_index(drop=True), df.iloc[0:0].copy()
    stratify = None
    if label_col is not None and label_col in df.columns:
        plabel = df.groupby(group_col)[label_col].max()
        patients = plabel.reset_index().rename(columns={label_col: "_plabel"})
        # Only stratify if both classes have >= 2 patients (sklearn requirement).
        vc = patients["_plabel"].value_counts()
        if len(vc) > 1 and vc.min() >= 2:
            stratify = patients["_plabel"]

    train_ids, test_ids = train_test_split(
        patients[group_col].values,
        test_size=test_size,
        random_state=seed,
        stratify=stratify.values if stratify is not None else None,
    )
    train_ids, test_ids = set(train_ids.tolist()), set(test_ids.tolist())
    train_df = df[df[group_col].isin(train_ids)].reset_index(drop=True)
    test_df = df[df[group_col].isin(test_ids)].reset_index(drop=True)
    return train_df, test_df

def subsample_cohort(df: pd.DataFrame, n: Optional[int], seed: int = 42,
                     label_col: str = "y", method: str = "stratified",
                     pos_frac: Optional[float] = None,
                     min_total: Optional[int] = None) -> pd.DataFrame:
    """Reduce a split to ``n`` rows for a budgeted run.

    ``method='stratified'`` (default): a seeded, prevalence-preserving random sample that
    matches the cohort's class balance and removes any ordering bias from the row index or
    ``subject_id``. ``method='head'`` takes the first ``n`` rows (``df.iloc[:n]``).

    Without ``pos_frac``, returns all rows when ``n`` is None or ``n >= len(df)``. Call this on
    one side of a ``patient_level_split`` so patient-disjointness is preserved (the sample is
    drawn within an already patient-disjoint pool).

    ``pos_frac`` (optional): return a class-balanced sample with this target positive fraction by
    undersampling the majority (keeping all of the scarce class). Apply it to the learning side only
    so concept discovery, refinement and the significance gate see enough positives; leave the test
    split at its natural prevalence and pair with :func:`prevalence_correct` for calibrated
    probabilities. ``min_total`` is a support floor: balancing never shrinks the sample below it
    (negatives are topped up and class weighting absorbs the residual imbalance), so the SAE stays
    well supported when positives are scarce. Balance is relaxed only as far as the data forces.
    """
    # Class-balanced undersampling (learning side only; see pos_frac in the docstring).
    if pos_frac is not None and label_col in df.columns and df[label_col].nunique() > 1:
        rng = np.random.default_rng(seed)
        pos_idx = np.where(df[label_col].values == 1)[0]
        neg_idx = np.where(df[label_col].values == 0)[0]
        cap = len(df) if n is None else int(n)
        n_pos = int(round(pos_frac * cap)); n_neg = cap - n_pos
        if n_pos > len(pos_idx):
            n_pos = len(pos_idx); n_neg = int(round(n_pos * (1 - pos_frac) / pos_frac))
        if n_neg > len(neg_idx):
            n_neg = len(neg_idx); n_pos = min(len(pos_idx), int(round(n_neg * pos_frac / (1 - pos_frac))))
        # Support floor: never shrink the sample below `min_total` just to hit the target ratio.
        # When positives are scarce, keep all chosen positives and top up negatives toward the
        # floor; the residual imbalance is then absorbed by class weighting in the predictor. This
        # keeps representation learning stable (enough samples per SAE neuron) instead of trading
        # data for a nominal 50/50 split. `min_total` is capped by what the pool actually has.
        if min_total is not None and (n_pos + n_neg) < min_total:
            n_neg = min(len(neg_idx), max(n_neg, int(min_total) - n_pos))
        sel = np.concatenate([rng.choice(pos_idx, size=n_pos, replace=False),
                              rng.choice(neg_idx, size=n_neg, replace=False)])
        sel.sort()
        return df.iloc[sel].reset_index(drop=True)
    if n is None or n >= len(df):
        return df.reset_index(drop=True)
    if method == "head":
        return df.iloc[:n].reset_index(drop=True)
    if label_col in df.columns and df[label_col].nunique() > 1:
        from sklearn.model_selection import train_test_split
        keep_idx, _ = train_test_split(np.arange(len(df)), train_size=int(n),
                                       random_state=seed, stratify=df[label_col].values)
        return df.iloc[np.sort(keep_idx)].reset_index(drop=True)
    return df.sample(n=int(n), random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------- #
# Writers (new): the split files of data/<task>/ and the manifest
# ---------------------------------------------------------------------------- #
def feature_signature(feature_columns: List[str], forbidden_columns: List[str]) -> str:
    """The 8-character signature of the task's feature and forbidden column lists.

    The same value as ``clip_h_lib.config_hash({'feat': ..., 'forbid': ...}, 8)`` in the runner
    notebooks, which make it part of the run folder name, so a change to either list starts a
    new run folder.
    """
    cfg = {"feat": sorted(feature_columns), "forbid": sorted(forbidden_columns)}
    payload = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def _stop_if_exists(paths: List[Path], overwrite: bool) -> None:
    existing = [p.name for p in paths if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{paths[0].parent} already holds {existing}. "
                              "Set OVERWRITE = True in the run script to replace them.")


def save_cohort(cohort: pd.DataFrame, out_dir: Path, overwrite: bool) -> Path:
    """Write the whole cohort as ``cohort.parquet`` (every row and column, types kept)."""
    path = Path(out_dir) / "cohort.parquet"
    _stop_if_exists([path], overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_parquet(path, index=False)
    return path


def save_split(out_dir: Path, task: str, train_df: pd.DataFrame, test_df: pd.DataFrame,
               feature_columns: List[str], overwrite: bool) -> Dict[str, str]:
    """Write the train and test rows. Returns the paths.

    ``{split}_cohort.parquet``   the rows exactly as the runner notebooks use them: IDs, the
                                 label ``y`` and every feature, with their types and index
    ``{split}_features.csv``     the feature columns only, no IDs (as the other tasks)
    ``{split}_labels.csv``       one column named after the task, the label
    """
    out_dir = Path(out_dir)
    files = {f"{s}_{kind}": out_dir / f"{s}_{kind}.{ext}" for s in ("train", "test")
             for kind, ext in (("cohort", "parquet"), ("features", "csv"), ("labels", "csv"))}
    _stop_if_exists(list(files.values()), overwrite)
    out_dir.mkdir(parents=True, exist_ok=True)
    for s, df in (("train", train_df), ("test", test_df)):
        df.to_parquet(files[f"{s}_cohort"])
        df[[c for c in feature_columns if c in df.columns]].to_csv(files[f"{s}_features"], index=False)
        pd.DataFrame({task: np.asarray(df["y"], dtype=int)}).to_csv(files[f"{s}_labels"], index=False)
    return {name: str(path) for name, path in files.items()}


def write_manifest(out_dir: Path, manifest: Dict[str, Any], overwrite: bool) -> None:
    """Write manifest.json: the record of one run's settings and counts."""
    path = Path(out_dir) / "manifest.json"
    _stop_if_exists([path], overwrite)
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
