# Sample draw (any class mix), feature-column selection, train/test split, split checks, and saving the split files.
"""From a labelled cohort to the train/test files.

The random calls in draw_sample and split_train_test decide WHICH patients
land in the sample and in each half. Changing them, even to an equivalent-looking
call, picks different patients and breaks every downstream result.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Never features, in any task: IDs, raw timestamps, and what labels are built from.
NEVER_FEATURES: list[str] = [
    "id", "subject_id", "hadm_id", "stay_id", "admittime", "dischtime",
    "edregtime", "edouttime", "deathtime", "hospital_expire_flag",
    "next_admit_time", "readmission_day", "has_readmission", "_los_hours",
]


def sample_counts(df: pd.DataFrame, label_col: str, total: int,
                  positive_share: float | str) -> tuple[int, int]:
    """How many positive and how many negative patients to draw, out of `total`.

    `positive_share` is the share of the sample with label 1: a number between
    0 and 1 (0.5 = balanced), or "prevalence" to keep the share of positive
    patients found in `df` itself.
    """
    if positive_share == "prevalence":
        labels = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
        share = df.loc[labels == 1, "subject_id"].nunique() / df["subject_id"].nunique()
    elif isinstance(positive_share, (int, float)) and 0 < positive_share < 1:
        share = float(positive_share)
    else:
        raise ValueError(f'positive share must be a number between 0 and 1, or "prevalence"; got {positive_share!r}')
    n_positive = round(total * share)
    n_negative = total - n_positive
    if min(n_positive, n_negative) < 1:
        raise ValueError(f"a sample of {total:,} with positive share {share:.4f} leaves a class empty")
    print(f"[draw] positive share {share:.4f} of {total:,} -> {n_positive:,} positive + {n_negative:,} negative")
    return n_positive, n_negative


def draw_sample(df: pd.DataFrame, label_col: str, n_positive: int, n_negative: int,
                seed: int) -> pd.DataFrame:
    """Draw `n_positive` positive and `n_negative` negative patients, one
    admission per patient.

    A patient counts as positive if any of their rows is positive, so nobody is
    drawn into both classes. Positive patients contribute their first positive
    row, negative patients their first row. One row per patient is also what
    keeps the train and test halves from sharing a patient.
    """
    rng = np.random.default_rng(seed)
    labels = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
    frame = df.assign(_label=labels).reset_index(drop=True)

    positive = frame.loc[frame["_label"] == 1, "subject_id"].unique()
    negative = np.setdiff1d(frame["subject_id"].unique(), positive)
    if len(positive) < n_positive or len(negative) < n_negative:
        raise ValueError(f"the draw needs {n_positive:,} positive / {n_negative:,} negative patients but only "
                         f"{len(positive):,} / {len(negative):,} are available")

    chosen_positive = rng.choice(np.sort(positive), n_positive, replace=False)
    chosen_negative = rng.choice(np.sort(negative), n_negative, replace=False)
    first_positive = frame[frame["_label"] == 1].groupby("subject_id", sort=False).head(1)
    first_any = frame.groupby("subject_id", sort=False).head(1)
    sample = pd.concat([
        first_positive[first_positive["subject_id"].isin(set(chosen_positive))],
        first_any[first_any["subject_id"].isin(set(chosen_negative))],
    ]).drop(columns="_label").sort_index().reset_index(drop=True)
    print(f"[draw] {len(df):,} rows -> {len(sample):,} ({n_positive:,} positive + {n_negative:,} negative, seed {seed})")
    return sample


def select_features(df: pd.DataFrame, columns: list[str], never: list[str]) -> pd.DataFrame:
    """Keep exactly `columns`, in that order. Nothing else can become a feature.

    Stops the run if a wanted column is missing from the cohort, or if one of
    them is on the `never` list (IDs, timestamps, label inputs).
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"feature column(s) missing from the cohort: {missing}")
    banned = sorted(set(columns) & set(never))
    if banned:
        raise ValueError(f"column(s) that may never be features: {banned}")
    return df[columns].reset_index(drop=True)


def split_train_test(X: pd.DataFrame, y: pd.DataFrame, n_test: int, seed: int
                     ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[int], list[int]]:
    """Stratified train/test split with exactly `n_test` test rows: each class
    is split separately and gives the test set its proportional share, so both
    halves keep the class mix. Returns the four tables and the row numbers
    (into X) that went to train and to test."""
    if len(X) != len(y):
        raise ValueError(f"X/y length mismatch: {len(X)} != {len(y)}")
    if not 0 < n_test < len(X):
        raise ValueError(f"test size must be between 1 and {len(X) - 1:,} patients; got {n_test:,}")
    classes = [list(index_values) for _, index_values in y.groupby(y.columns[0], sort=False).groups.items()]

    # Each class's share of the test rows, rounded down; rows still missing go
    # to the classes with the largest remainders, so the total is exactly n_test.
    counts = [len(rows) * n_test // len(X) for rows in classes]
    by_remainder = sorted(range(len(classes)), key=lambda i: len(classes[i]) * n_test % len(X), reverse=True)
    for i in by_remainder[: n_test - sum(counts)]:
        counts[i] += 1
    if any(count == 0 or count == len(rows) for count, rows in zip(counts, classes)):
        raise ValueError(f"a test size of {n_test:,} leaves a class with no test rows or no train rows")

    rng = random.Random(seed)
    train_rows: list[int] = []
    test_rows: list[int] = []
    for rows, count in zip(classes, counts):
        rng.shuffle(rows)
        test_rows.extend(rows[:count])
        train_rows.extend(rows[count:])

    # One more seeded shuffle fixes the final row order of each half.
    order_rng = random.Random(seed)
    train_rows, test_rows = sorted(train_rows), sorted(test_rows)
    order_rng.shuffle(train_rows)
    order_rng.shuffle(test_rows)
    return (X.iloc[train_rows].reset_index(drop=True), X.iloc[test_rows].reset_index(drop=True),
            y.iloc[train_rows].reset_index(drop=True), y.iloc[test_rows].reset_index(drop=True),
            train_rows, test_rows)


def check_split(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame,
                y_test: pd.DataFrame, patient_ids: pd.Series, train_rows: list[int],
                test_rows: list[int], never: list[str], n_test: int) -> dict[str, Any]:
    """Stop the run if the split is not well-formed; return its counts.

    Checks: features and labels have the same rows; train and test have the same
    columns; no banned column is a feature; the test set has exactly `n_test`
    rows; no patient is on both sides.
    """
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("features and labels have different row counts")
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("train and test feature columns differ")
    banned = sorted(set(never) & set(X_train.columns))
    if banned:
        raise ValueError(f"banned column(s) among the features: {banned}")
    if len(X_test) != n_test:
        raise ValueError(f"test set has {len(X_test):,} rows, expected {n_test:,}")
    train_patients = set(patient_ids.iloc[train_rows].dropna().astype(str))
    test_patients = set(patient_ids.iloc[test_rows].dropna().astype(str))
    overlap = sorted(train_patients & test_patients)
    if overlap:
        raise ValueError(f"{len(overlap)} patient(s) in both train and test, e.g. {overlap[:10]}")
    return {
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "train_label_counts": {str(k): int(v) for k, v in y_train.iloc[:, 0].value_counts().items()},
        "test_label_counts": {str(k): int(v) for k, v in y_test.iloc[:, 0].value_counts().items()},
        "train_patients": len(train_patients),
        "test_patients": len(test_patients),
    }


def save_split(out_dir: Path, X_train: pd.DataFrame, X_test: pd.DataFrame,
               y_train: pd.DataFrame, y_test: pd.DataFrame, overwrite: bool) -> dict[str, str]:
    """Write train/test features and labels as four CSVs; return their paths.
    Stops if any of them is already there and `overwrite` is false."""
    tables = {"train_features": X_train, "test_features": X_test,
              "train_labels": y_train, "test_labels": y_test}
    existing = [f"{name}.csv" for name in tables if (out_dir / f"{name}.csv").exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{out_dir} already holds {existing}. Set OVERWRITE = True in the run script to replace them.")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)
    return {name: str(out_dir / f"{name}.csv") for name in tables}


def write_manifest(out_dir: Path, manifest: dict[str, Any], overwrite: bool) -> None:
    """Write manifest.json: the record of one run's settings and counts.
    Stops if it is already there and `overwrite` is false."""
    path = out_dir / "manifest.json"
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Set OVERWRITE = True in the run script to replace it.")
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
