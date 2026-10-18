"""Phase 0: load the profiles and labels of a split.

    prepare_split          builds {split}.jsonl when it is missing, then loads it
    build_handoff          joins {split}_profile.jsonl + {split}_labels.csv into {split}.jsonl
                           ({id, profile, label} per row); stops on code-status text
    load_split             reads {split}.jsonl into (profiles, labels, ids)
    report_splits          prints the size and positive rate of each split
    stratified_subsample   seeded subsample that keeps the outcome rate
    write_json, read_json, json_safe   small JSON helpers of the runner notebooks
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

# Final notebook-side leakage guard. The cleaning pipeline should already have stripped
# code-status text; this is defense-in-depth so a stray profile fails loudly at load time.
_FORBIDDEN = re.compile(
    r"\bDNR\b|do[-\s]*not[-\s]*resuscitate|code\s+status|\bDNI\b|\bRNS\s+status\b|resuscitation[-\s]+status",
    re.IGNORECASE,
)


def read_jsonl(path):
    """Read a JSONL file into a list of dicts (blank lines skipped)."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_label_csv(path):
    """Read a single-column label CSV into a list of ints."""
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or len(reader.fieldnames) != 1:
            raise ValueError(f"Expected one label column in {path}; got {reader.fieldnames}")
        col = reader.fieldnames[0]
        return [int(float(row[col])) for row in reader]


def coerce_binary_label(x):
    """Coerce a label (bool/int/float/str) to {0, 1}; raise on anything ambiguous."""
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, float) and x in (0.0, 1.0):
        return int(x)
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y", "positive", "readmitted", "readmission"}:
        return 1
    if s in {"0", "false", "no", "n", "negative", "not readmitted", "no readmission"}:
        return 0
    raise ValueError(f"Cannot coerce label: {x!r}")


def build_handoff(split, task_dir, force=False):
    """Join ``{split}_profile.jsonl`` + ``{split}_labels.csv`` -> ``{split}.jsonl``.

    Each output row is ``{id, profile, label}``. Returns the output path. Skips the
    rebuild if the file already exists and ``force`` is False.
    """
    task_dir = Path(task_dir)
    out_path = task_dir / f"{split}.jsonl"
    if out_path.exists() and not force:
        print(f"{split}: keeping existing {out_path}")
        return out_path
    profiles = read_jsonl(task_dir / f"{split}_profile.jsonl")
    labels = read_label_csv(task_dir / f"{split}_labels.csv")
    if len(profiles) != len(labels):
        raise ValueError(f"{split}: profile/label mismatch {len(profiles)} != {len(labels)}")
    with open(out_path, "w", encoding="utf-8") as f:
        for idx, (row, label) in enumerate(zip(profiles, labels)):
            text = str(row.get("profile", ""))
            if _FORBIDDEN.search(text):
                raise ValueError(f"{split}: forbidden DNR/RNS/code-status text remains at row {idx}")
            f.write(json.dumps({"id": row.get("idx", idx), "profile": text, "label": int(label)}) + "\n")
    print(f"{split}: wrote {out_path} with label counts {dict(Counter(labels))}")
    return out_path


def load_split(jsonl_path, limit=None):
    """Load a handoff JSONL -> ``(profiles, labels, ids)`` with labels as an int array."""
    rows = read_jsonl(jsonl_path)
    if limit is not None:
        rows = rows[:limit]
    profiles = [str(r["profile"]) for r in rows]
    labels = np.array([coerce_binary_label(r["label"]) for r in rows], dtype=int)
    ids = [str(r.get("id", i)) for i, r in enumerate(rows)]
    return profiles, labels, ids


def prepare_split(split, task_dir, force=False, limit=None):
    """``build_handoff`` + ``load_split`` in one call.

    Returns ``(profiles, labels, ids, jsonl_path)``.
    """
    path = build_handoff(split, task_dir, force=force)
    profiles, labels, ids = load_split(path, limit=limit)
    return profiles, labels, ids, path


def report_splits(splits, title="PHASE 0  |  Load data + class balance",
                  pos_name="positive", neg_name="negative"):
    """Print a banner with per-split counts and the positive rate.

    ``splits`` is an ordered dict like ``{'Train': y_train, 'Test': y_test}`` where each
    value is an array/list of 0/1 labels. ``pos_name`` / ``neg_name`` name the two outcome
    classes in the printed lines (e.g. 'long stay' / 'not long stay').
    """
    bar = "=" * 60
    print(bar)
    print(title)
    print(bar)
    for name, y in splits.items():
        y = np.asarray(list(y), dtype=int)
        n = len(y)
        if not n:
            print(f"{name:>6}: (empty)")
            continue
        pos = int(y.sum())
        print(f"{name:>6}: n={n:<6}  {pos_name}={pos:<5}  {neg_name}={n - pos:<6}  pos rate={pos / n:.3f}")
    summary = " | ".join(
        f"{name} pos rate {np.asarray(list(y), dtype=int).mean():.3f}" for name, y in splits.items() if len(list(y))
    )
    print("  " + summary)


def stratified_subsample(profiles, y, ids, n, seed):
    """Seeded subsample of ``n`` rows that keeps the label rate; ``n=None`` keeps every row.

    Returns ``(profiles, labels, ids)`` in the original row order.
    """
    if n is None or n >= len(y):
        return profiles, y, ids
    rng = np.random.default_rng(seed)
    take = []
    for cls in np.unique(y):
        cls_idx = np.flatnonzero(y == cls)
        k = int(round(n * len(cls_idx) / len(y)))
        take.extend(rng.choice(cls_idx, size=min(k, len(cls_idx)), replace=False))
    if len(take) > n:
        take = list(rng.choice(take, size=n, replace=False))
    elif len(take) < n:
        rest = np.setdiff1d(np.arange(len(y)), take)
        take.extend(rng.choice(rest, size=n - len(take), replace=False))
    take = sorted(int(i) for i in take)
    return [profiles[i] for i in take], y[take], [ids[i] for i in take]


# --- small JSON helpers used by the runner notebooks ----------------------------------------
def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def read_json(path, default):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def json_safe(value):
    """``value`` with numpy numbers and arrays turned into plain Python, for ``json.dumps``."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
