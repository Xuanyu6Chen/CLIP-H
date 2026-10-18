"""Phase 2A (end): drop low-fidelity hypotheses and exact duplicates from the pool."""

import re
from typing import Tuple

import numpy as np
import pandas as pd


def norm_hypothesis(x) -> str:
    """Lower-case, letters and digits only, single spaces: the form used to spot exact duplicates."""
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9]+', ' ', str(x).lower())).strip()


def best_sort_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by fidelity score, then by |selection score|, then by original order."""
    score_cols = [c for c in df.columns if c.endswith('_fidelity_score')]
    target_cols = [c for c in df.columns if c.startswith('target_')]
    out = df.copy()
    out['_rank_score'] = out[score_cols[0]].fillna(-np.inf) if score_cols else 0.0
    out['_rank_target'] = out[target_cols[0]].abs().fillna(0.0) if target_cols else 0.0
    return out.sort_values(['_rank_score', '_rank_target', '_generation_rank'], ascending=[False, False, True])


def curate(pool_df: pd.DataFrame, min_fidelity: float) -> Tuple[pd.DataFrame, dict]:
    """Fidelity gate, then exact-duplicate removal.

    Drops hypotheses whose fidelity score is below ``min_fidelity``. Grouped weak-neuron
    hypotheses have no fidelity score and are always kept; if the gate would drop every row
    it is skipped. Then keeps the best-ranked row of each set of identical texts.

    Returns (curated table sorted best first, summary counts).
    """
    out = pool_df.copy().reset_index(drop=True)
    out['_generation_rank'] = range(len(out))
    before = len(out)
    _fcol = next((c for c in out.columns if c.endswith('_fidelity_score')), None)
    dropped_fidelity = 0
    if _fcol is not None and min_fidelity and min_fidelity > 0:
        _fid = pd.to_numeric(out[_fcol], errors='coerce')
        _keep = _fid.isna() | (_fid >= min_fidelity)
        if int(_keep.sum()) >= 1:
            dropped_fidelity = int((~_keep).sum())
            out = out[_keep.values].reset_index(drop=True)
    # Exact-duplicate removal (normalized text), keeping the best-ranked per duplicate.
    ranked = best_sort_columns(out)
    ranked['_norm'] = ranked['interpretation'].map(norm_hypothesis)
    out = ranked.drop_duplicates('_norm', keep='first').sort_values('_generation_rank').reset_index(drop=True)
    exact = before - dropped_fidelity - len(out)
    curated = best_sort_columns(out.copy()).drop(columns=['_rank_score', '_rank_target', '_norm'], errors='ignore')
    summary = {
        'pool_size': int(before),
        'min_fidelity': float(min_fidelity),
        'dropped_low_fidelity': int(dropped_fidelity),
        'exact_merged': int(exact),
        'curated_candidate_count': int(len(curated)),
    }
    return curated, summary
