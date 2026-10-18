"""Phase 2E: keep only hypotheses that hold up on the confirmation split.

The confirmation split is the part of the training set that Phase 2B never used
(``self_refine.discovery_confirmation_split`` with the same seed and fraction). No LLM call
happens here: the input is the 0/1 annotation matrix of the hypotheses on those profiles.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from evaluation import compute_hypothesis_association_tests


def confirmation_gate(
    annotation_matrix: np.ndarray,
    labels: Sequence,
    hypotheses: List[str],
    *,
    fdr_alpha: float,
    n_bootstrap: int,
    stability_threshold: float,
    estimability_floor: int,
    min_survivors: int,
    top_k: Optional[int],
    seed: int,
    verbose: bool = True,
) -> Tuple[List[str], pd.DataFrame, dict]:
    """Significance and stability gate.

    A hypothesis is certified when all three hold:
      - it is associated with the label at Benjamini-Hochberg FDR level ``fdr_alpha``;
      - it stays FDR-significant with the same sign in at least ``stability_threshold`` of
        ``n_bootstrap`` resamples of the confirmation profiles;
      - it is true for at least ``estimability_floor`` confirmation profiles.

    After that: with ``top_k`` set, the kept set is filled up to ``top_k`` by lowest q-value;
    if fewer than ``min_survivors`` are kept, the largest |log odds ratio| ones are added;
    if nothing is kept, every hypothesis is kept so the run can proceed.

    ``verbose`` prints one line per hypothesis (kept or dropped, significance, stability).

    Returns:
        kept: hypotheses that pass, in the order they were kept
        tests: association test table on the full confirmation set
        summary: the rule, counts, the dropped hypotheses and each hypothesis' stability
    """
    gate_M = np.asarray(annotation_matrix)
    gate_y = np.asarray(labels, dtype=int)
    n_gate = len(gate_y)

    # Full-confirmation FDR association (Fisher / z-test + Benjamini-Hochberg).
    _, gate_tests = compute_hypothesis_association_tests(
        annotation_matrix=gate_M, labels=gate_y, hypotheses=hypotheses, fdr_alpha=fdr_alpha)
    full = gate_tests.set_index('hypothesis')
    full_sig = {h: bool(full.loc[h, 'significant_fdr']) for h in hypotheses}
    full_sign = {h: float(np.sign(full.loc[h, 'log_odds_ratio'])) for h in hypotheses}

    # Bootstrap stability: fraction of confirmation resamples where each hypothesis stays
    # FDR-significant with the SAME direction as the full sample.
    rng = np.random.default_rng(seed)
    hits = {h: 0 for h in hypotheses}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_gate, size=n_gate)
        _, bt = compute_hypothesis_association_tests(
            annotation_matrix=gate_M[idx], labels=gate_y[idx], hypotheses=hypotheses, fdr_alpha=fdr_alpha)
        bt = bt.set_index('hypothesis')
        for h in hypotheses:
            if bool(bt.loc[h, 'significant_fdr']) and np.sign(bt.loc[h, 'log_odds_ratio']) == full_sign[h]:
                hits[h] += 1
    stability = {h: hits[h] / n_bootstrap for h in hypotheses}

    fire = {h: int(gate_M[:, k].sum()) for k, h in enumerate(hypotheses)}
    keep_mask = [full_sig[h] and stability[h] >= stability_threshold and fire[h] >= estimability_floor
                 for h in hypotheses]
    kept = [h for h, k in zip(hypotheses, keep_mask) if k]
    dropped = [h for h, k in zip(hypotheses, keep_mask) if not k]
    n_certified = len(kept)
    if top_k and len(kept) < top_k:
        # Fixed final-set budget, ranked on CONFIRMATION evidence only (never touches test).
        # Certified hypotheses are kept first; the rest backfill by q so the set size is a
        # pre-declared protocol constant rather than "however many happened to survive".
        _pool = [h for h in hypotheses if fire[h] >= estimability_floor and h not in kept]
        _pool.sort(key=lambda h: float(full.loc[h, 'q_value_fdr']))
        _backfill = _pool[:top_k - len(kept)]
        if _backfill:
            print(f'  [top-K budget] {len(kept)} certified; backfilling {len(_backfill)} '
                  f'by confirmation q to top_k={top_k}.')
        kept = kept + _backfill
        keep_mask = [h in kept for h in hypotheses]
        dropped = [h for h in hypotheses if h not in kept]

    if verbose:
        print(f'\nPhase 2E gate: {len(hypotheses)} -> {len(kept)} kept '
              f'(FDR alpha={fdr_alpha}, stability>={stability_threshold:.0%} over {n_bootstrap} bootstraps)')
        for h in hypotheses:
            flag = 'KEEP' if h in kept else 'drop'
            print(f"  [{flag}] sig={full_sig[h]!s:>5} stability={stability[h]:.2f} true for={fire[h]:>4}  {h}")

    if len(kept) < min_survivors:
        _pool = [h for h in hypotheses if fire[h] >= estimability_floor and h not in kept]
        _pool.sort(key=lambda h: abs(float(full.loc[h, 'log_odds_ratio'])), reverse=True)
        _extra = _pool[:max(0, min_survivors - len(kept))]
        if _extra:
            print(f'  [safety floor] only {len(kept)} passed the gate; adding {len(_extra)} '
                  f'provisional (top-|effect|) to reach min_survivors={min_survivors}.')
        kept = kept + _extra
        keep_mask = [h in kept for h in hypotheses]
        dropped = [h for h in hypotheses if h not in kept]
    if not kept:
        print('  [warn] gate would drop everything; keeping the full set so the run can proceed.')
        kept = list(hypotheses)
        keep_mask = [True] * len(hypotheses); dropped = []

    summary = {
        'enabled': True, 'n_confirmation_annotated': int(n_gate), 'fdr_alpha': float(fdr_alpha),
        'bootstrap': int(n_bootstrap), 'stability_threshold': float(stability_threshold),
        'rule': 'fdr_and_stability_and_fire',
        'selection': ('keep if q_value_fdr <= fdr_alpha AND bootstrap_stability >= stability_threshold '
                      'AND n_present >= estimability_floor'),
        'n_in': len(keep_mask), 'n_kept': len(kept), 'n_dropped': len(dropped),
        'n_certified': n_certified, 'top_k': top_k,
        'dropped': dropped, 'stability': {h: round(stability[h], 3) for h in stability},
    }
    return kept, gate_tests, summary
