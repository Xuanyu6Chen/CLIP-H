"""Phase 2D: merge hypotheses that are true for nearly the same patients.

    phi_matrix               pairwise phi (correlation of two 0/1 columns)
    connected_components     groups of hypotheses linked by merge decisions
    predictive_dedup         the merge rule: automatic above ``auto_thr``, LLM-confirmed below it
    make_dedup_adjudicator   the LLM call that confirms or refuses a borderline merge
    log_stage_table          writes a list of rows as a CSV

Adapted from ``clip_h/mortality/mimic_eval/clip_h_lib.py`` in this repository. The prompt is in
``prompts/dedup.txt``.
"""
from __future__ import annotations

import json
import os
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from prompts import load_prompt


def log_stage_table(run_dir: str, name: str, rows: Sequence[Dict]) -> str:
    """Write ``rows`` (a list of dicts) as ``<run_dir>/<name>.csv``. Returns the path."""
    fname = name if str(name).endswith(".csv") else f"{name}.csv"
    path = os.path.join(run_dir, fname)
    pd.DataFrame(list(rows)).to_csv(path, index=False)
    return path


def phi_matrix(vectors: Sequence) -> "np.ndarray":
    """Pairwise phi coefficient (Pearson correlation on binary indicator columns) for a
    set of concept firing vectors. ``vectors`` is ``[n_concepts, n_patients]`` (0/1).
    Returns an ``[n, n]`` matrix in [-1, 1]; a constant (never/always firing) concept
    correlates 0 with everything, diagonal is 1. Pure/deterministic."""
    X = np.asarray([np.asarray(v, dtype=float) for v in vectors], dtype=float)
    n = X.shape[0]
    if n == 0:
        return np.zeros((0, 0))
    Xb = (X != 0).astype(float)
    Xc = Xb - Xb.mean(axis=1, keepdims=True)
    norm = np.sqrt((Xc ** 2).sum(axis=1))
    denom = np.outer(norm, norm)
    cov = Xc @ Xc.T
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = np.where(denom > 0, cov / denom, 0.0)
    np.fill_diagonal(phi, 1.0)
    return phi


def connected_components(n: int, edges: Sequence) -> List[List[int]]:
    """Union-find: partition ``range(n)`` into connected components from undirected
    ``edges`` (index pairs). Returns a list of sorted index-groups (singletons
    included). Pure."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        a, b = int(a), int(b)
        if 0 <= a < n and 0 <= b < n:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [sorted(g) for g in groups.values()]


def predictive_dedup(vectors: Sequence, labels: Sequence[str],
                     directions: Sequence[str], meta: Optional[Sequence[Dict]] = None,
                     corr_thr: float = 0.90, auto_thr: float = 0.95,
                     adjudicate_fn: Optional[Callable] = None) -> Dict:
    """The single firing-vector redundancy merge. Two concepts are merged only when they
    fire on nearly the same patients AND share a non-unclear ``expected_direction``:

        phi >= auto_thr            -> auto-merge (near-certain statistical duplicate)
        corr_thr <= phi < auto_thr -> ``adjudicate_fn`` may CONFIRM the merge (else keep apart)
        phi <  corr_thr            -> keep separate (distinct firing, never touched)

    Opposite/unclear-direction pairs never merge. ``adjudicate_fn(a_label, b_label, a_meta,
    b_meta, stats) -> (decision, reason)`` returns ``'merge'``/``'keep_separate'``; it can only
    VETO the borderline band (statistics proposes, the LLM refuses). With ``adjudicate_fn=None``
    or on any error the borderline band defaults to keep_separate (conservative). Merges are
    transitive: A-B and B-C put A, B and C in one cluster. Each merged cluster keeps a single
    representative: the highest ``clinical_validity`` in ``meta``, then the one true for the
    fewest patients, then the shortest text.

    Returns a dict with ``source_map`` ({rep_label: [member label texts incl. rep]} for merged
    clusters only), ``kept`` (final representative/atomic labels in original order), ``pairs``
    (every phi>=corr_thr pair with its tier/decision, for logging), and ``clusters`` (merged
    groups with per-member phi-to-rep). Pure/deterministic given ``adjudicate_fn``.
    """
    labels = [str(x) for x in labels]
    n = len(labels)
    meta = list(meta) if meta is not None else [{} for _ in range(n)]
    prevalence = [float(np.mean(np.asarray(v, dtype=float) != 0)) if len(np.asarray(v)) else 0.0
                  for v in vectors]
    fires = [int(np.sum(np.asarray(v, dtype=float) != 0)) for v in vectors]
    bin_vecs = [np.asarray(v, dtype=float) != 0 for v in vectors]

    def _dir(i):
        d = str(directions[i]).strip().lower() if i < len(directions) else "unclear"
        return d if d not in ("", "unclear", "nan", "none") else None

    P = phi_matrix(vectors) if n else np.zeros((0, 0))
    edges: List[Tuple[int, int]] = []
    pairs: List[Dict] = []
    for i in range(n):
        for j in range(i + 1, n):
            phi = float(P[i, j])
            if phi < corr_thr:
                continue
            same_dir = _dir(i) is not None and _dir(i) == _dir(j)
            stats = {"phi": phi, "cofire_a": fires[i], "cofire_b": fires[j],
                     "cofire_both": int(np.sum(bin_vecs[i] & bin_vecs[j])),
                     "direction": _dir(i) or "unclear"}
            decision, reason = "keep_separate", ""
            if not same_dir:
                tier = "direction_blocked"
            elif phi >= auto_thr:
                tier, decision, reason = "auto", "merge", "phi >= auto_thr"
                edges.append((i, j))
            else:
                if adjudicate_fn is not None:
                    try:
                        decision, reason = adjudicate_fn(labels[i], labels[j], meta[i], meta[j], stats)
                        decision = str(decision).strip().lower()
                    except Exception as exc:
                        print(f"WARNING: dedup confirm call failed ({type(exc).__name__}); pair kept separate", flush=True)
                        decision, reason = "keep_separate", "adjudicator error -> keep"
                    print(f"  [dedup] phi={phi:.2f}, LLM says {decision}: \"{labels[i][:60]}\" vs \"{labels[j][:60]}\"", flush=True)
                else:
                    reason = "no adjudicator -> keep"
                if decision == "merge":
                    tier = "llm_confirmed"; edges.append((i, j))
                else:
                    tier, decision = "llm_vetoed_kept", "keep_separate"
            pairs.append({"concept_a": labels[i], "concept_b": labels[j], "phi": round(phi, 4),
                          "cofire_a": fires[i], "cofire_b": fires[j], "cofire_both": stats["cofire_both"],
                          "same_direction": bool(same_dir), "tier": tier,
                          "llm_decision": decision, "llm_reason": reason})

    comps = connected_components(n, edges)
    comp_of = {i: c for c in range(len(comps)) for i in comps[c]}

    def _rep(idxs: List[int]) -> int:
        def key(i):
            v = float(meta[i].get("clinical_validity", 2)) if isinstance(meta[i], dict) else 2.0
            return (-v, prevalence[i], len(labels[i]), labels[i])
        return min(idxs, key=key)

    source_map: Dict[str, List[str]] = {}
    clusters: List[Dict] = []
    rep_label_of_comp: Dict[int, str] = {}
    for c, idxs in enumerate(comps):
        rep = _rep(idxs)
        rep_label_of_comp[c] = labels[rep]
        if len(idxs) > 1:
            members = [labels[rep]] + [labels[k] for k in idxs if k != rep]
            source_map[labels[rep]] = members
            clusters.append({"representative": labels[rep], "members": members,
                             "phi_to_rep": {labels[k]: round(float(P[rep, k]), 4)
                                            for k in idxs if k != rep}})

    seen: set = set()
    kept: List[str] = []
    for i in range(n):
        rl = rep_label_of_comp[comp_of[i]]
        if rl not in seen:
            seen.add(rl); kept.append(rl)
    return {"source_map": source_map, "kept": kept, "pairs": pairs, "clusters": clusters}


PARSE_TRIES = 3   # how often a confirm reply that cannot be read is asked for again
DEDUP_CONFIRMER_SYSTEM = load_prompt('dedup', 'system')


def make_dedup_adjudicator(model: str, task: str = "", pos_label: str = "the positive class",
                           complete_fn: Callable = None, timeout: int = 120, seed: int = 0):
    """Return ``adjudicate_fn(a, b, a_meta, b_meta, stats) -> (decision, reason)`` backed by one
    LLM call that is shown how many patients each hypothesis and both are true for (asked again
    up to ``PARSE_TRIES`` times when the reply cannot be read). It may only CONFIRM a
    statistically-proposed merge: it answers 'merge' when the two are a paraphrase or a different
    granularity of one variable, and 'keep_separate' otherwise (distinct constructs that merely
    co-occur). Any error or unreadable reply -> keep_separate. ``complete_fn`` replaces the LLM
    call (for a run without the API)."""
    if complete_fn is None:
        from llm_client import get_completion as complete_fn

    def adjudicate_fn(a, b, a_meta, b_meta, stats):
        prompt = load_prompt('dedup', 'user').format(
            task=task, pos_label=pos_label, phi=f"{stats.get('phi', float('nan')):.2f}",
            concept_a=a, n_a=stats.get('cofire_a', 0), concept_b=b, n_b=stats.get('cofire_b', 0),
            n_both=stats.get('cofire_both', 0))
        problem = ""
        for attempt in range(PARSE_TRIES):   # a reply that cannot be read is asked for again
            try:
                raw = complete_fn(prompt=prompt, model=model, system=DEDUP_CONFIRMER_SYSTEM,
                                  max_tokens=200, timeout=timeout, seed=seed, attempt=attempt)
                s = re.sub(r"^```(?:json)?|```$", "", str(raw).strip(), flags=re.MULTILINE).strip()
                m = re.search(r"\{.*\}", s, re.DOTALL)
                obj = json.loads(m.group(0) if m else s)
                decision = str(obj.get("decision", "")).strip().lower()
                if decision in ("merge", "keep_separate"):
                    return decision, str(obj.get("reason", "")).strip()
                problem = "no valid decision in the reply"
            except Exception as exc:
                problem = type(exc).__name__
        print(f"  [dedup] confirm: no readable reply in {PARSE_TRIES} tries ({problem}); pair kept separate")
        return "keep_separate", "adjudicator error -> keep"
    return adjudicate_fn
