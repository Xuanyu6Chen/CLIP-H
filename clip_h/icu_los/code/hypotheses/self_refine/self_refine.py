"""Self-Refine control loop for CLIP-H hypothesis sets.

Implements the Self-Refine paradigm (Madaan et al., 2023, arXiv:2303.17651):
*generate -> self-feedback -> refine*, looped, with a **frozen LLM**. The feedback
here is **data-driven** rather than free-form self-critique: it is measured on a
DISCOVERY split of the training data (how often each hypothesis is true), together
with a clinical critic, so the loop improves the hypothesis SET before the clinical
review step.

Statistical guard (paper-faithful adaptation): all feedback reads only the discovery
split; any held-out confirmation / test split is never touched here, so the
multiple-testing penalty is paid once downstream, not once per iteration.

Contents: `discovery_confirmation_split`, then the Phase 2B loop `agentic_refine` with its
parts `make_feedback`, `make_critic_fn` and `make_refine_fn_agentic`. LLM calls go through
`llm_client.get_completion`; the prompts are in `prompts/self_refine.txt`.
"""
from __future__ import annotations

import json
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from prompts import load_prompt


# =============================================================================
# Discovery / confirmation split
# =============================================================================
def discovery_confirmation_split(n: int, frac: float = 0.7, seed: int = 0,
                                 labels: Optional[Sequence] = None):
    """Boolean masks partitioning `n` training rows into discovery (`frac`) and
    confirmation (`1 - frac`). Disjoint and exhaustive.

    With ``labels`` the split is stratified: each class is split at ``frac``, so both parts
    keep the outcome rate. Without it the split is one random permutation of all rows.
    """
    rng = np.random.default_rng(seed)
    disc = np.zeros(n, dtype=bool)
    conf = np.zeros(n, dtype=bool)
    if labels is None:
        order = rng.permutation(n)
        k = int(round(frac * n))
        disc[order[:k]] = True
        conf[order[k:]] = True
        return disc, conf
    labels = np.asarray(labels)
    assert len(labels) == n, "labels length must match n"
    for cls in np.unique(labels):
        cls_idx = rng.permutation(np.where(labels == cls)[0])
        k = int(round(frac * len(cls_idx)))
        disc[cls_idx[:k]] = True
        conf[cls_idx[k:]] = True
    return disc, conf


# =============================================================================
# Phase 2B: critic-in-the-loop refinement
# (adapted from `clip_h/mortality/mimic_eval/self_refine.py` in this repository)
#
# `agentic_refine` combines two kinds of feedback: a "too rare, broaden it" flag from
# `make_feedback` (computed from the annotations only) and the actions of a clinical
# `critic_fn` (accept / sharpen / split / reground / drop). Merging duplicates and
# testing against the outcome are left to Phase 2D and Phase 2E, so this step only
# improves the wording. The refine prompt keeps the "Patient has/is/was X" format.
# =============================================================================
PARSE_TRIES = 3   # how often a refine or critic reply that cannot be used is asked for again
REFINE_SYSTEM = load_prompt('self_refine', 'refine_system')
CRITIC_LOOP_SYSTEM = load_prompt('self_refine', 'critic_system')


def _parse_list(s: str) -> List[str]:
    s = re.sub(r"^```(?:json)?|```$", "", str(s).strip(), flags=re.MULTILINE).strip()
    try:
        m = re.search(r"\{.*\}|\[.*\]", s, re.DOTALL)
        obj = json.loads(m.group(0) if m else s)
    except Exception:
        return []
    if isinstance(obj, dict):
        obj = obj.get("hypotheses") or obj.get("revised") or next(iter(obj.values()), [])
    out = []
    for it in obj if isinstance(obj, list) else []:
        out.append(str(it.get("hypothesis", it) if isinstance(it, dict) else it).strip())
    return [h for h in out if h]


def make_feedback(hyps: Sequence[str], ann: Dict[str, Sequence], min_prev: float = 0.02
                  ) -> Tuple[str, Dict[str, object]]:
    """Emit prevalence-driven, actionable feedback for the refine loop (the Self-Refine "feedback"
    step). This is a **quality** signal only: its inputs are the hypotheses and their annotations,
    never the outcome label.

    Per hypothesis it flags NICHE (prevalence < ``min_prev``) and asks the generator to BROADEN it to
    a more prevalent range on the SAME clinical dimension (staying faithful to the evidence) rather
    than dropping it. Significance (a function of the outcome) is deliberately NOT tested here -- that
    is the gate's job on held-out data; and redundancy is left to the Phase 2D dedup.

    Returns the feedback text + a summary dict. ``n_niche`` counts hypotheses flagged for broaden.
    """
    n = len(next(iter(ann.values()))) if ann else 0
    hyps = list(hyps)
    recs = []
    for h in hyps:
        x = np.asarray(ann.get(h, np.zeros(n)), dtype=float)
        prev = float(np.mean(x != 0)) if len(x) else 0.0
        recs.append({"hypothesis": h, "prevalence": prev, "niche": prev < min_prev})

    lines = []
    for r in recs:
        if r["niche"]:
            lines.append(
                f'- "{r["hypothesis"]}": NICHE (fires in only {r["prevalence"]*100:.1f}% of patients '
                "-- broaden it to a more prevalent range on the SAME clinical dimension, staying "
                "faithful to the evidence; do not jump to a coarser umbrella category. If it cannot be "
                "faithfully broadened, leave it unchanged.)")

    n_flagged = len(lines)
    summary = {"n": len(hyps), "n_niche": sum(r["niche"] for r in recs), "n_flagged": n_flagged,
               "records": recs}
    feedback = ("All hypotheses are sufficiently prevalent; no broadening needed."
                if n_flagged == 0 else
                "Some concepts are too rare to estimate reliably -- broaden them faithfully:\n"
                + "\n".join(lines))
    return feedback, summary


def make_refine_fn_agentic(model: str, complete_fn: Callable = None, timeout: int = 300, seed: int = 0):
    """Return refine_fn(hyps, feedback) -> revised hyps, backed by the LLM. Each revised hypothesis
    must stay a single, generalizable, data-grounded clinical predicate (no invented specifics).
    Without a usable reply in ``PARSE_TRIES`` tries it returns None."""
    if complete_fn is None:
        from llm_client import get_completion as complete_fn

    def refine_fn(hyps: Sequence[str], feedback: str) -> List[str]:
        prompt = load_prompt('self_refine', 'refine').format(
            hypotheses="\n".join(f"- {h}" for h in hyps), feedback=feedback)
        problem = ""
        for attempt in range(PARSE_TRIES):   # a reply that cannot be used is asked for again
            try:
                raw = complete_fn(prompt=prompt, model=model, system=REFINE_SYSTEM,
                                  max_tokens=4000, timeout=timeout, seed=seed, attempt=attempt)
                out = _parse_list(raw)
                if out:
                    return out
                problem = "the reply held no hypothesis list"
            except Exception as exc:
                problem = f"the call failed ({type(exc).__name__})"
        print(f"  [self-refine] refine: {problem} in {PARSE_TRIES} tries; hypotheses left unchanged")
        return None
    return refine_fn


def make_critic_fn(model: str, task: str = "", pos_label: str = "the positive class",
                   complete_fn: Callable = None, timeout: int = 300, seed: int = 0):
    """Return ``critic_fn(hyps, prevalence) -> {hyp: {action, feedback}}`` backed by one LLM call
    (asked again up to ``PARSE_TRIES`` times when the reply cannot be read). The critic sees each
    hypothesis with its prevalence on the discovery sample and never the outcome. ``action`` in
    {accept, sharpen, split, reground, drop}: sharpen a vague/weak concept, split a conflated one,
    re-ground a vague/non-observable concept as a concrete feature, drop a leaky/tautological/
    irredeemable one, else accept. Demographic attributes are valid features and are never
    penalised for being demographic. Without a readable reply it returns None and
    ``agentic_refine`` counts every hypothesis as accepted.
    ``complete_fn`` replaces the LLM call (for a run without the API)."""
    if complete_fn is None:
        from llm_client import get_completion as complete_fn

    def critic_fn(hyps: Sequence[str], prevalence: Dict[str, float]) -> Dict[str, Dict]:
        numbered = "\n".join(
            f'{i}: "{h}"  [prevalence={prevalence.get(h, float("nan"))*100:.0f}%]'
            for i, h in enumerate(hyps))
        prompt = load_prompt('self_refine', 'critic').format(task=task, pos_label=pos_label, candidates=numbered)
        reviews, problem = None, ""
        for attempt in range(PARSE_TRIES):   # a reply that cannot be read is asked for again
            try:
                raw = complete_fn(prompt=prompt, model=model, system=CRITIC_LOOP_SYSTEM,
                                  max_tokens=3000, timeout=timeout, seed=seed, attempt=attempt)
                s = re.sub(r"^```(?:json)?|```$", "", str(raw).strip(), flags=re.MULTILINE).strip()
                m = re.search(r"\{.*\}", s, re.DOTALL)
                reviews = json.loads(m.group(0) if m else s).get("reviews", [])
                break
            except Exception as exc:
                problem = type(exc).__name__
        if reviews is None:
            print(f"  [self-refine] critic: no readable reply in {PARSE_TRIES} tries ({problem}); "
                  "every hypothesis counted as accepted")
            return None
        out = {h: {"action": "accept", "feedback": ""} for h in hyps}
        for r in reviews:
            try:
                i = int(r["id"])
            except Exception:
                continue
            if 0 <= i < len(hyps):
                out[hyps[i]] = {"action": str(r.get("action", "accept")).lower().strip(),
                                "feedback": str(r.get("reason", "")).strip()}
        return out
    return critic_fn


def agentic_refine(hyps: Sequence[str], n_profiles: int,
                   annotate_fn: Callable[[List[str]], Dict[str, Sequence]],
                   refine_fn: Callable[[Sequence[str], str], List[str]],
                   critic_fn: Optional[Callable] = None,
                   iters: int = 3, min_prev: float = 0.02) -> Tuple[List[str], List[dict]]:
    """Critic-in-the-loop refinement on the DISCOVERY sample -- a QUALITY step that takes no outcome
    label: its inputs are the hypotheses, the number of sample profiles (``n_profiles``) and
    ``annotate_fn``, which returns the 0/1 annotations of a list of hypotheses on those profiles.
    Each round combines TWO feedback sources: (i) a too-rare flag from ``make_feedback`` that asks the
    generator to BROADEN the concept faithfully, and (ii) a clinical ``critic_fn`` returning
    per-hypothesis actions (sharpen / split / re-ground / drop). Critic 'drop's are applied directly;
    sharpen/split/reground/broaden become guidance for ``refine_fn`` (the generator). Redundancy and
    significance are handled downstream (Phase 2D dedup and the Phase 2E gate), not here. Stops when
    nothing is flagged and the critic asks for nothing, when the refiner changes nothing, or after
    ``iters`` rounds. Never touches confirmation/test data.
    Returns (refined_hyps, trace); ``trace`` has one entry per round with each hypothesis'
    prevalence and critic action. A round whose critic or refiner gave no usable reply (the
    function returned None or raised) goes on without it and is marked ``critic_failed`` or
    ``refine_failed``, so that it cannot be read as a round in which nothing needed changing."""
    hyps = list(dict.fromkeys(h for h in hyps if str(h).strip()))
    ann = annotate_fn(hyps)
    trace = []
    for t in range(iters):
        fb_data, summary = make_feedback(hyps, ann, min_prev=min_prev)
        actions, crit_lines, critic_failed = {}, [], False
        if critic_fn is not None:
            print(f"  [self-refine] round {t + 1} of {iters}: critic reviewing {len(hyps)} hypotheses...", flush=True)
            prev = {h: float(np.mean(np.asarray(ann.get(h, np.zeros(n_profiles)), dtype=float) != 0))
                    for h in hyps}
            try:
                actions = critic_fn(hyps, prev)
            except Exception as exc:
                print(f"WARNING: self-refine critic failed ({type(exc).__name__}: {exc}); "
                      "this round runs without critic actions", flush=True)
                actions = None
            critic_failed = actions is None
            actions = actions or {}
            for h in hyps:
                a = actions.get(h, {})
                act = str(a.get("action", "accept")).lower()
                if act in ("sharpen", "split", "reground") and a.get("feedback"):
                    crit_lines.append(f'- "{h}": {act.upper()} -- {a["feedback"]}')
        dropped = [h for h in hyps if str(actions.get(h, {}).get("action", "")).lower() == "drop"]
        kept = [h for h in hyps if h not in dropped]
        n_actions = summary["n_flagged"] + len(crit_lines) + len(dropped)
        _recs = {r["hypothesis"]: r for r in summary["records"]}
        iter_actions = [{"hypothesis": h,
                         "prevalence": _recs.get(h, {}).get("prevalence"),
                         "niche_broaden": bool(_recs.get(h, {}).get("niche", False)),
                         "critic_action": str(actions.get(h, {}).get("action", "accept")).lower(),
                         "critic_feedback": str(actions.get(h, {}).get("feedback", ""))}
                        for h in hyps]
        trace.append({"iter": t, "n": len(hyps), "n_flagged": summary["n_flagged"],
                      "n_niche": summary["n_niche"], "n_broadened": summary["n_niche"],
                      "n_dropped": len(dropped), "n_critic_edits": len(crit_lines),
                      "critic_failed": critic_failed, "refine_failed": False,
                      "actions": iter_actions})
        if n_actions == 0:
            break
        feedback = fb_data
        if crit_lines:
            feedback += "\n\nClinical-critic guidance:\n" + "\n".join(crit_lines)
        if dropped:
            feedback += "\n\nDrop these entirely: " + "; ".join(f'"{h}"' for h in dropped)
        print(f"  [self-refine] round {t + 1} of {iters}: {summary['n_niche']} too rare, {len(crit_lines)} to "
              f"edit, {len(dropped)} to drop; refiner rewriting {len(kept)} hypotheses...", flush=True)
        revised = refine_fn(kept, feedback)
        if revised is None:
            trace[-1]["refine_failed"] = True
            revised = kept
        new = list(dict.fromkeys(h for h in revised if str(h).strip())) or kept
        if set(new) == set(hyps):                 # generator proposed no textual change
            if dropped:                           # but drops still shrink the set
                hyps, ann = kept, annotate_fn(kept)
            break
        hyps, ann = new, annotate_fn(new)
    return hyps, trace
