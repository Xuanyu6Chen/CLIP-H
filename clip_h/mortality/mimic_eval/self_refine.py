"""Self-refinement of CLIP-H hypotheses (Phase 2.2), following Self-Refine (Madaan et al., 2023,
arXiv:2303.17651).

A frozen LLM alternates feedback and refinement over the hypothesis set; nothing is trained. Each
round combines two feedback sources: a prevalence check that flags hypotheses too rare to estimate
reliably, and a clinical critic that sees each hypothesis and its prevalence but not its
association with the outcome. The loop never reads the outcome label.

Sample splitting: `discovery_confirmation_split` partitions the training split into a discovery
split (70% by default) and a disjoint confirmation split. Refinement (Phase 2.2) and deduplication
(Phase 2.4) use only the discovery split, and the significance gate (Phase 2.5) tests each
hypothesis once on the confirmation split. The multiple-testing correction is therefore applied a
single time, on patients that refinement and deduplication did not use, rather than compounding over
refinement rounds. The runner notebooks assert that the two masks are disjoint.

LLM and annotation calls are passed in as callables. The `make_*_fn` factories build them on
`hypothesaes.llm_api.get_completion`.
"""
from __future__ import annotations

import json
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import os
import time
import numpy as np

# Retry budget for the refine and critic calls (environment variables SR_TRIES and SR_WAIT_S).
# Each is a single large LLM call, and a failure is silent and biased toward stopping: refine
# returns its input unchanged and the critic returns 'accept' for every hypothesis, so
# agentic_refine sees no requested edits and stops as if it had converged. Retrying keeps a
# transient network error from ending refinement after the first round.
_SR_TRIES = int(os.environ.get("SR_TRIES", "3"))
_SR_WAIT_S = float(os.environ.get("SR_WAIT_S", "30"))
SR_FAILURES = []          # stage names ('refine', 'critic') whose call failed on every attempt


def _sr_call(fn, stage):
    """Call ``fn`` up to ``_SR_TRIES`` times, waiting ``_SR_WAIT_S`` seconds between attempts.

    Returns the first successful result. If every attempt raises, appends ``stage`` to
    ``SR_FAILURES`` and returns None.
    """
    for _t in range(_SR_TRIES):
        try:
            return fn()
        except Exception:
            if _t == _SR_TRIES - 1:
                SR_FAILURES.append(stage)
                return None
            time.sleep(_SR_WAIT_S)
    return None

# --------------------------------------------------------------------------- #
# Sample splitting
# --------------------------------------------------------------------------- #
def discovery_confirmation_split(n: int, frac: float = 0.7, seed: int = 42,
                                 labels: Optional[Sequence] = None
                                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Return disjoint boolean masks ``(discovery, confirmation)`` over ``n`` rows.

    Apply to the training split only. The discovery mask selects a ``frac`` share of the rows
    (rounded) and the confirmation mask selects the rest, so together they partition all rows.
    Refinement and deduplication read the discovery rows; the significance gate reads the
    confirmation rows once. The held-out test set is not involved. The split is reproducible for
    a given ``seed``.

    If ``labels`` is given, the split is stratified by label (each class is split at ``frac``), so
    both masks keep the outcome prevalence. This matters for imbalanced tasks such as in-hospital
    mortality, where an unstratified confirmation split can leave too few positives for a stable
    significance gate. With ``labels=None`` the split is a plain random permutation.
    """
    rng = np.random.default_rng(seed)
    disc = np.zeros(n, dtype=bool); conf = np.zeros(n, dtype=bool)
    if labels is None:
        idx = rng.permutation(n)
        k = int(round(frac * n))
        disc[idx[:k]] = True; conf[idx[k:]] = True
        return disc, conf
    labels = np.asarray(labels)
    assert len(labels) == n, "labels length must match n"
    for cls in np.unique(labels):
        cls_idx = rng.permutation(np.where(labels == cls)[0])
        k = int(round(frac * len(cls_idx)))
        disc[cls_idx[:k]] = True
        conf[cls_idx[k:]] = True
    return disc, conf


# --------------------------------------------------------------------------- #
# Prevalence feedback (computed on the discovery split)
# --------------------------------------------------------------------------- #
def make_feedback(hyps: Sequence[str], ann: Dict[str, Sequence], y: Optional[Sequence] = None,
                  embed_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
                  min_prev: float = 0.02, sig_alpha: float = 0.05, sim_thresh: float = 0.85
                  ) -> Tuple[str, Dict[str, object]]:
    """Build prevalence-driven feedback for the rewriter (the "feedback" step of Self-Refine).

    The feedback is a quality signal computed from the annotations ``ann`` alone; it never reads
    the outcome label. A hypothesis whose prevalence (share of rows where its annotation is
    nonzero) is below ``min_prev`` is flagged ``NICHE``, and the feedback asks the rewriter to
    broaden it to a more prevalent range on the same clinical dimension, staying faithful to the
    evidence, rather than drop it. Significance is not tested here because it depends on the
    outcome; the significance gate tests it on the confirmation split. Redundancy is handled by
    firing-vector deduplication (Phase 2.4).

    ``y``, ``embed_fn``, ``sig_alpha`` and ``sim_thresh`` are accepted for interface compatibility
    and ignored, except that ``len(y)`` sets the row count when ``ann`` is empty.

    Returns ``(feedback, summary)``: the feedback text and a dict with ``n`` (number of
    hypotheses), ``n_niche`` and ``n_flagged`` (hypotheses flagged for broadening), ``records``
    (per-hypothesis prevalence and niche flag), and ``n_insignificant`` and ``n_redundant_pairs``,
    which are always 0 and kept for interface compatibility.
    """
    n = len(next(iter(ann.values()))) if ann else (len(y) if y is not None else 0)
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
    summary = {"n": len(hyps), "n_niche": sum(r["niche"] for r in recs),
               "n_insignificant": 0, "n_redundant_pairs": 0, "n_flagged": n_flagged,
               "records": recs}
    feedback = ("All hypotheses are sufficiently prevalent; no broadening needed."
                if n_flagged == 0 else
                "Some concepts are too rare to estimate reliably -- broaden them faithfully:\n"
                + "\n".join(lines))
    return feedback, summary


# --------------------------------------------------------------------------- #
# JSON parsing and the LLM-backed rewriter (``complete_fn`` is injectable)
# --------------------------------------------------------------------------- #
def _parse_list(s: str) -> List[str]:
    """Parse an LLM reply into a list of hypothesis strings; return [] if no list can be parsed."""
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


REFINE_SYSTEM = (
    "You are a senior clinical-informatics scientist revising a set of machine-generated clinical "
    "hypotheses that are used as binary features to predict a hospital outcome. Respond with only "
    "valid JSON."
)


def make_refine_fn(model: str, complete_fn: Callable = None, timeout: int = 300, seed: int = 0):
    """Return ``refine_fn(hyps, feedback) -> list[str]``, the rewriter, backed by one LLM call.

    The prompt permits only sharpening, splitting, re-grounding, faithful broadening of concepts
    flagged ``NICHE``, and rewording, and asks that each hypothesis stay a single, generalizable,
    data-grounded clinical predicate with no invented specifics. If the call fails on every retry
    or the reply cannot be parsed, ``refine_fn`` returns the input hypotheses unchanged.
    ``complete_fn`` defaults to ``hypothesaes.llm_api.get_completion``.
    """
    if complete_fn is None:
        from hypothesaes.llm_api import get_completion as complete_fn  # noqa
    def refine_fn(hyps: Sequence[str], feedback: str) -> List[str]:
        prompt = (
            "Revise the hypothesis set below by applying the feedback (measured on held-out discovery "
            "data). Make only these kinds of edits: sharpen a vague concept, split a conflated one, "
            "re-ground a vague or not-directly-observable concept as a concrete checkable feature, "
            "broaden a concept flagged NICHE "
            "to a more prevalent range on the SAME clinical dimension (stay faithful to the evidence "
            "-- widening the same variable, never jumping to a coarser umbrella category), and reword "
            "for clarity. Keep every hypothesis a single, generalizable, falsifiable clinical "
            "predicate grounded in admission data. Do not invent specifics, do not combine several "
            "conditions into one, and do not merge two distinct concepts into a broader one -- "
            "redundancy is handled by a later step.\n\n"
            "<current_hypotheses>\n" + "\n".join(f"- {h}" for h in hyps) + "\n</current_hypotheses>\n\n"
            "<feedback>\n" + feedback + "\n</feedback>\n\n"
            'Return only valid JSON in this exact form: {"hypotheses": ["<concept>", ...]}.')
        try:
            raw = _sr_call(lambda: complete_fn(prompt=prompt, model=model, system=REFINE_SYSTEM,
                                               max_tokens=4000, timeout=timeout, seed=seed),
                           "refine")
            if raw is None:
                return list(hyps)          # every attempt failed; the stage is in SR_FAILURES
            out = _parse_list(raw)
            return out or list(hyps)
        except Exception:
            return list(hyps)
    return refine_fn


# --------------------------------------------------------------------------- #
# Critic-in-the-loop refinement (Phase 2.2)
# --------------------------------------------------------------------------- #
CRITIC_LOOP_SYSTEM = (
    "You are a senior clinical-informatics scientist auditing machine-generated clinical hypotheses "
    "that will become binary features for predicting a hospital outcome. You are shown each "
    "hypothesis and how often it is true (prevalence) but NOT its association with the outcome, so "
    "your judgment stays an honest a-priori one. For each hypothesis choose ONE action and give a "
    "one-line reason. Respond with only valid JSON."
)


def make_critic_fn(model: str, task: str = "", pos_label: str = "the positive class",
                   complete_fn: Callable = None, timeout: int = 300, seed: int = 0):
    """Return ``critic_fn(hyps, prevalence) -> {hyp: {"action": ..., "feedback": ...}}``, a
    clinical critic backed by one LLM call.

    The critic sees each hypothesis and its prevalence but not its association with the outcome.
    It chooses one action per hypothesis: ``sharpen`` (valid but vague or too narrow), ``split``
    (conflates two distinct clinical signals), ``reground`` (vague or not directly observable;
    restate as a concrete, checkable feature), ``drop`` (leaks or restates the outcome, or is
    clinically vacuous) or ``accept``, and ``feedback`` holds its one-line reason. Demographic
    attributes are treated as valid features and are not penalised for being demographic.
    Hypotheses the reply does not cover default to ``accept``; if the call fails on every retry
    or the reply cannot be parsed, every hypothesis is returned as ``accept``. ``complete_fn``
    defaults to ``hypothesaes.llm_api.get_completion``.
    """
    if complete_fn is None:
        from hypothesaes.llm_api import get_completion as complete_fn  # noqa

    def critic_fn(hyps: Sequence[str], prevalence: Dict[str, float]) -> Dict[str, Dict]:
        numbered = "\n".join(
            f'{i}: "{h}"  [prevalence={prevalence.get(h, float("nan"))*100:.0f}%]'
            for i, h in enumerate(hyps))
        prompt = (
            f"Audit these candidate hypotheses for predicting the {task} outcome (positive class: "
            f"{pos_label}). For each id choose action: 'drop' (leaks/restates the outcome, or is "
            "clinically vacuous), 'sharpen' (valid but vague or too narrow), 'split' (conflates two "
            "distinct clinical signals), 'reground' (too vague or not directly observable from the "
            "record -- restate it as a concrete, checkable feature), or 'accept'. Give a one-line "
            "reason. Demographic attributes (age, sex, race, insurance, language) are valid, "
            "first-class features: judge them on the same concreteness bar as any clinical variable "
            "and never drop, reground, or downweight a hypothesis merely for being demographic.\n\n"
            "<candidates>\n" + numbered + "\n</candidates>\n\n"
            'Return only JSON: {"reviews":[{"id":<int>,"action":"...","reason":"..."}, ...]}.')
        try:
            raw = _sr_call(lambda: complete_fn(prompt=prompt, model=model,
                                               system=CRITIC_LOOP_SYSTEM, max_tokens=3000,
                                               timeout=timeout, seed=seed), "critic")
            if raw is None:
                return {h: {"action": "accept", "feedback": ""} for h in hyps}
            s = re.sub(r"^```(?:json)?|```$", "", str(raw).strip(), flags=re.MULTILINE).strip()
            m = re.search(r"\{.*\}", s, re.DOTALL)
            reviews = json.loads(m.group(0) if m else s).get("reviews", [])
        except Exception:
            return {h: {"action": "accept", "feedback": ""} for h in hyps}
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


def agentic_refine(hyps: Sequence[str], y_disc: Sequence,
                   annotate_fn: Callable[[List[str]], Dict[str, Sequence]],
                   refine_fn: Callable[[Sequence[str], str], List[str]],
                   critic_fn: Optional[Callable] = None, embed_fn: Optional[Callable] = None,
                   iters: int = 3, min_prev: float = 0.02, sig_alpha: float = 0.05,
                   sim_thresh: float = 0.85) -> Tuple[List[str], List[dict]]:
    """Refine a hypothesis set on the discovery split with a critic in the loop (Phase 2.2).

    This is a quality step that never reads the outcome label (``y_disc`` is used only for its
    length) and never touches confirmation or test data. The hypotheses are annotated with
    ``annotate_fn`` before the first round and after every rewrite. Each round combines two
    feedback sources: (i) ``make_feedback`` flags hypotheses with prevalence below ``min_prev``
    for faithful broadening, and (ii) the optional ``critic_fn`` returns one action per
    hypothesis. Critic drops are applied directly; sharpen, split and reground actions that carry
    a reason, together with the broaden flags, become guidance for ``refine_fn``, which rewrites
    the remaining set. Redundancy and significance are handled downstream by deduplication
    (Phase 2.4) and the significance gate (Phase 2.5).

    The loop stops when a round flags nothing and the critic requests no action, when the
    rewriter returns the same set of texts (that round's critic drops are still applied), or
    after ``iters`` rounds. ``embed_fn``, ``sig_alpha`` and ``sim_thresh`` are accepted for
    interface compatibility and ignored.

    Args:
        hyps: initial hypotheses; blank and duplicate entries are removed, order is kept.
        y_disc: discovery-split labels, used only for their length.
        annotate_fn: maps a list of hypotheses to ``{hypothesis: annotation vector over the
            discovery rows}``; a nonzero entry means the hypothesis fires for that patient.
        refine_fn: the rewriter, e.g. from ``make_refine_fn``.
        critic_fn: the clinical critic, e.g. from ``make_critic_fn``; None disables it.
        iters: maximum number of rounds.
        min_prev: prevalence below which a hypothesis is flagged for broadening.

    Returns:
        ``(refined_hyps, trace)``. ``trace`` has one dict per round with summary counts and an
        ``actions`` list (per hypothesis: prevalence, broaden flag, critic action and reason).
        Rounds that call the rewriter also record the ``removed`` and ``introduced`` texts.
    """
    hyps = list(dict.fromkeys(h for h in hyps if str(h).strip()))
    ann = annotate_fn(hyps)
    trace = []
    for t in range(iters):
        fb_data, summary = make_feedback(hyps, ann, min_prev=min_prev)
        actions, crit_lines = {}, []
        if critic_fn is not None:
            prev = {h: float(np.mean(np.asarray(ann.get(h, np.zeros(len(y_disc))), dtype=float) != 0))
                    for h in hyps}
            try:
                actions = critic_fn(hyps, prev) or {}
            except Exception:
                actions = {}
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
                      "n_insignificant": 0, "n_redundant_pairs": 0,
                      "n_dropped": len(dropped), "n_critic_edits": len(crit_lines),
                      "actions": iter_actions})
        if n_actions == 0:
            break
        feedback = fb_data
        if crit_lines:
            feedback += "\n\nClinical-critic guidance:\n" + "\n".join(crit_lines)
        if dropped:
            feedback += "\n\nDrop these entirely: " + "; ".join(f'"{h}"' for h in dropped)
        new = list(dict.fromkeys(h for h in refine_fn(kept, feedback) if str(h).strip())) or kept
        # The rewrite is a single set-to-set LLM call, so there is no per-item parent link. Record
        # which texts left and which entered the set, so each refined text in
        # self_refine_trace.json can be traced to the round that introduced it.
        trace[-1]["removed"] = [h for h in hyps if h not in set(new)]
        trace[-1]["introduced"] = [h for h in new if h not in set(hyps)]
        if set(new) == set(hyps):                 # the rewriter changed no text
            if dropped:                           # apply this round's critic drops before stopping
                hyps, ann = kept, annotate_fn(kept)
            break
        hyps, ann = new, annotate_fn(new)
    return hyps, trace

