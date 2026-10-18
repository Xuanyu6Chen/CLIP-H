"""
clip_h_lib: shared library for the CLIP-H MIMIC-IV runners.

This module holds the data, statistics and bookkeeping code shared by the
in-hospital mortality and discharge-destination (Home vs other) runner notebooks
(``mortality_runner_gpt.ipynb``, ``mortality_runner_deepseek.ipynb``,
``discharge_runner_gpt.ipynb``, ``discharge_runner_deepseek.ipynb``), so the
notebooks stay thin and the logic stays testable:

  - Reproducibility:   set_global_seed
  - Prevalence:        prevalence_correct
  - Phase 2.1 (neuron interpretation):
                       clinical_interpret_instructions
  - Phase 2.3 (SCREEN, clinical critique):
                       clinical_instrument_grounding, two_stage_screen_vote,
                       cohen_kappa
  - Phase 2.4 (DEDUP by firing vectors):
                       phi_matrix, predictive_dedup, make_dedup_adjudicator,
                       connected_components, union_indicator,
                       annotate_features
  - Phase 2.5 (significance gate):
                       stable_hypotheses, stability_selection, min_fire_count,
                       univariate_test, subgroup_rate_test, bh_fdr, bonferroni
  - Phase 3 (interpretable prediction):
                       logit_inference, tuned_threshold, make_resamples,
                       bootstrap_ci, prevalence_matched, score_with_ci
  - Feature baselines: run_feature_baselines, feature_baseline_columns
                       (TF-IDF+LR, structured LR, XGBoost via baselines_ci)
  - Lineage and outputs:
                       build_lineage, lineage_candidates_from_stage_logs,
                       log_stage_table, write_predictions, record_threshold
  - Robustness:        preflight_config, ping_model,
                       validate_embeddings, validate_hypotheses
  - Bookkeeping:       config_hash, run_dir, write_manifest

Self-refinement (Phase 2.2) lives in ``self_refine.py``; the feature baselines'
model code lives in ``baselines_ci.py``. Reading MIMIC-IV, the cohort, the split, the
class-balanced draw and the patient profiles live in ``mimic_cleaning/src_2/`` at the
repository root; ``mimic_cleaning/run/run_<task>.py`` writes them to ``data/<task>/``, which
the runner notebooks read.

Design notes
------------
* Determinism: apart from ``set_global_seed``, every randomised step takes an
  explicit seed and uses a local ``np.random.default_rng`` (or ``random_state``)
  rather than global RNG state.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time as _time
from typing import Dict, List, Optional, Sequence, Tuple, Callable

import numpy as np
import pandas as pd


# ============================================================================ #
# Reproducibility
# ============================================================================ #
def set_global_seed(seed: int = 42) -> None:
    """Seed every RNG that can affect the pipeline, for reproducible runs.

    Seeds Python ``random``, NumPy, ``PYTHONHASHSEED``, and (if available) torch
    on CPU+CUDA with deterministic algorithms / cuDNN. Safe to call when torch
    isn't installed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # older torch without warn_only
            torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass


# ============================================================================ #
# Prevalence correction (Phase 3)
# ============================================================================ #
def prevalence_correct(p, train_prev: float, test_prev: float, eps: float = 1e-6):
    """Prior-correction for a model trained at a different class prevalence than the test set.

    When the training set is class-balanced (via undersampling) but the test set keeps the natural
    prevalence, a probabilistic classifier's outputs are biased toward the balanced prior. This shifts each
    prediction's logit by the log prior-odds ratio so the returned probabilities are calibrated to
    ``test_prev`` (Elkan, "The Foundations of Cost-Sensitive Learning", IJCAI 2001; King & Zeng,
    "Logistic Regression in Rare Events Data", Political Analysis 2001). Only threshold/calibration
    metrics (accuracy, F1, Brier) change; AUROC/AUPRC are ranking metrics and are invariant to this
    monotonic transform. A no-op when the two prevalences are equal.
    """
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    tr = min(max(float(train_prev), eps), 1 - eps)
    te = min(max(float(test_prev), eps), 1 - eps)
    shift = np.log(te / (1 - te)) - np.log(tr / (1 - tr))
    return 1.0 / (1.0 + np.exp(-(np.log(p / (1 - p)) + shift)))


# ---------------------------------------------------------------------------- #
# Clinical-steering interpreter instructions (Phase 2.1).
# Passed as ``task_specific_instructions`` to hypothesaes.generate_hypotheses so
# the interpreter names each SAE neuron as a clinically meaningful, admission-time
# concept rather than defaulting to many narrow demographic bins. It does not
# exclude demographics (they remain admissible when they are the true separator);
# it (a) prefers a clinical framing when the signal is clinical, and (b) asks for a
# single clinically standard granularity for demographic concepts, so one age effect
# does not surface as many overlapping age-band candidates. Only the outcome phrase
# changes between tasks.
# ---------------------------------------------------------------------------- #
TASK_OUTCOME_PHRASE: Dict[str, str] = {
    "mortality": "in-hospital mortality (whether the patient dies during this admission)",
    "discharge": "discharge destination (home versus another care facility)",
}


def clinical_interpret_instructions(task_or_phrase: str) -> str:
    """Task-specific interpreter guidance that steers hypothesis naming toward
    clinically meaningful, admission-time concepts for a given prediction outcome,
    without excluding demographics.

    ``task_or_phrase`` may be a task key in ``TASK_OUTCOME_PHRASE`` (e.g.
    ``"mortality"``) or a free-text outcome phrase. Pass the returned string as
    ``task_specific_instructions`` to ``generate_hypotheses``.
    """
    outcome = TASK_OUTCOME_PHRASE.get(task_or_phrase, task_or_phrase)
    return (
        f"These text samples are hospital admission profiles, and the feature you name becomes a "
        f"hypothesis that a physician will review and that is used to predict {outcome}. Name the "
        f"feature the way a clinician would describe a single patient-level finding present at "
        f"admission, phrased as \"patients with ...\", so it reads as a clinical statement a "
        f"reviewer could confirm from the chart.\n"
        "When the distinguishing signal is clinical, use that framing in preference to a demographic "
        "one: laboratory abnormalities and their direction (for example low albumin, elevated "
        "creatinine, anaemia, hyponatraemia), vital-sign or acuity patterns, organ dysfunction, "
        "comorbidity burden, admission type or acuity, and care-need or functional-status signals such "
        "as admission from a skilled-nursing facility. A coherent compound pattern is welcome when the "
        "evidence supports one recognizable clinical picture (for example \"anaemia with renal "
        "impairment\"); avoid stacking unrelated attributes that only happen to co-occur.\n"
        "Demographic or administrative attributes (age, sex, race, insurance, marital status, language) "
        "are allowed and should not be avoided when they are genuinely what separates the positive "
        "samples, but name them at one clinically standard granularity, for example a single age "
        "threshold such as \"75 years or older\", rather than several overlapping narrow bands, and "
        "express them as a range or direction rather than an exact value.\n"
        "Prefer wording that maps to a recognized clinical mechanism, so a reviewing physician can see "
        "why the feature would relate to the outcome. Ground every feature in what is written in the "
        "profile; do not infer diagnoses that are not stated."
    )


# ---------------------------------------------------------------------------- #
# Validated clinical risk instruments per task. Shown to the SCREEN reviewers
# (Phase 2.3) as reference knowledge that grounds the clinical_validity and
# mechanism judgments only; it is never a reason to raise a score. The
# machine-generated concepts are thus audited against the instruments a clinician
# would reach for.
# ---------------------------------------------------------------------------- #
TASK_CLINICAL_INSTRUMENTS: Dict[str, str] = {
    "mortality": (
        "validated inpatient-mortality instruments (OASIS, the early SOFA score, APACHE, and the "
        "Charlson comorbidity index). Admission-relevant components include deranged vital signs "
        "(heart rate, mean arterial pressure, temperature, respiratory rate), impaired "
        "consciousness (low Glasgow Coma Scale), advanced age, mechanical ventilation, low urine "
        "output, and acute organ-dysfunction labs (lactate, creatinine, bilirubin, platelets)"
    ),
    "discharge": (
        "validated predictors of discharge to a facility rather than home (functional and mobility "
        "status, care needs, and comorbidity burden). Relevant components include impaired "
        "functional or mobility status, need for assistance with activities of daily living, "
        "admission from a skilled-nursing facility, advanced age, and a high comorbidity burden"
    ),
}


def clinical_instrument_grounding(task: str) -> str:
    """Reviewer-facing reference knowledge for the SCREEN stage: the validated clinical
    risk instrument(s) for ``task`` plus a leakage guardrail. Returns ``''`` for an
    unknown task.

    The grounding informs the reviewer's ``clinical_validity`` and ``mechanism`` judgments
    only. It must never raise a score merely because a concept matches an instrument
    component, and the reviewer does not invent concepts to cover components that no
    hypothesis names.
    """
    body = TASK_CLINICAL_INSTRUMENTS.get(task, "")
    if not body:
        return ""
    return (
        "Reference knowledge for your judgment -- " + body + ". Use these instruments only to "
        "inform clinical_validity and the mechanism: do not raise clinical_validity merely because "
        "a concept matches a component, and do not expect the hypotheses to cover every component."
    )


# ============================================================================ #
# Evaluation (Phase 3) and feature baselines
# ============================================================================ #


def tuned_threshold(y_train: Sequence, p_train: Sequence) -> float:
    """Decision threshold that maximises F1 on the training predictions.

    Returns 0.5 when the training labels have a single class or the precision-recall curve has
    no thresholds. Defined at module level so every method that uses a train-tuned threshold
    applies the same rule: threshold-dependent metrics (F1, accuracy, precision, recall) are
    comparable across methods only when each method's threshold is chosen the same way.
    """
    from sklearn.metrics import precision_recall_curve
    y_train = np.asarray(y_train).astype(int)
    p_train = np.asarray(p_train, dtype=float)
    if len(np.unique(y_train)) < 2:
        return 0.5
    prec, rec, thr = precision_recall_curve(y_train, p_train)
    if thr.size == 0:
        return 0.5
    f1s = (2 * prec[:-1] * rec[:-1]) / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[int(np.nanargmax(f1s))])


def make_resamples(n: int, n_boot: int = 10000, seed: int = 42,
                   groups: Optional[Sequence] = None) -> "np.ndarray":
    """Draw ``n_boot`` bootstrap index vectors once, to be shared by every metric and method.

    Sharing the resamples makes CIs and pairwise deltas comparable across methods: two methods
    evaluated on independently drawn resamples cannot be differenced, because the sampling noise
    does not cancel. Without ``groups``, returns an ``[n_boot, n]`` integer array of row indices.

    When ``groups`` is given (e.g. ``subject_id``), resampling is done at the group level: whole
    patients are drawn with replacement and all of their rows come along. A test set can hold
    several admissions of one patient, and row-level resampling would treat them as independent
    evidence and understate the CI. Group resamples have variable length, so this returns an
    object array of index vectors in that case.
    """
    rng = np.random.default_rng(seed)
    if groups is None:
        return rng.integers(0, n, size=(n_boot, n))
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    rows_of = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    out = np.empty(n_boot, dtype=object)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        out[b] = np.concatenate([rows_of[g] for g in pick]) if len(uniq) else np.array([], int)
    return out


def bootstrap_ci(
    y_true: Sequence,
    y_score: Sequence,
    metric: str = "auroc",
    n_boot: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
    threshold: float = 0.5,
    resamples: Optional["np.ndarray"] = None,
) -> Dict[str, float]:
    """Bootstrap a 95% CI for a metric over the test set.

    ``metric`` in {"auroc", "accuracy", "f1", "balanced_accuracy", "auprc"}; the
    threshold-dependent metrics binarise ``y_score`` at ``threshold`` (default 0.5; pass the
    method's operating threshold explicitly).

    Pass ``resamples`` from ``make_resamples`` to reuse one shared set of bootstrap draws
    across metrics and methods; otherwise a fresh row-level set is drawn from ``seed``.
    Returns {point, lo, hi, se, half_width, metric, n_boot_used}.
    """
    from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                                 balanced_accuracy_score, average_precision_score)

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    def compute(yt, ys):
        if metric == "auroc":
            if len(np.unique(yt)) < 2:   # AUROC undefined on a single-class sample
                return float("nan")
            return roc_auc_score(yt, ys)
        if metric == "auprc":
            if len(np.unique(yt)) < 2:
                return float("nan")
            return average_precision_score(yt, ys)
        yhat = (ys >= threshold).astype(int)
        if metric == "accuracy":
            return accuracy_score(yt, yhat)
        if metric == "balanced_accuracy":
            if len(np.unique(yt)) < 2:
                return float("nan")
            return balanced_accuracy_score(yt, yhat)
        if metric == "f1":
            return f1_score(yt, yhat, zero_division=0)
        raise ValueError(f"unknown metric {metric}")

    point = float(compute(y_true, y_score))
    n = len(y_true)
    if resamples is None:
        resamples = make_resamples(n, n_boot=n_boot, seed=seed)
    stats = []
    for idx in resamples:
        idx = np.asarray(idx, dtype=int)
        yt, ys = y_true[idx], y_score[idx]
        # Drop any resample that lost a class, for every metric, not only the ones that are
        # mathematically undefined on one class. Keeping such a draw for F1 alone would evaluate
        # the metrics on different subsets of the same resample set, so their intervals would no
        # longer be comparable or jointly interpretable. baselines_ci.score_with_ci applies the
        # same rule (it skips the draw before scoring either AUROC or F1).
        if len(np.unique(yt)) < 2:
            continue
        stats.append(compute(yt, ys))
    stats = np.asarray(stats, dtype=float)
    lo = float(np.percentile(stats, 100 * alpha / 2)) if len(stats) else float("nan")
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2))) if len(stats) else float("nan")
    se = float(np.std(stats)) if len(stats) else float("nan")
    return {"point": point, "lo": lo, "hi": hi, "se": se,
            "half_width": (hi - lo) / 2 if len(stats) else float("nan"),
            "metric": metric, "n_boot_used": int(len(stats))}


def feature_baseline_columns(train_df: pd.DataFrame) -> List[str]:
    """The numeric feature columns the LR / XGBoost baselines consume.

    Every int or float column except ``y``, ``subject_id`` and ``hadm_id``. Defined once so every
    caller uses the same definition of "structured features"; a silent column-set difference
    would make baseline numbers incomparable while each looked correct.
    """
    return [c for c in train_df.columns
            if train_df[c].dtype.kind in "if" and c not in ("y", "subject_id", "hadm_id")]


def prevalence_matched(y, prob, seed: int = 42):
    """Prevalence-matched operating point: top-k positive, k = number of true positives.

    Delegates to ``baselines_ci.prevalence_matched`` so every method uses the same operating
    point. It is calibration-free (it never assumes that 0.5 means anything for a given model),
    which makes one F1 column comparable across an L2-LR, XGBoost and an LLM arm whose
    confidences are 0.7/0.8/0.9. Ties break by seeded permutation rather than row order, so a
    frame sorted by outcome cannot hand a model free accuracy.
    """
    import baselines_ci as B
    return B.prevalence_matched(y, prob, seed=seed)


def score_with_ci(y, prob, pred=None, n_boot: int = 10000, seed: int = 42) -> Dict:
    """AUROC and positive-class F1 with 95% percentile bootstrap CIs, in percent.

    Delegates to ``baselines_ci.score_with_ci``. CLIP-H and the baselines are all scored with
    this one function, because a table whose cells come from different estimators cannot be
    compared cell to cell. Returns ``auroc``/``f1`` plus ``*_ci`` and ``*_pm`` (the half-width
    printed in parentheses).

    ``pred`` defaults to the prevalence-matched operating point; pass one explicitly only for a
    method that has a genuine threshold of its own.
    """
    import baselines_ci as B
    return B.score_with_ci(y, prob, pred=pred, n_boot=n_boot, seed=seed)


def run_feature_baselines(train_df: pd.DataFrame, test_df: pd.DataFrame,
                          train_profiles: Sequence[str], test_profiles: Sequence[str],
                          y_train: Sequence, y_test: Sequence, *, task: str,
                          seed: int = 42, resamples: Optional["np.ndarray"] = None,
                          n_boot: int = 10000, alpha: float = 0.05,
                          threshold_mode: str = "prevalence_matched",
                          fixed_threshold: float = 0.5, verbose: bool = True) -> Dict:
    """Fit and score TF-IDF+LR, structured LR and XGBoost on one held-out split.

    This is the single definition of the results table's feature-baseline block. The models are
    fit by ``baselines_ci.fit_baselines`` with the task's leakage rule from
    ``baselines_ci.LEAK_RULES``, on the same train/test split as CLIP-H. A pooled train+test
    resplit would not be a valid comparator, because CLIP-H's hypothesis library is selected
    using the train labels.

    Each method's test predictions are binarised at the prevalence-matched operating point and
    scored with ``baselines_ci.score_with_ci``. ``threshold_mode``, ``fixed_threshold`` and
    ``alpha`` are accepted but not used, so every entry of the returned ``thresholds`` is NaN.

    ``resamples`` (from ``make_resamples``) is checked and returned unchanged, not consumed:
    ``score_with_ci`` draws its own row-level bootstrap from ``seed``, which matches
    ``make_resamples(n, n_boot, seed)`` without groups. Group-level (subject) resamples raise
    ``ValueError``, because the baselines cannot be scored on them.

    Returns ``{'table': DataFrame, 'results': {name: fit-dict}, 'preds': {name: y_pred},
    'num_cols': [...], 'thresholds': {name: float}, 'resamples': resamples}``.
    """
    import baselines_ci as B

    y_train = np.asarray(y_train).astype(int)
    y_test = np.asarray(y_test).astype(int)
    leak = B.LEAK_RULES.get(task)
    if leak is None:
        raise ValueError(
            f"no leakage rule for task {task!r}. Add one to baselines_ci.LEAK_RULES; without a "
            f"rule, leaking columns would pass silently. Known: {sorted(B.LEAK_RULES)}")

    # The three baselines come from baselines_ci, the shared reference implementation, so every
    # task's baseline rows are produced by the same estimator.
    # score_with_ci draws its own row-level resamples from `seed`. With EVAL_RESAMPLE_BY='row'
    # those match make_resamples(n, n_boot, seed), so the baselines and CLIP-H share bootstrap
    # draws. That does not hold for subject-level draws, and scoring the baselines row-level
    # while CLIP-H is scored subject-level would put two different estimators in one column,
    # so group resamples raise an error instead.
    if resamples is not None and getattr(resamples, "dtype", None) == object:
        raise ValueError(
            "run_feature_baselines cannot use group (subject-level) resamples: "
            "baselines_ci.score_with_ci draws its own row-level bootstrap. Set "
            "EVAL_RESAMPLE_BY='row' so the two coincide.")

    probs = B.fit_baselines(list(train_profiles), y_train, train_df,
                            list(test_profiles), test_df, leak, seed=seed, verbose=verbose)

    rows, results, preds = [], {}, {}
    for name, prob in probs.items():
        # Prevalence-matched operating point: top-k positive, k = #true positives, ties broken
        # by seeded permutation. Neither 0.5 nor train-tuned: it is calibration-free, so the
        # same rule is meaningful for an L2-LR and for an LLM arm emitting 0.7/0.8/0.9.
        pred = B.prevalence_matched(y_test, prob, seed=seed)
        r = B.score_with_ci(y_test, prob, pred=pred, n_boot=n_boot, seed=seed)
        results[name] = {"y_prob": prob, "y_pred": pred,
                         "auroc": r["auroc"] / 100.0, "f1": r["f1"] / 100.0}
        preds[name] = pred
        rows.append({"method": name,
                     "auroc": r["auroc"] / 100.0,
                     "auroc_lo": r["auroc_ci"][0] / 100.0, "auroc_hi": r["auroc_ci"][1] / 100.0,
                     "auroc_half_width": r["auroc_pm"] / 100.0,
                     "f1": r["f1"] / 100.0,
                     "f1_lo": r["f1_ci"][0] / 100.0, "f1_hi": r["f1_ci"][1] / 100.0,
                     "f1_half_width": r["f1_pm"] / 100.0,
                     "operating_point": "prevalence_matched",
                     "n_test": int(len(y_test))})
    return {"table": pd.DataFrame(rows), "results": results, "preds": preds,
            "num_cols": feature_baseline_columns(train_df),
            "thresholds": {k: float("nan") for k in probs}, "resamples": resamples}


# ============================================================================ #
# Significance gate (Phase 2.5) and inference statistics
# ============================================================================ #
def min_fire_count(n: int) -> int:
    """N-adaptive minimum number of train patients a hypothesis must fire in.

    ``max(ceil(0.02 * n), ceil(sqrt(n)))``. The 2% term sets a minimum firing rate,
    removing "exception-of-an-exception" hypotheses (e.g. age == 91) that look correct
    on a handful of patients; the ``sqrt(n)`` term is a statistical-power floor (a
    prevalence or effect estimate's error scales like 1/sqrt(n)), so the threshold is
    never tiny at low n yet grows with n. There is no hardcoded patient count; the
    floor adapts to the actual train size.
    Examples: n=100 gives 10, n=500 gives 23, n=5000 gives 100, n=300000 gives 6000.
    """
    import math
    n = int(max(0, n))
    if n == 0:
        return 0
    return int(max(math.ceil(0.02 * n), math.ceil(math.sqrt(n))))


def _logodds_diff(x_bin: np.ndarray, y: np.ndarray) -> float:
    """Univariate log-odds difference logit(P(y|fired)) - logit(P(y|not fired)).

    Haldane-Anscombe 0.5 correction so the estimate is finite even when a cell is
    empty. Sign equals the sign of a single-feature logistic-regression coefficient,
    i.e. the hypothesis's data-driven direction toward the positive class.
    """
    fired = x_bin > 0
    n1, n0 = int(fired.sum()), int((~fired).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    a = float(y[fired].sum()); b = float(y[~fired].sum())
    p1 = (a + 0.5) / (n1 + 1.0)
    p0 = (b + 0.5) / (n0 + 1.0)
    return float(np.log(p1 / (1 - p1)) - np.log(p0 / (1 - p0)))


def univariate_test(x_bin: np.ndarray, y: np.ndarray) -> "tuple[float, float]":
    """Two-sided p-value (Fisher's exact on the fired×outcome 2×2) + log-odds effect.

    Fisher's exact is valid at the small cell counts a single rare concept produces,
    where a chi-square / Wald approximation would be unreliable. Returns (p, effect);
    p = 1.0 when the concept is constant (fires for everyone or no one).
    """
    x = (np.asarray(x_bin, dtype=float) > 0)
    y = np.asarray(y, dtype=float) > 0
    n1, n0 = int(x.sum()), int((~x).sum())
    if n1 == 0 or n0 == 0:
        return 1.0, float("nan")
    a = int((x & y).sum()); b = int((x & ~y).sum())
    c = int((~x & y).sum()); d = int((~x & ~y).sum())
    try:
        from scipy.stats import fisher_exact
        p = float(fisher_exact([[a, b], [c, d]], alternative="two-sided")[1])
    except Exception:                       # scipy unavailable: p is NaN (BH treats it as 1.0)
        p = float("nan")
    return p, _logodds_diff(np.asarray(x_bin, dtype=float), np.asarray(y, dtype=float))


def bh_fdr(pvals: Sequence[float], q: float = 0.10) -> "tuple[np.ndarray, np.ndarray]":
    """Benjamini-Hochberg FDR control. Returns (rejected_bool, qvalues).

    Controls the expected false-discovery proportion at ``q`` across the family of
    hypotheses tested. This matters because testing, say, 25 concepts at α=0.05 yields one
    or two spurious "significant" hits by chance. NaN p-values are treated as 1.0 (not
    rejected).
    """
    p = np.asarray([1.0 if (v != v or v is None) else float(v) for v in pvals], dtype=float)
    m = len(p)
    if m == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    thresh = q * (np.arange(1, m + 1) / m)
    below = ranked <= thresh
    kmax = np.nonzero(below)[0].max() + 1 if below.any() else 0
    rejected = np.zeros(m, dtype=bool)
    if kmax > 0:
        rejected[order[:kmax]] = True
    # Monotone q-values (BH-adjusted p), for reporting.
    qvals_sorted = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    qvals = np.empty(m, dtype=float)
    qvals[order] = np.clip(qvals_sorted, 0, 1)
    return rejected, qvals


def bonferroni(pvals: Sequence[float], alpha: float = 0.05) -> "tuple[np.ndarray, np.ndarray]":
    """Bonferroni family-wise correction. Returns (rejected_bool, adjusted_p).

    The conservative complement to ``bh_fdr``: controls the probability of any false
    positive across the family (FWER) at ``alpha`` by testing each hypothesis at
    ``alpha/m``. Adjusted p = min(p*m, 1). NaN p-values are treated as 1.0 (not rejected).
    Reported only; the significance gate uses BH-FDR.
    """
    p = np.asarray([1.0 if (v != v or v is None) else float(v) for v in pvals], dtype=float)
    m = len(p)
    if m == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)
    adj = np.clip(p * m, 0.0, 1.0)
    return (p < (alpha / m)), adj


def subgroup_rate_test(x_bin: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Interpretable, prevalence-aware subgroup test: is the outcome rate among patients
    for whom a hypothesis fires significantly different from the rate among the rest?

    Returns rate_if_fires, rate_if_not, base_rate, diff (fires minus not), a 95% CI on that
    difference (two-proportion normal approximation), ``p`` (Fisher's exact via
    ``univariate_test``, exact at the small cell counts a rare concept produces) and
    ``n_fire``. This is the statistic behind statements like "patients with X go home at
    rate a versus a base rate b (p=...)".
    """
    x = np.asarray(x_bin, dtype=float) > 0
    yv = np.asarray(y, dtype=float)
    n1, n0 = int(x.sum()), int((~x).sum())
    base = float(yv.mean()) if len(yv) else float("nan")
    if n1 == 0 or n0 == 0:
        return {"rate_if_fires": float("nan"), "rate_if_not": float("nan"), "base_rate": base,
                "diff": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
                "p": 1.0, "n_fire": n1}
    p1 = float(yv[x].mean()); p0 = float(yv[~x].mean())
    se = float(np.sqrt(max(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0, 0.0)))
    diff = p1 - p0
    p_val = univariate_test(x_bin, y)[0]
    return {"rate_if_fires": p1, "rate_if_not": p0, "base_rate": base, "diff": diff,
            "ci_lo": diff - 1.96 * se, "ci_hi": diff + 1.96 * se, "p": p_val, "n_fire": n1}


def logit_inference(X: np.ndarray, y: Sequence, names: Sequence[str],
                    seed: int = 42) -> pd.DataFrame:
    """Multivariate logistic-regression inference (signed coefficients and Wald p-values).

    Fits an unregularized logistic regression purely for inference (the descriptive Phase 3
    refit); the regularized L2 head remains the predictor, because regularized coefficients are
    biased for inference. Wald statistics come from the Fisher information of the fitted model:
    ``cov = pinv(Xᵀ W X)`` with ``W = diag(p(1−p))``, ``se = √diag``, ``z = coef/se``,
    ``p = 2(1−Φ|z|)``. Returns one row per feature: ``name, coef, se, z, pvalue, ci_lo, ci_hi,
    unstable`` (95% CI). On perfect separation or non-convergence (NaN, zero or very large se)
    the row is flagged ``unstable=True`` and ``pvalue`` falls back to the univariate Fisher p,
    so degenerate data never raises.
    """
    from sklearn.linear_model import LogisticRegression
    from scipy.stats import norm

    Xa = np.asarray(X, dtype=float)
    yv = np.asarray(y, dtype=float)
    names = list(names)
    n, d = Xa.shape
    rows = []
    try:
        clf = LogisticRegression(penalty=None, max_iter=5000, solver="lbfgs")
        clf.fit(Xa, yv)
        coefs = clf.coef_[0]
        Xd = np.column_stack([np.ones(n), Xa])               # design w/ intercept
        pr = clf.predict_proba(Xa)[:, 1]
        W = np.clip(pr * (1 - pr), 1e-9, None)
        cov = np.linalg.pinv(Xd.T * W @ Xd)                  # (d+1)x(d+1)
        se_all = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        se = se_all[1:]                                      # drop intercept
    except Exception:
        coefs = np.full(d, np.nan); se = np.full(d, np.nan)
    for j, nm in enumerate(names):
        c = float(coefs[j]); s = float(se[j])
        unstable = (not np.isfinite(s)) or s > 1e3 or s == 0.0
        if not unstable:
            z = c / s
            pv = float(2 * (1 - norm.cdf(abs(z))))
            lo, hi = c - 1.96 * s, c + 1.96 * s
        else:                                                # separation: univariate fallback
            z = float("nan")
            pv = univariate_test(Xa[:, j], yv)[0]
            lo = hi = float("nan")
        rows.append({"name": nm, "coef": c, "se": s, "z": z, "pvalue": pv,
                     "ci_lo": lo, "ci_hi": hi, "unstable": bool(unstable)})
    return pd.DataFrame(rows)


def stability_selection(
    annotations: Dict[str, Sequence],
    y: Sequence,
    B: int = 200,
    q: float = 0.10,
    min_fire: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, float]:
    """Meinshausen-Bühlmann stability selection: per-hypothesis selection frequency.

    Bootstrap-resample the rows ``B`` times; in each draw recompute Fisher p-values for
    every hypothesis, apply BH-FDR, and mark a hypothesis "selected" if it is FDR-
    significant and fires in at least ``min_fire`` resampled patients. The returned
    selection frequency (fraction of valid draws selecting it; draws that lose a class are
    skipped) is a reproducibility-based trust score that is robust to the single-split luck
    a one-shot significance test is vulnerable to. It works on cached annotations, so it
    adds no API cost.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if min_fire is None:
        min_fire = min_fire_count(n)
    hyps = list(annotations.keys())
    X = {h: (np.asarray(annotations[h], dtype=float) > 0).astype(float) for h in hyps}
    rng = np.random.default_rng(seed)
    counts = {h: 0 for h in hyps}
    valid = 0
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        valid += 1
        pvals = [univariate_test(X[h][idx], yb)[0] for h in hyps]
        rej, _ = bh_fdr(pvals, q=q)
        for h, r in zip(hyps, rej):
            if r and int(X[h][idx].sum()) >= min_fire:
                counts[h] += 1
    denom = max(1, valid)
    return {h: counts[h] / denom for h in hyps}


def stable_hypotheses(
    annotations: Dict[str, Sequence],
    y: Sequence,
    n_splits: int = 5,
    seed: int = 42,
    n_boot: int = 1000,
    min_sign_consistency: float = 0.8,
    min_fire: Optional[int] = None,
    fdr_q: float = 0.10,
    stability_pi: float = 0.6,
    stability_B: int = 200,
) -> pd.DataFrame:
    """Significance and reproducibility gate for hypotheses (Phase 2.5, hard drop).

    For each hypothesis (a binary concept column) measure its univariate association
    with the outcome and keep only those that are simultaneously:
      (i)   *FDR-significant*: Fisher's exact p, then Benjamini-Hochberg control at
            ``fdr_q`` across the whole family (so a few chance hits do not survive);
      (ii)  *stable*: selected (FDR-significant and estimable) in at least
            ``stability_pi`` of ``stability_B`` bootstrap resamples; and
      (iii) *estimable*: fires in at least ``min_fire`` train patients (pass
            ``min_fire_count(len(y))``; that is the default when None).

    Also reports the bootstrap-CI significance, per-fold sign consistency, subgroup outcome
    rates and Bonferroni-adjusted p-values for transparency; these do not affect ``keep``.
    The gate removes small-sample noise and irreproducible concepts, so only stable signal
    reaches the Phase 3 fit.

    Returns one row per hypothesis with: ``n_fire``, ``univariate_coef``,
    ``ci_lo``/``ci_hi``, ``pvalue``, ``qvalue``, ``fdr_significant``, ``significant``
    (= fdr_significant, the authoritative flag), ``ci_significant``,
    ``sign_consistency``, ``selection_frequency``, ``univariate_direction``, ``min_fire``,
    ``rate_if_fires``, ``rate_if_not``, ``base_rate``, ``rate_diff``,
    ``rate_diff_ci_lo``/``rate_diff_ci_hi``, ``bonferroni_p``, ``bonferroni_significant``
    and ``keep``.
    """
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(y, dtype=float)
    n = len(y)
    if min_fire is None:
        min_fire = min_fire_count(n)
    rng = np.random.default_rng(seed)
    sel_freq = stability_selection(annotations, y, B=stability_B, q=fdr_q,
                                   min_fire=min_fire, seed=seed)
    rows = []
    for h, col in annotations.items():
        x = (np.asarray(col, dtype=float) > 0).astype(float)
        n_fire = int(x.sum())
        coef = _logodds_diff(x, y)
        pval = univariate_test(x, y)[0]
        # Bootstrap CI of the univariate coefficient.
        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            g = _logodds_diff(x[idx], y[idx])
            if g == g:  # not NaN
                boots.append(g)
        if boots:
            ci_lo = float(np.percentile(boots, 2.5))
            ci_hi = float(np.percentile(boots, 97.5))
        else:
            ci_lo = ci_hi = float("nan")
        ci_significant = bool(boots) and (ci_lo > 0 or ci_hi < 0)
        # Per-fold sign consistency.
        signs = []
        if len(np.unique(y)) > 1 and 0 < n_fire < n:
            try:
                skf = StratifiedKFold(n_splits=min(n_splits, int(y.sum()), int(n - y.sum())),
                                      shuffle=True, random_state=seed)
                for _, fold_idx in skf.split(x, y):
                    g = _logodds_diff(x[fold_idx], y[fold_idx])
                    if g == g and g != 0:
                        signs.append(np.sign(g))
            except ValueError:
                signs = []
        if signs and coef == coef and coef != 0:
            sign_consistency = float(np.mean(np.sign(coef) == np.asarray(signs)))
        else:
            sign_consistency = float("nan")
        direction = ("increase" if coef > 0 else "decrease") if (coef == coef and coef != 0) else "unclear"
        sub = subgroup_rate_test(x, y)       # interpretable subgroup rate vs base rate + CI
        rows.append({
            "hypothesis": h, "n_fire": n_fire, "univariate_coef": coef,
            "ci_lo": ci_lo, "ci_hi": ci_hi, "pvalue": pval,
            "ci_significant": ci_significant, "sign_consistency": sign_consistency,
            "selection_frequency": float(sel_freq.get(h, 0.0)),
            "univariate_direction": direction, "min_fire": int(min_fire),
            "rate_if_fires": sub["rate_if_fires"], "rate_if_not": sub["rate_if_not"],
            "base_rate": sub["base_rate"], "rate_diff": sub["diff"],
            "rate_diff_ci_lo": sub["ci_lo"], "rate_diff_ci_hi": sub["ci_hi"],
        })

    # Multiple-testing correction across all hypotheses, then the hard-drop keep decision.
    # BH-FDR is the gate; Bonferroni (FWER, conservative) is reported alongside.
    rejected, qvals = bh_fdr([r["pvalue"] for r in rows], q=fdr_q)
    bonf_rej, bonf_p = bonferroni([r["pvalue"] for r in rows], alpha=0.05)   # conventional FWER bar
    for r, rej, qv, brej, bp in zip(rows, rejected, qvals, bonf_rej, bonf_p):
        r["qvalue"] = float(qv)
        r["fdr_significant"] = bool(rej)
        r["significant"] = bool(rej)        # authoritative significance = FDR-corrected
        r["bonferroni_p"] = float(bp)
        r["bonferroni_significant"] = bool(brej)
        r["keep"] = bool(rej
                         and r["n_fire"] >= min_fire     # estimability floor, not a prevalence threshold
                         and r["selection_frequency"] >= stability_pi)
    return pd.DataFrame(rows)


# ============================================================================ #
# Inter-judge agreement (Phase 2.3)
# ============================================================================ #


def cohen_kappa(a: Sequence, b: Sequence) -> float:
    """Cohen's κ inter-rater agreement between two judges (e.g. their direction calls).

    Returns NaN when κ is undefined (no ratings, or both raters constant)."""
    from sklearn.metrics import cohen_kappa_score
    a = list(a); b = list(b)
    if not a or len(set(a)) < 2 and len(set(b)) < 2:
        return float("nan")
    try:
        return float(cohen_kappa_score(a, b))
    except Exception:
        return float("nan")


# ============================================================================ #
# Curation helpers for SCREEN (Phase 2.3), DEDUP (Phase 2.4) and the merged-feature
# annotation used by the gate and the fit. Pure functions with no LLM or SAE access.
# ============================================================================ #
_MAG_ORDER = {"none": 0, "weak": 1, "moderate": 2, "strong": 3}


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


def union_indicator(vectors: Sequence) -> "np.ndarray":
    """Elementwise logical OR (max) over binary indicator vectors: the merged feature
    fires iff any source concept fires. Pure."""
    arr = np.asarray(vectors, dtype=float)
    if arr.ndim == 1:
        return (arr != 0).astype(int)
    return (arr != 0).any(axis=0).astype(int)


def annotate_features(profiles: Sequence, labels: Sequence[str],
                      source_map: Dict, annotate_fn: Callable) -> Dict:
    """Build each label's indicator, decoupling merged labels from their indicators.

    For each label: if it is a merged label (present in ``source_map``), annotate every
    original member concept text and OR them; otherwise annotate the label directly.
    ``annotate_fn(texts)`` must return ``{text: vector}``. Merged features are thus
    built from the members' cached annotations, never from annotating the representative
    label alone (deterministic, no extra annotation cost, monotone-safe). Each distinct text
    is annotated once. ``profiles`` is not read here; ``annotate_fn`` determines which
    profiles are annotated. Pure with respect to the injected ``annotate_fn``.
    """
    need: List[str] = []
    for lab in labels:
        srcs = source_map.get(lab)
        need.extend(srcs if srcs else [lab])
    seen: set = set()
    uniq = [t for t in need if not (t in seen or seen.add(t))]
    ann = annotate_fn(uniq)
    out: Dict[str, "np.ndarray"] = {}
    for lab in labels:
        srcs = source_map.get(lab)
        out[lab] = union_indicator([np.asarray(ann[t]) for t in srcs]) if srcs \
            else np.asarray(ann[lab])
    return out


# ============================================================================ #
# Firing-vector DEDUP (Phase 2.4), the single redundancy-merge stage. Redundancy is
# a property of the per-patient firing vectors (do two concepts fire on the same
# patients?), not of the label wording. `phi_matrix` scores that; `predictive_dedup`
# merges only same-direction pairs whose firing correlates, with an LLM confirmer
# that can veto (never propose) a merge in a borderline band. Pure given the
# injected `adjudicate_fn`.
# ============================================================================ #
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


def predictive_dedup(vectors: Sequence, labels: Sequence[str],
                     directions: Sequence[str], meta: Optional[Sequence[Dict]] = None,
                     corr_thr: float = 0.90, auto_thr: float = 0.95,
                     adjudicate_fn: Optional[Callable] = None) -> Dict:
    """The single firing-vector redundancy merge. Two concepts are merged only when they
    fire on nearly the same patients and share a non-unclear ``expected_direction``:

        phi >= auto_thr            -> auto-merge (near-certain statistical duplicate)
        corr_thr <= phi < auto_thr -> ``adjudicate_fn`` may confirm the merge (else keep apart)
        phi <  corr_thr            -> keep separate (distinct firing, never touched)

    Opposite/unclear-direction pairs never merge. ``adjudicate_fn(a_label, b_label, a_meta,
    b_meta, stats) -> (decision, reason)`` returns ``'merge'``/``'keep_separate'``; it acts only
    on the borderline band and can only veto (statistics proposes, the LLM may refuse). With
    ``adjudicate_fn=None`` or on any error the borderline band defaults to keep_separate
    (conservative). Each merged cluster keeps a single representative: its broadest member, with
    ties broken by clinical_validity, then shortest label, then alphabetically. Near-perfect
    complementary pairs (phi <= -auto_thr, e.g. "male" vs "female") are not merged; one side is
    dropped so the pair does not enter the fit as two collinear columns.

    Returns a dict with ``source_map`` ({rep_label: [member label texts incl. rep]} for merged
    clusters only; feeds ``annotate_features`` for the OR-union indicator), ``kept`` (final
    representative/atomic labels in original order), ``pairs`` (every phi>=corr_thr pair with its
    tier/decision, for logging), ``clusters`` (merged groups with per-member phi-to-rep and
    coverage of the union), ``complements`` (every pair with phi <= -corr_thr) and
    ``dropped_complement`` ({dropped_label: kept_label}).
    Pure/deterministic given ``adjudicate_fn``.
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
    complements: List[Dict] = []
    for i in range(n):
        for j in range(i + 1, n):
            phi = float(P[i, j])
            # Complementary pairs: two concepts that fire on disjoint patients so strongly
            # that one is essentially the negation of the other ("male sex" vs "female sex",
            # phi near -1). Merging them by OR is meaningless (the union fires on everyone),
            # and certifying both is worse: they enter the design matrix as collinear columns,
            # split one coefficient into two unstable halves with scrambled signs, and
            # inflate the certified-hypothesis count with a single variable counted twice.
            # Record every strong complement; below, drop one side of each near-perfect
            # (phi <= -auto_thr) complement whose members both survived as representatives.
            if phi <= -corr_thr:
                complements.append({
                    "concept_a": labels[i], "concept_b": labels[j], "phi": round(phi, 4),
                    "fires_a": fires[i], "fires_b": fires[j],
                    "cofire_both": int(np.sum(bin_vecs[i] & bin_vecs[j])),
                    "collinear": bool(phi <= -auto_thr),
                })
                continue
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
                    except Exception:
                        decision, reason = "keep_separate", "adjudicator error -> keep"
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

    def _rep(idxs: List[int]) -> int:
        """Pick the cluster's representative: the broadest member.

        The merged feature is the OR-union of the members (``union_indicator``), so it fires
        wherever the most inclusive member fires. Choosing the most specific member would give
        a label that describes a strictly narrower concept than the feature being fit,
        certified and shown to clinicians (e.g. a cluster of {"aged 82 or older", "aged 80 or
        older"} labelled "82 or older" while the indicator is 80+). Sorting by prevalence,
        descending, makes the label a faithful description of the union: for the nested bands
        that dominate these clusters, the broadest member is the union. Ties break on clinical
        validity, then shortest label, then alphabetically, so the choice is deterministic."""
        def key(i):
            v = float(meta[i].get("clinical_validity", 2)) if isinstance(meta[i], dict) else 2.0
            return (-prevalence[i], -v, len(labels[i]), labels[i])
        return min(idxs, key=key)

    # Connected components admit transitive merges: if pairs (a, b) and (b, c) both clear
    # corr_thr, a and c land in one cluster even when phi(a, c) is well below it, and the
    # cluster's OR-union is broader than anything the pairwise evidence justifies. Refine each
    # component so every member is directly correlated with the representative at >= corr_thr;
    # members that fall short are re-clustered among themselves (repeatedly), never discarded.
    comps_raw = connected_components(n, edges)
    comps: List[List[int]] = []
    for idxs in comps_raw:
        pool = list(idxs)
        while pool:
            rep = _rep(pool)
            grp = [rep] + [k for k in pool if k != rep and float(P[rep, k]) >= corr_thr]
            grp_set = set(grp)
            comps.append(sorted(grp))
            pool = [k for k in pool if k not in grp_set]
    comp_of = {i: c for c in range(len(comps)) for i in comps[c]}

    source_map: Dict[str, List[str]] = {}
    clusters: List[Dict] = []
    rep_label_of_comp: Dict[int, str] = {}
    rep_idx_of_comp: Dict[int, int] = {}
    for c, idxs in enumerate(comps):
        rep = _rep(idxs)
        rep_label_of_comp[c] = labels[rep]
        rep_idx_of_comp[c] = rep
        if len(idxs) > 1:
            members = [labels[rep]] + [labels[k] for k in idxs if k != rep]
            source_map[labels[rep]] = members
            # Audit the label-vs-feature correspondence the representative rule is supposed to
            # guarantee: what fraction of the union's firing does the representative alone
            # account for? 1.0 means the label describes the feature exactly (nested bands).
            uni = union_indicator([bin_vecs[k] for k in idxs]).astype(bool)
            n_uni = int(uni.sum())
            clusters.append({"representative": labels[rep], "members": members,
                             "phi_to_rep": {labels[k]: round(float(P[rep, k]), 4)
                                            for k in idxs if k != rep},
                             "rep_prevalence": round(prevalence[rep], 4),
                             "union_prevalence": round(float(uni.mean()) if uni.size else 0.0, 4),
                             "rep_covers_union": round(
                                 float(np.sum(bin_vecs[rep] & uni)) / n_uni if n_uni else 1.0, 4)})

    # Drop one member of each near-perfect complementary pair whose two sides both survived as
    # cluster representatives (see the `complements` block above). The kept side is the one the
    # `_rep` ordering prefers, so the choice is deterministic; the dropped label is logged.
    dropped_complement: Dict[str, str] = {}
    surviving = {rep_idx_of_comp[c] for c in range(len(comps))}
    label_to_idx = {labels[i]: i for i in range(n)}
    for rec in complements:
        if not rec["collinear"]:
            continue
        i, j = label_to_idx.get(rec["concept_a"]), label_to_idx.get(rec["concept_b"])
        if i is None or j is None or i not in surviving or j not in surviving:
            continue
        keep_i = _rep([i, j])
        drop_i = j if keep_i == i else i
        dropped_complement[labels[drop_i]] = labels[keep_i]
        surviving.discard(drop_i)
        rec["dropped"] = labels[drop_i]
        rec["kept"] = labels[keep_i]

    seen: set = set()
    kept: List[str] = []
    for i in range(n):
        c = comp_of[i]
        rl = rep_label_of_comp[c]
        if rep_idx_of_comp[c] not in surviving:
            continue
        if rl not in seen:
            seen.add(rl); kept.append(rl)
    source_map = {k: v for k, v in source_map.items() if k not in dropped_complement}
    return {"source_map": source_map, "kept": kept, "pairs": pairs, "clusters": clusters,
            "complements": complements, "dropped_complement": dropped_complement}


# Pairs the DEDUP adjudicator could not reach. Populated by make_dedup_adjudicator and surfaced
# by the runners so an outage during DEDUP is visible rather than hidden inside a reason string.
ADJUDICATOR_FAILURES: List[str] = []


DEDUP_CONFIRMER_SYSTEM = (
    "You are a clinical-informatics adjudicator deciding whether two machine-generated clinical "
    "concepts name the SAME underlying construct or two DISTINCT ones. A statistical screen has "
    "already found they fire on almost the same patients; you are the tie-breaker that prevents "
    "collapsing two genuinely different signals. Respond with only valid JSON."
)


def make_dedup_adjudicator(model: str, task: str = "", pos_label: str = "the positive class",
                           complete_fn: Callable = None, timeout: int = 120, seed: int = 0):
    """Return ``adjudicate_fn(a, b, a_meta, b_meta, stats) -> (decision, reason)`` backed by one
    prevalence-aware LLM call. It may only confirm a statistically proposed merge: it answers
    'merge' when the two are a paraphrase or a different granularity of one variable, and
    'keep_separate' otherwise (distinct constructs that merely co-occur).

    Failure-safe: a failed API call is retried (``DEDUP_ADJ_TRIES`` attempts, default 3, with
    ``DEDUP_ADJ_WAIT_S`` seconds between them, default 30); if every attempt fails, the pair is
    recorded in ``ADJUDICATOR_FAILURES`` and kept separate. An unparseable reply also means
    keep_separate (conservative veto). ``complete_fn`` is injectable and defaults to
    ``hypothesaes.llm_api.get_completion``."""
    if complete_fn is None:
        from hypothesaes.llm_api import get_completion as complete_fn  # noqa

    def adjudicate_fn(a, b, a_meta, b_meta, stats):
        prompt = (
            f"Two candidate concepts predict the {task} outcome (positive class: {pos_label}) and "
            f"fire on nearly the same patients (phi={stats.get('phi', float('nan')):.2f}). Decide "
            "whether they name the SAME underlying clinical construct through the SAME mechanism, or "
            "two genuinely distinct constructs that merely co-occur.\n"
            "- Answer \"merge\" only when one is a paraphrase or a different granularity of the other "
            "(same variable / mechanism).\n"
            "- Answer \"keep_separate\" when they are distinct clinical constructs, even if they "
            "overlap heavily (for example a cause and its downstream marker, or two organ systems "
            "that co-occur).\n\n"
            f"<concept_a>\n{a}  [fires in {stats.get('cofire_a', 0)} patients]\n</concept_a>\n"
            f"<concept_b>\n{b}  [fires in {stats.get('cofire_b', 0)} patients]\n</concept_b>\n"
            f"<firing_overlap>phi={stats.get('phi', float('nan')):.2f}; both fire in "
            f"{stats.get('cofire_both', 0)} patients</firing_overlap>\n\n"
            "<examples>\n"
            '<example>"anaemia" vs "low hemoglobin on admission" -> '
            '{"decision":"merge","reason":"same construct, different wording"}</example>\n'
            '<example>"on dialysis" vs "elevated creatinine" -> '
            '{"decision":"keep_separate","reason":"cause vs downstream marker; distinct mechanisms that co-occur"}</example>\n'
            '<example>"brief ICU stay" vs "prolonged ICU stay" -> '
            '{"decision":"keep_separate","reason":"opposite ends of one variable; different patient groups"}</example>\n'
            '<example>"sepsis" vs "positive blood culture treated with antibiotics" -> '
            '{"decision":"merge","reason":"different granularity of the same clinical state"}</example>\n'
            "</examples>\n\n"
            'Return only JSON: {"decision":"merge"|"keep_separate","reason":"<short>"}.')
        # Retry before falling back. The failure mode is silent and directional: an error
        # returns "keep_separate", so a network outage during DEDUP leaves borderline pairs
        # un-merged and inflates the certified-hypothesis count, with no trace except a reason
        # string in dedup_pairs.csv. Waiting for the connection avoids that.
        _tries = int(os.environ.get("DEDUP_ADJ_TRIES", "3"))
        _wait = float(os.environ.get("DEDUP_ADJ_WAIT_S", "30"))
        for _t in range(_tries):
            try:
                return _adjudicate_once(prompt)
            except Exception:
                if _t == _tries - 1:
                    ADJUDICATOR_FAILURES.append(f"{a} || {b}")
                    return "keep_separate", f"adjudicator unavailable after {_tries} tries -> keep"
                _time.sleep(_wait)
        return "keep_separate", "adjudicator error -> keep"

    def _adjudicate_once(prompt):
        # The API call is deliberately not wrapped: a network failure must propagate so the
        # caller's retry loop can wait for the connection. Only a parse failure (the model
        # answered but the reply could not be read) falls back, because retrying will not help.
        raw = complete_fn(prompt=prompt, model=model, system=DEDUP_CONFIRMER_SYSTEM,
                          max_tokens=200, timeout=timeout, seed=seed)
        try:
            s = re.sub(r"^```(?:json)?|```$", "", str(raw).strip(), flags=re.MULTILINE).strip()
            m = re.search(r"\{.*\}", s, re.DOTALL)
            obj = json.loads(m.group(0) if m else s)
            decision = str(obj.get("decision", "keep_separate")).strip().lower()
            reason = str(obj.get("reason", "")).strip()
            if decision not in ("merge", "keep_separate"):
                decision, reason = "keep_separate", "unparseable decision -> keep"
            return decision, reason
        except Exception:
            return "keep_separate", "unparseable reply -> keep"
    return adjudicate_fn


def log_stage_table(run_dir: str, name: str, rows: Sequence[Dict]) -> str:
    """Write a per-stage audit table (list of dicts to CSV) under ``run_dir``. Returns the path.
    A small, uniform writer for the per-stage logs and the end-to-end lineage."""
    fname = name if str(name).endswith(".csv") else f"{name}.csv"
    path = os.path.join(run_dir, fname)
    pd.DataFrame(list(rows)).to_csv(path, index=False)
    return path


def write_predictions(pred_root: str, task: str, method: str, generator: str,
                      patient_ids: Sequence, y_true: Sequence,
                      y_score: Optional[Sequence] = None,
                      y_pred: Optional[Sequence] = None,
                      subject_ids: Optional[Sequence] = None) -> str:
    """Write one method's held-out test predictions to a CSV and return its path.

    The file is ``{pred_root}/{task}/{method}/{generator}.csv`` (one file per generator arm)
    with columns ``patient_id, y_true, y_score, y_pred`` (plus ``subject_id`` when supplied, so
    an evaluator can resample at the patient level rather than the admission level). These
    files let other methods be scored on exactly the same patients as CLIP-H, under one
    protocol, and make paired comparisons possible.

    ``y_score`` may be None for a method that only emits a hard label; it is then written as
    NaN, and AUROC for that method should be reported as not available rather than computed
    from hard labels. When ``y_pred`` is None it is ``y_score >= 0.5``.
    """
    y_true = np.asarray(y_true).astype(int)
    n = len(y_true)
    if y_score is None and y_pred is None:
        raise ValueError("write_predictions needs at least one of y_score / y_pred")
    if y_pred is None:
        y_pred = (np.asarray(y_score, dtype=float) >= 0.5).astype(int)
    cols = {"patient_id": list(patient_ids), "y_true": y_true,
            "y_score": (np.asarray(y_score, dtype=float) if y_score is not None
                        else np.full(n, np.nan)),
            "y_pred": np.asarray(y_pred).astype(int)}
    if subject_ids is not None:
        cols["subject_id"] = list(subject_ids)
    for k, v in cols.items():
        if len(v) != n:
            raise ValueError(f"write_predictions: column {k} has length {len(v)}, expected {n}")
    out_dir = os.path.join(pred_root, str(task), str(method))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{generator}.csv")
    pd.DataFrame(cols).to_csv(path, index=False)
    return path


def record_threshold(pred_root: str, task: str, key: str, threshold: float) -> str:
    """Record a method's decision threshold in ``{pred_root}/{task}/thresholds.json``.

    The file maps a method key to the threshold that produced its ``y_pred`` in
    :func:`write_predictions`, so F1 can be recomputed at each method's own operating point
    rather than at a common 0.5. Merges into any existing file so methods can be recorded in
    any order. A non-finite threshold (the prevalence-matched operating point has none) is
    stored as JSON null.
    """
    os.makedirs(os.path.join(pred_root, str(task)), exist_ok=True)
    path = os.path.join(pred_root, str(task), "thresholds.json")
    data: Dict[str, float] = {}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            data = {}
    # NaN is the expected value under the prevalence-matched operating point (top-k has no single
    # threshold), and json.dump would emit a bare `NaN` token, which Python accepts but other JSON
    # parsers reject. Store null, as write_manifest does.
    _t = float(threshold)
    data[str(key)] = _t if _t == _t and abs(_t) != float("inf") else None
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    return path


def build_lineage(candidates: Sequence[Dict], screen_status: Optional[Dict] = None,
                  dedup: Optional[Dict] = None, gate_status: Optional[Dict] = None,
                  coefs: Optional[Dict] = None) -> List[Dict]:
    """Reconcile the per-stage outcomes into one lineage row per screened candidate, so it is
    unambiguous what became of every hypothesis end to end.

    ``candidates``   : the concepts that entered SCREEN, as dicts with at least ``hypothesis``
                       (optional: ``origin_neuron_id``, ``raw_hypothesis``, ``provenance``).
    ``screen_status``: ``{hypothesis: 'kept'|'dropped:<reason>'}`` (absent -> 'kept').
    ``dedup``        : the ``predictive_dedup`` result (uses ``source_map`` + ``kept``).
    ``gate_status``  : ``{rep_label: 'certified'|'dropped:<criterion>'}``. When the dict is empty the
                       gate is treated as not run (everything reaching it is 'certified'); when it is
                       non-empty, a hypothesis it does not name is 'unresolved', never 'certified'.
    ``coefs``        : ``{rep_label: float}`` final logistic coefficients.

    Each row carries exactly one ``final_status`` in {certified, merged_into:<rep>, dropped@SCREEN,
    dropped@DEDUP, dropped@GATE, unresolved}. Merged rows point at a representative that is present
    in ``dedup['kept']``; ``dropped@DEDUP`` rows are the complement-dropped concepts
    (``dedup['dropped_complement']`` = ``{dropped: kept}``). A hypothesis is reported as
    certified only when the gate certified it (or the gate was not run), so concepts that never
    reached the gate are not counted as certified.
    """
    screen_status = screen_status or {}
    gate_status = gate_status or {}
    coefs = coefs or {}
    source_map = (dedup or {}).get("source_map", {}) or {}
    dropped_complement = (dedup or {}).get("dropped_complement", {}) or {}
    member_rep: Dict[str, str] = {}
    for rep, members in source_map.items():
        for m in members:
            member_rep[m] = rep

    rows: List[Dict] = []
    for c in candidates:
        hyp = str(c.get("hypothesis", ""))
        ss = str(screen_status.get(hyp, "kept"))
        rep = member_rep.get(hyp, hyp)
        row = {"origin_neuron_id": c.get("origin_neuron_id"),
               "raw_hypothesis": c.get("raw_hypothesis", hyp),
               "provenance": c.get("provenance", "neuron" if c.get("origin_neuron_id") is not None else "unknown"),
               "hypothesis": hyp, "screen_outcome": "kept" if not ss.startswith("dropped") else ss,
               "dedup_representative": rep, "final_status": None,
               "reason": "", "final_coefficient": None}
        if ss.startswith("dropped"):
            row["final_status"] = "dropped@SCREEN"
            row["reason"] = ss.split(":", 1)[1] if ":" in ss else ""
        elif hyp in dropped_complement:
            row["final_status"] = "dropped@DEDUP"
            row["reason"] = f"complement_of:{dropped_complement[hyp]}"
            row["dedup_representative"] = dropped_complement[hyp]
        elif rep != hyp:
            row["final_status"] = f"merged_into:{rep}"
            row["reason"] = "firing-vector duplicate"
        else:
            gs = gate_status.get(hyp)
            if gs is None:
                gs = "certified" if not gate_status else "unresolved"
            gs = str(gs)
            if gs.startswith("dropped"):
                row["final_status"] = "dropped@GATE"
                row["reason"] = gs.split(":", 1)[1] if ":" in gs else ""
            elif gs == "unresolved":
                row["final_status"] = "unresolved"
                row["reason"] = "not present in gate input"
            else:
                row["final_status"] = "certified"
                row["final_coefficient"] = coefs.get(hyp)
        rows.append(row)
    return rows


def lineage_candidates_from_stage_logs(run_dir: str) -> List[Dict]:
    """Build the ``candidates`` list for :func:`build_lineage` from a run dir's stage logs, carrying
    neuron provenance where it is recoverable.

    ``stage_screen.csv`` gives the concepts that entered SCREEN; ``stage_generate.csv`` gives the
    (neuron_idx, interpretation) pairs the interpreter produced. Self-refinement rewrites text in a
    single set-to-set LLM call and does not record which output came from which input, so a
    neuron id can be attached only when the screened text is byte-identical to a generated
    interpretation. Every row gets a ``provenance`` field: ``'neuron'`` (origin_neuron_id set) or
    ``'refined'`` (rewritten by self-refinement; parent not recorded). If several neurons produced
    the same text (exact duplicates are collapsed before refine), all ids are recorded, ``|``-joined.
    """
    import pandas as _pd
    screen = _pd.read_csv(os.path.join(run_dir, "stage_screen.csv"))
    gen_path = os.path.join(run_dir, "stage_generate.csv")
    by_text: Dict[str, List[str]] = {}
    if os.path.exists(gen_path):
        gen = _pd.read_csv(gen_path)
        for _, r in gen.iterrows():
            by_text.setdefault(str(r["hypothesis"]).strip(), []).append(str(r.get("neuron_idx")))
    out: List[Dict] = []
    for _, r in screen.iterrows():
        hyp = str(r["hypothesis"])
        ids = by_text.get(hyp.strip())
        out.append({"hypothesis": hyp,
                    "origin_neuron_id": "|".join(ids) if ids else None,
                    "raw_hypothesis": hyp if ids else None,
                    "provenance": "neuron" if ids else "refined"})
    return out


def two_stage_screen_vote(votes: Dict) -> Dict:
    """Two-stage self-consistency vote for one hypothesis (Phase 2.3): aggregate each model's
    votes, then combine across models. ``votes`` = ``{model: {axis: [vote, ...]}}``
    with axes ``direction, magnitude, validity, demographic, leakage, coherent``.

    Within a model: majority for categorical axes, median for validity/leakage.
    Across models: direction = the shared call if all models agree, else 'unclear' (and
    ``critic_disagreement`` is set); validity = mean of per-model medians; leakage = max;
    demographic = OR; magnitude = the shared call if all agree, else the weakest (lowest
    ordinal); coherent = AND. ``critic_confidence`` blends within-model vote agreement with
    cross-model agreement. When no model produced a direction vote the result fails closed
    (a verdict that drops the hypothesis, with ``screen_failed`` True). Pure.
    """
    from collections import Counter
    models = [m for m in votes if votes[m].get("direction")]
    if not models:
        # Fail closed. No judge produced a parseable vote, so nothing is known about this
        # hypothesis. A permissive default (e.g. validity=2, leakage=0, coherent=True) would let
        # an API outage or a run of malformed JSON quietly admit unscreened concepts with a
        # middling validity that clears every downstream threshold. Return the verdict that
        # drops the hypothesis and mark it so the caller can count the failures.
        return {"expected_direction": "unclear", "expected_magnitude": "moderate",
                "clinical_validity": 0, "is_demographic": False, "leakage_risk": 3,
                "coherent": False, "critic_confidence": 0.0, "critic_disagreement": False,
                "screen_failed": True}

    def _maj(xs, default):
        xs = list(xs)
        return Counter(xs).most_common(1)[0][0] if xs else default

    def _med(xs, default):
        xs = [float(x) for x in xs]
        return float(np.median(xs)) if xs else default

    per, dir_frac = {}, []
    for m in models:
        v = votes[m]
        d = [str(x).lower() for x in v.get("direction", [])]
        win = _maj(d, "unclear")
        dir_frac.append((d.count(win) / len(d)) if d else 0.0)
        per[m] = {
            "direction": win,
            "magnitude": _maj([str(x).lower() for x in v.get("magnitude", [])], "moderate"),
            "validity": _med(v.get("validity", []), 2.0),
            "demographic": _maj([bool(x) for x in v.get("demographic", [])], False),
            "leakage": _med(v.get("leakage", []), 0.0),
            "coherent": _maj([bool(x) for x in v.get("coherent", [])], True),
        }
    dirs = {per[m]["direction"] for m in models}
    agree = len(dirs) == 1
    mags = [per[m]["magnitude"] for m in models]
    magnitude = mags[0] if len(set(mags)) == 1 else min(mags, key=lambda mm: _MAG_ORDER.get(mm, 2))
    confidence = float(np.mean(dir_frac)) * (1.0 if agree else 0.5)
    return {
        "expected_direction": per[models[0]]["direction"] if agree else "unclear",
        "expected_magnitude": magnitude,
        "clinical_validity": int(round(float(np.mean([per[m]["validity"] for m in models])))),
        "is_demographic": any(per[m]["demographic"] for m in models),
        "leakage_risk": int(max(per[m]["leakage"] for m in models)),
        "coherent": all(per[m]["coherent"] for m in models),
        "critic_confidence": round(confidence, 3),
        "critic_disagreement": (not agree) and len(models) >= 2,
        "screen_failed": False,
    }


# ============================================================================ #
# Robustness: preflight (fail-fast) and per-phase validation tripwires
# ============================================================================ #
def preflight_config(cfg: Dict) -> "list[tuple]":
    """Validate the config and environment before any costly compute.

    Checks that data/<task>/ was written, the sample sizes, ``sae_k <= sae_m``, that ``fdr_q`` and
    ``stability_pi`` lie in [0, 1], and, when the critic needs the Claude judge, the
    Anthropic key and package. Returns a list of (level, message); raises RuntimeError if any
    are 'fatal'. Cheap and deterministic: the point is to stop a multi-hour run in seconds
    when something is misconfigured, rather than after the expensive work.
    """
    out = []
    def add(level, msg): out.append((level, msg))

    dd = cfg.get("data_dir")
    if not dd or not os.path.exists(os.path.join(dd, "manifest.json")):
        add("fatal", f"no manifest.json in {dd!r}: write data/<task>/ first with "
                     "mimic_cleaning/run/run_<task>.py")
    for k in ("n_train", "n_test"):
        v = cfg.get(k)
        if v is not None and (not isinstance(v, int) or v <= 0):
            add("fatal", f"{k} must be a positive int or None (got {v!r})")
    if cfg.get("sae_k") and cfg.get("sae_m") and cfg["sae_k"] > cfg["sae_m"]:
        add("fatal", f"SAE_K ({cfg['sae_k']}) > SAE_M ({cfg['sae_m']})")
    for k in ("fdr_q", "stability_pi"):
        v = cfg.get(k)
        if v is not None and not (0.0 <= float(v) <= 1.0):
            add("fatal", f"{k} must be in [0,1] (got {v})")
    if str(cfg.get("critic_provider")) in ("anthropic", "both"):
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key or "..." in key:
            add("warn", "CRITIC_PROVIDER needs the Claude judge but ANTHROPIC_API_KEY is not set; "
                        "set it, or set CRITIC_PROVIDER='openai'.")
        try:
            import anthropic  # noqa: F401
        except ImportError:
            add("warn", "the anthropic package is not installed; run pip install anthropic, "
                        "or set CRITIC_PROVIDER='openai'.")

    fatal = [m for lvl, m in out if lvl == "fatal"]
    if fatal:
        raise RuntimeError("Preflight failed (fix before running):\n  - " + "\n  - ".join(fatal))
    return out


def ping_model(model: str, timeout: float = 120.0, seed: int = 0) -> "tuple[bool, str]":
    """Tiny live API call to verify that a model is reachable and authorized (a few tokens).

    Used in preflight so access, auth and parameter errors (e.g. no access to the model, a bad
    key, a reasoning model rejecting ``seed``, a missing anthropic package) surface in seconds
    rather than after hours of embedding and annotation. ``seed`` is passed so the ping
    exercises the same parameter surface as the real interpreter and critic calls
    (``get_completion`` adds reasoning_effort and the token floor for reasoning models
    automatically). Returns ``(ok, message)`` and never raises.
    """
    try:
        from hypothesaes.llm_api import get_completion
        txt = get_completion(prompt="Reply with exactly: ok", model=model,
                             max_tokens=5, timeout=timeout, seed=seed)
        return True, f"{model}: reachable ({str(txt).strip()[:20]!r})"
    except Exception as e:  # noqa: BLE001 (any failure here is a preflight signal)
        return False, f"{model}: {type(e).__name__}: {str(e)[:160]}"


def validate_embeddings(emb, n_expected: int) -> bool:
    """Tripwire after embedding: shape + finite-value check."""
    arr = np.asarray(emb)
    if arr.size == 0:
        raise ValueError("embeddings are empty")
    if len(arr) != n_expected:
        raise ValueError(f"embedding count {len(arr)} != expected {n_expected}")
    if not np.isfinite(arr).all():
        raise ValueError("embeddings contain NaN/Inf")
    return True


def validate_hypotheses(df: pd.DataFrame, min_n: int = 1) -> bool:
    """Tripwire after generation: non-empty + has the interpretation column."""
    if df is None or len(df) < min_n:
        raise ValueError(f"only {0 if df is None else len(df)} hypotheses generated (< {min_n})")
    if "interpretation" not in df.columns:
        raise ValueError("hypotheses frame missing 'interpretation' column")
    return True


# ============================================================================ #
# Bookkeeping: config hash, run dir, manifest
# ============================================================================ #
def config_hash(cfg: Dict, length: int = 10) -> str:
    """Stable short hash of a config dict (order-independent)."""
    payload = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def run_dir(base_dir: str, task: str, cfg: Dict, seed: int) -> str:
    """Return (and create) the per-run directory keyed by config hash + seed."""
    d = os.path.join(base_dir, f"{task}_{config_hash(cfg)}_seed{seed}")
    os.makedirs(d, exist_ok=True)
    return d


def write_manifest(run_directory: str, manifest: Dict) -> str:
    """Write the reproducibility manifest for a run to ``manifest.json`` and return its path.

    The manifest records the Python version, platform and key package versions, merged with
    ``manifest``. NumPy values become JSON numbers and non-finite floats become null.
    """
    import platform
    import sys

    import importlib.metadata as _imeta
    pkg_versions = {}
    for mod in ("numpy", "pandas", "scikit-learn", "scipy", "xgboost", "torch", "openai"):
        try:
            pkg_versions[mod] = _imeta.version(mod)
        except Exception:
            pkg_versions[mod] = None
    full = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": pkg_versions,
        **manifest,
    }
    def _jsonable(o):
        """NumPy scalars and arrays become JSON values; everything else becomes str.

        A bare ``default=str`` would turn every np.float64 into a quoted string, so a value that
        reached the manifest without an explicit ``float()`` cast (e.g. from a DataFrame's
        ``to_dict('records')``) would silently stop being comparable, plottable or
        arithmetic-safe for downstream readers.
        """
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            v = float(o)
            return v if v == v and abs(v) != np.inf else None   # NaN/inf are not valid JSON
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    def _clean(o):
        """Replace non-finite floats with None throughout.

        ``json.dump`` emits bare ``NaN``/``Infinity`` tokens for Python floats, which Python's
        own loader accepts but which are not valid JSON (RFC 8259), so other readers would
        reject the manifest. The ``default=`` hook cannot fix this: it only fires for objects the
        encoder cannot serialise natively, and a Python float always can. Thresholds are NaN
        under the prevalence-matched operating point, so this case is routine.
        """
        if isinstance(o, float):
            return o if o == o and abs(o) != float("inf") else None
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        return o

    path = os.path.join(run_directory, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clean(json.loads(json.dumps(full, default=_jsonable))), f, indent=2)
    return path
