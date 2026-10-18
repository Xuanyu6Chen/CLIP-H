# Portions adapted from HypotheSAEs v0.2.0 (https://github.com/rmovva/HypotheSAEs),
# Apache License 2.0. See THIRD_PARTY_NOTICES.md at the repository root.
"""Statistics on the 0/1 annotations of the hypotheses. No LLM call happens here.

    compute_hypothesis_association_tests   2x2 test of each hypothesis against the outcome, with
                                           Benjamini-Hochberg FDR (Phase 2E gate, Phase 3A)
    fit_and_score_heldout                  the reported result: logistic regression fit on the
                                           TRAIN annotations, scored on TEST (Phase 3A)
    bootstrap_intervals                    95% intervals of that result
    score_hypotheses                       per-hypothesis coefficients and separation scores from a
                                           model fit on the test set itself (describes the
                                           hypotheses; not the reported result)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, recall_score, precision_score,
)
from scipy.stats import fisher_exact, norm, pearsonr, ttest_ind
import statsmodels.api as sm


ASSOCIATION_TEST_COLUMNS = [
    'hypothesis',
    'n_total',
    'n_positive',
    'n_negative',
    'n_present',
    'n_absent',
    'a_present_positive',
    'b_present_negative',
    'c_absent_positive',
    'd_absent_negative',
    'prevalence',
    'positive_rate_when_present',
    'positive_rate_when_absent',
    'risk_difference',
    'odds_ratio',
    'log_odds_ratio',
    'or_ci_low',
    'or_ci_high',
    'p_value',
    'test_used',
    'q_value_fdr',
    'significant_fdr',
    'sparse_warning',
]


def _validate_binary_vector(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError(f'{name} must be a 1D vector; got shape {values.shape}')
    unique = set(np.unique(values).tolist())
    if not unique.issubset({0, 1, False, True}):
        raise ValueError(f'{name} must contain only binary 0/1 values; got {sorted(unique)}')
    return values.astype(int)


def _validate_association_inputs(
    annotation_matrix: np.ndarray,
    labels: np.ndarray,
    hypotheses: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    annotation_matrix = np.asarray(annotation_matrix)
    if annotation_matrix.ndim != 2:
        raise ValueError(f'annotation_matrix must be 2D; got shape {annotation_matrix.shape}')
    if annotation_matrix.size == 0:
        raise ValueError('annotation_matrix must not be empty')
    matrix_unique = set(np.unique(annotation_matrix).tolist())
    if not matrix_unique.issubset({0, 1, False, True}):
        raise ValueError(
            'annotation_matrix must contain only binary 0/1 values; '
            f'got {sorted(matrix_unique)}'
        )

    labels = _validate_binary_vector(labels, 'labels')
    hypotheses = [str(h) for h in hypotheses]
    if annotation_matrix.shape[0] != len(labels):
        raise ValueError(
            'annotation_matrix row count must match labels length; '
            f'got {annotation_matrix.shape[0]} rows and {len(labels)} labels'
        )
    if annotation_matrix.shape[1] != len(hypotheses):
        raise ValueError(
            'annotation_matrix column count must match hypotheses length; '
            f'got {annotation_matrix.shape[1]} columns and {len(hypotheses)} hypotheses'
        )

    return annotation_matrix.astype(int), labels, hypotheses


def _benjamini_hochberg(p_values: List[float], alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    p_values = np.asarray(p_values, dtype=float)
    if len(p_values) == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)
    if np.any(~np.isfinite(p_values)):
        raise ValueError('p_values must be finite before FDR correction')

    n_tests = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]
    ranked_q = ranked * n_tests / np.arange(1, n_tests + 1)
    ranked_q = np.minimum.accumulate(ranked_q[::-1])[::-1]
    ranked_q = np.clip(ranked_q, 0, 1)

    q_values = np.empty_like(ranked_q)
    q_values[order] = ranked_q
    return (q_values <= alpha), q_values.astype(float)


def _two_proportion_ztest_p_value(a: int, n_present: int, c: int, n_absent: int) -> float:
    p_present = a / n_present
    p_absent = c / n_absent
    pooled = (a + c) / (n_present + n_absent)
    se = np.sqrt(pooled * (1 - pooled) * ((1 / n_present) + (1 / n_absent)))
    if se == 0:
        return 1.0 if p_present == p_absent else 0.0
    z_score = (p_present - p_absent) / se
    return float(2 * norm.sf(abs(z_score)))


def compute_hypothesis_association_tests(
    annotation_matrix: np.ndarray,
    labels: np.ndarray,
    hypotheses: List[str],
    fdr_alpha: float = 0.05,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    """
    Compute threshold-free per-hypothesis association tests from saved annotations.

    Each hypothesis column is reduced to a 2x2 table:
        present/absent x positive/negative label.
    The function performs Fisher's exact test for sparse tables and a two-sided
    two-proportion z-test otherwise, then applies Benjamini-Hochberg FDR across
    all hypothesis p-values. No LLM calls are made here.
    """
    if not (0 < fdr_alpha <= 1):
        raise ValueError(f'fdr_alpha must be in (0, 1]; got {fdr_alpha}')

    annotation_matrix, labels, hypotheses = _validate_association_inputs(
        annotation_matrix=annotation_matrix,
        labels=labels,
        hypotheses=hypotheses,
    )

    n_total = int(len(labels))
    n_positive = int(np.sum(labels == 1))
    n_negative = int(np.sum(labels == 0))
    rows = []
    p_values = []

    for col_idx, hypothesis in enumerate(hypotheses):
        present = annotation_matrix[:, col_idx] == 1
        absent = ~present

        a = int(np.sum(present & (labels == 1)))
        b = int(np.sum(present & (labels == 0)))
        c = int(np.sum(absent & (labels == 1)))
        d = int(np.sum(absent & (labels == 0)))
        table = np.array([[a, b], [c, d]], dtype=float)

        n_present = a + b
        n_absent = c + d
        prevalence = n_present / n_total
        present_rate = a / n_present if n_present else np.nan
        absent_rate = c / n_absent if n_absent else np.nan
        risk_difference = present_rate - absent_rate if n_present and n_absent else np.nan

        expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / n_total
        sparse_warning = bool(np.any(table == 0) or np.any(expected < 5))

        if sparse_warning:
            _, p_value = fisher_exact([[a, b], [c, d]], alternative='two-sided')
            test_used = 'fisher_exact'
            if not np.isfinite(p_value):
                p_value = 1.0
        else:
            p_value = _two_proportion_ztest_p_value(a, n_present, c, n_absent)
            test_used = 'two_proportion_ztest'

        # Haldane-Anscombe correction is only for effect-size stability.
        effect_cells = table + 0.5 if np.any(table == 0) else table
        ea, eb, ec, ed = effect_cells.ravel()
        odds_ratio = (ea * ed) / (eb * ec)
        log_odds_ratio = float(np.log(odds_ratio))
        se_log_or = float(np.sqrt((1 / ea) + (1 / eb) + (1 / ec) + (1 / ed)))
        or_ci_low = float(np.exp(log_odds_ratio - 1.96 * se_log_or))
        or_ci_high = float(np.exp(log_odds_ratio + 1.96 * se_log_or))

        p_values.append(float(p_value))
        rows.append({
            'hypothesis': hypothesis,
            'n_total': n_total,
            'n_positive': n_positive,
            'n_negative': n_negative,
            'n_present': int(n_present),
            'n_absent': int(n_absent),
            'a_present_positive': a,
            'b_present_negative': b,
            'c_absent_positive': c,
            'd_absent_negative': d,
            'prevalence': float(prevalence),
            'positive_rate_when_present': float(present_rate) if n_present else np.nan,
            'positive_rate_when_absent': float(absent_rate) if n_absent else np.nan,
            'risk_difference': float(risk_difference) if n_present and n_absent else np.nan,
            'odds_ratio': float(odds_ratio),
            'log_odds_ratio': log_odds_ratio,
            'or_ci_low': or_ci_low,
            'or_ci_high': or_ci_high,
            'p_value': float(p_value),
            'test_used': test_used,
            'sparse_warning': sparse_warning,
        })

    rejected, q_values = _benjamini_hochberg(p_values, alpha=fdr_alpha)
    association_df = pd.DataFrame(rows)
    association_df['q_value_fdr'] = q_values
    association_df['significant_fdr'] = rejected
    association_df = association_df[ASSOCIATION_TEST_COLUMNS]
    association_df = association_df.sort_values(
        ['q_value_fdr', 'p_value', 'risk_difference'],
        ascending=[True, True, False],
        na_position='last',
    ).reset_index(drop=True)

    test_counts = {
        str(test): int(count)
        for test, count in association_df['test_used'].value_counts().items()
    }
    association_summary = {
        'n_profiles': n_total,
        'n_hypotheses': int(len(hypotheses)),
        'n_positive': n_positive,
        'n_negative': n_negative,
        'fdr_alpha': float(fdr_alpha),
        'n_significant_fdr': int(association_df['significant_fdr'].sum()),
        'n_sparse_tests': int(association_df['sparse_warning'].sum()),
        'test_counts': test_counts,
        'output_method': '2x2_fisher_or_two_proportion_ztest_bh_fdr',
    }

    return association_summary, association_df


def compute_hypothesis_separation_scores(
    hypothesis_annotations: Dict[str, np.ndarray],
    y_true: np.ndarray
) -> Dict[str, Tuple[float, float]]:
    """
    The separation score of a hypothesis is the outcome rate of the profiles it is true for minus
    the outcome rate of the profiles it is not true for; the p-value is from a t-test between the
    two groups. ``hypothesis_annotations`` holds one 0/1 vector per hypothesis.

    Returns dict mapping hypothesis to (separation score, p_value).
    """
    results = {}
    for hypothesis, annotations in hypothesis_annotations.items():
        pos_vals = y_true[annotations == 1]
        neg_vals = y_true[annotations == 0]
        effect_size = np.mean(pos_vals) - np.mean(neg_vals)
        _, p_value = ttest_ind(pos_vals, neg_vals)
        results[hypothesis] = (effect_size, p_value)

    return results


def compute_ols_metrics(
    hypothesis_annotations: Dict[str, np.ndarray],
    y_true: np.ndarray,
    classification: bool = False,
    print_summary: bool = False
) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]]]:
    """
    Fit a single OLS/Logistic regression of the binary hypothesis-annotation
    matrix on the target, and report how well that combined model separates the
    classes.

    HOW EACH NUMBER IS PRODUCED (read before trusting the numbers)
    --------------------------------------------------------------
    Model: y ~ const + [hypothesis_1 ... hypothesis_k], one binary column per
    hypothesis. For classification this is an unregularized statsmodels Logit
    (MLE/Newton), no class_weight, intercept included.

    *** In-sample caveat ***: this model is fit AND predicted on the same rows
    (the rows in `hypothesis_annotations`; in the pipeline, the test set). Its
    AUROC/AUPRC are therefore optimistic upper bounds, not held-out results:
    they say how separable these rows are with this set of hypotheses. The
    held-out result of the pipeline comes from `fit_and_score_heldout`, which
    fits on the TRAIN annotations and scores TEST.

    Per-metric provenance (classification branch):
    - prevalence      : mean(y_true). The positive base rate; the AUPRC a random
                        or constant classifier would achieve.
    - auroc           : roc_auc_score(y_true, p). Threshold-FREE rank quality of
                        the predicted probability p. 0.5 = chance.
    - auprc           : average_precision_score(y_true, p). Threshold-FREE area
                        under precision-recall; compare against `prevalence`.
    - auprc_lift      : auprc / prevalence. >1 means real signal over base rate.
    - f1_at_0.5 / acc_at_0.5 / recall_at_0.5 / precision_at_0.5 / acc:
                        the hard 0.5 cut on p (`acc` = `acc_at_0.5`). On an
                        imbalanced target p rarely exceeds 0.5, so these go to ~0;
                        that is an effect of the cut, not of the hypotheses. Do
                        not report them as the F1 / recall of the set.
    - f1 / recall / precision:
                        seeded prevalence-matched top-k metrics. Exactly k scores
                        are predicted positive, where k is the observed positive
                        count. This is the shared rule used by Direct Hypotheses,
                        CLIP-H, and feature baselines.
    - r2              : McFadden pseudo-R^2 from the Logit fit (prsquared).

    Regression branch keeps r2 = squared Pearson corr of y_true vs predictions.

    Returns:
        Tuple of (metrics dict, {hypothesis: (coef, pval)} dict)
    """
    hypotheses = list(hypothesis_annotations.keys())
    # X columns are the per-hypothesis binary annotation vectors; const = intercept.
    X = np.array([hypothesis_annotations[h] for h in hypotheses]).T
    X = sm.add_constant(X)

    if classification:
        model = sm.Logit(y_true, X)
    else:
        model = sm.OLS(y_true, X)

    try:
        results = model.fit(disp=0)
    except Exception as e:
        print(f"Error fitting model: {e}, trying OLS instead")
        model = sm.OLS(y_true, X)
        results = model.fit()

    if print_summary:
        print(results.summary())

    metrics = {}
    # In-sample predicted probabilities (fit and predicted on the same X) -- see caveat.
    y_pred = results.predict(X)
    y_true = np.asarray(y_true).astype(int)

    if classification:
        prevalence = float(np.mean(y_true))
        auprc = float(average_precision_score(y_true, y_pred))
        # Headline, threshold-free numbers (the honest summary of separability).
        metrics['prevalence'] = prevalence
        metrics['auroc'] = float(roc_auc_score(y_true, y_pred))
        metrics['auprc'] = auprc
        metrics['auprc_lift'] = (auprc / prevalence) if prevalence > 0 else float('nan')

        # The hard 0.5 cut. Not the reported F1/recall (see the docstring).
        pred_05 = (y_pred > 0.5).astype(int)
        n_pred_pos_05 = int(pred_05.sum())
        metrics['acc_at_0.5'] = float(np.mean(pred_05 == y_true))
        metrics['f1_at_0.5'] = float(f1_score(y_true, pred_05, zero_division=0))
        metrics['recall_at_0.5'] = float(recall_score(y_true, pred_05, zero_division=0))
        # Guard divide-by-zero: precision is undefined when nothing is predicted positive.
        metrics['precision_at_0.5'] = float(precision_score(y_true, pred_05, zero_division=0))
        metrics['n_pred_pos_at_0.5'] = n_pred_pos_05

        # Shared F1 rule: seeded top-k, with k fixed to the observed positive count.
        # AUROC/AUPRC above continue to use y_pred directly and are unchanged.
        k, n = int(y_true.sum()), len(y_true)
        tie_break = np.random.default_rng(42).permutation(n)
        order = np.lexsort((tie_break, -np.asarray(y_pred, dtype=float)))
        pred_topk = np.zeros(n, dtype=int)
        pred_topk[order[:k]] = 1
        metrics['f1'] = float(f1_score(y_true, pred_topk, zero_division=0))
        metrics['recall'] = float(recall_score(y_true, pred_topk, zero_division=0))
        metrics['precision'] = float(precision_score(y_true, pred_topk, zero_division=0))
        metrics['operating_point'] = 'prevalence-matched top-k (seeded tie-break)'
        metrics['n_predicted_positive'] = int(pred_topk.sum())
        metrics['acc'] = metrics['acc_at_0.5']   # the same number under its short name

        if hasattr(results, 'prsquared'):
            metrics['r2'] = float(results.prsquared)
    else:
        r, _ = pearsonr(y_true, y_pred)
        metrics['r2'] = r**2

    # Per-hypothesis logistic coefficient and its p-value (skip the intercept).
    coefs_pvals = {
        hyp: (coef, pval)
        for hyp, coef, pval in zip(hypotheses, results.params[1:], results.pvalues[1:])
    }

    return metrics, coefs_pvals


def score_hypotheses(
    hypothesis_annotations: Dict[str, np.ndarray],
    y_true: np.ndarray,
    classification: bool = False,
    corrected_pval_threshold: float = 0.1,
    print_summary: bool = False,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Describe a set of hypotheses on the rows they were annotated on (in the pipeline: the test set).

    Fits one model on those rows (``compute_ols_metrics``) and computes each hypothesis'
    separation score. ``Significant`` in the metrics counts the hypotheses whose coefficient
    p-value is below ``corrected_pval_threshold`` / number of hypotheses (Bonferroni).

    Returns:
        Tuple of (metrics dict, DataFrame with one row per hypothesis, highest separation score first)
    """
    # Get OLS metrics
    metrics, coefs_pvals = compute_ols_metrics(
        hypothesis_annotations=hypothesis_annotations,
        y_true=y_true,
        classification=classification,
        print_summary=print_summary
    )

    # Compute count of significant hypotheses after Bonferroni correction
    corrected_p_value_threshold = corrected_pval_threshold / len(hypothesis_annotations)
    significant_hypotheses = [hyp for hyp, (_, pval) in coefs_pvals.items() if pval < corrected_p_value_threshold]
    metrics['Significant'] = (len(significant_hypotheses), len(hypothesis_annotations), corrected_p_value_threshold)
    
    # Create base dataframe
    hypothesis_df = pd.DataFrame({
        'hypothesis': list(coefs_pvals.keys()),
        'regression_coef': [coef for coef, _ in coefs_pvals.values()],
        'regression_pval': [pval for _, pval in coefs_pvals.values()]
    })
    
    # Add feature prevalence
    hypothesis_df['feature_prevalence'] = [
        np.mean(hypothesis_annotations[h] != 0)
        for h in hypothesis_df['hypothesis']
    ]
    
    # Add separation scores
    separation_scores = compute_hypothesis_separation_scores(
        hypothesis_annotations=hypothesis_annotations,
        y_true=y_true
    )
    hypothesis_df['separation_score'] = [
        separation_scores[h][0] for h in hypothesis_df['hypothesis']
    ]
    hypothesis_df['separation_pval'] = [
        separation_scores[h][1] for h in hypothesis_df['hypothesis']
    ]
    
    # Highest separation score first
    hypothesis_df = hypothesis_df[['hypothesis', 'separation_score', 'separation_pval', 'regression_coef', 'regression_pval', 'feature_prevalence']]
    hypothesis_df = hypothesis_df.sort_values('separation_score', ascending=False)
    
    return metrics, hypothesis_df


def fit_and_score_heldout(X_train, y_train, X_test, y_test, seed):
    """Fit a logistic regression on the training annotations and score the test set.

    The model is L2 ``LogisticRegression(class_weight='balanced')`` on the 0/1 annotation
    matrices. AUROC and AUPRC use the predicted probabilities. F1, precision and recall use
    one operating point: the k highest-scored test profiles are predicted positive, where k
    is the number of positives in the test set (ties broken by a seeded permutation).

    Returns ``(metrics dict, test scores, model coefficients)``; the coefficients are in the
    column order of the annotation matrices.
    """
    from sklearn.linear_model import LogisticRegression

    X_train, X_test = np.asarray(X_train, dtype=float), np.asarray(X_test, dtype=float)
    y_train, y_test = np.asarray(y_train, dtype=int), np.asarray(y_test, dtype=int)
    model = LogisticRegression(max_iter=2000, class_weight='balanced')
    model.fit(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]

    k, n = int(y_test.sum()), len(scores)
    tie_break = np.random.default_rng(seed).permutation(n)
    order = np.lexsort((tie_break, -scores))
    pred = np.zeros(n, dtype=int)
    pred[order[:k]] = 1
    metrics = {
        'auroc': float(roc_auc_score(y_test, scores)),
        'auprc': float(average_precision_score(y_test, scores)),
        'f1': float(f1_score(y_test, pred, zero_division=0)),
        'precision': float(precision_score(y_test, pred, zero_division=0)),
        'recall': float(recall_score(y_test, pred, zero_division=0)),
        'operating_point': 'prevalence-matched top-k (seeded tie-break)',
        'n_predicted_positive': int(pred.sum()),
        'prevalence': float(y_test.mean()),
        'n_train_fit': int(len(y_train)), 'n_test': int(len(y_test)),
        'model': "L2 LogisticRegression(class_weight='balanced') on concept annotations; fit TRAIN, score TEST",
    }
    return metrics, scores, model.coef_[0].copy()


def bootstrap_intervals(y_test, scores, seed, n_bootstrap=10000, level=0.95):
    """Percentile bootstrap intervals of the held-out metrics, from resampling the test set.

    The model is not refit: each resample draws test patients with replacement and recomputes
    AUROC, AUPRC and the top-k F1, precision and recall on their saved scores (k = positives in
    the resample). A resample with one class only is skipped.

    Returns ``{metric: [low, high]}`` plus ``'n_bootstrap'`` and ``'level'``.
    """
    y_test, scores = np.asarray(y_test, dtype=int), np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y_test)
    draws = {name: [] for name in ('auroc', 'auprc', 'f1', 'precision', 'recall')}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        y, s = y_test[idx], scores[idx]
        if y.min() == y.max():
            continue
        order = np.lexsort((rng.permutation(n), -s))
        pred = np.zeros(n, dtype=int)
        pred[order[:int(y.sum())]] = 1
        draws['auroc'].append(roc_auc_score(y, s))
        draws['auprc'].append(average_precision_score(y, s))
        draws['f1'].append(f1_score(y, pred, zero_division=0))
        draws['precision'].append(precision_score(y, pred, zero_division=0))
        draws['recall'].append(recall_score(y, pred, zero_division=0))
    tail = (1.0 - level) / 2.0 * 100.0
    out = {name: [float(np.percentile(v, tail)), float(np.percentile(v, 100.0 - tail))] for name, v in draws.items()}
    out.update(n_bootstrap=int(n_bootstrap), level=float(level))
    return out
