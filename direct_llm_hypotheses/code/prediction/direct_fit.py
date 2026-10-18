"""Phase 3A model of this baseline: cross-validated logistic regression on the annotations.

CLIP-H's ``evaluation.fit_and_score_heldout`` fits ``LogisticRegression(class_weight='balanced')``
with C = 1. This baseline picks C by cross-validation (``LogisticRegressionCV``), the fit behind
its reported numbers. Everything else (top-k F1 rule, metric names) matches CLIP-H, so the
two outputs read the same and ``evaluation.bootstrap_intervals`` applies to both.
"""
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def gate_train_rows(y_train, n, seed):
    """Train rows the prediction model is fit on: all of them when ``n`` is None or covers the set,
    else a seeded pick of ``n`` that keeps the label rate (the pick of the 1,000-row mortality and
    discharge runs)."""
    y = np.asarray(y_train, dtype=int)
    if n is None or n >= len(y):
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    n_pos = min(len(pos), int(round(len(pos) * n / len(y))))
    n_neg = min(len(neg), n - n_pos)
    return np.sort(np.concatenate([rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)]))


def fit_and_score_heldout_cv(X_train, y_train, X_test, y_test, seed):
    """Fit on TRAIN annotations, score TEST. Returns ``(metrics dict, test scores, coefficients)``."""
    X_train, X_test = np.asarray(X_train, dtype=float), np.asarray(X_test, dtype=float)
    y_train, y_test = np.asarray(y_train, dtype=int), np.asarray(y_test, dtype=int)
    cv = max(2, min(5, int(np.bincount(y_train).min())))
    model = LogisticRegressionCV(Cs=10, cv=cv, max_iter=4000, class_weight="balanced",
                                 scoring="roc_auc", random_state=seed)
    model.fit(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]

    k, n = int(y_test.sum()), len(scores)
    order = np.lexsort((np.random.default_rng(seed).permutation(n), -scores))
    pred = np.zeros(n, dtype=int)
    pred[order[:k]] = 1
    metrics = {
        "auroc": float(roc_auc_score(y_test, scores)),
        "auprc": float(average_precision_score(y_test, scores)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "operating_point": "prevalence-matched top-k (seeded tie-break)",
        "n_predicted_positive": int(pred.sum()),
        "prevalence": float(y_test.mean()),
        "n_train_fit": int(len(y_train)), "n_test": int(len(y_test)),
        "C_selected": float(model.C_[0]),
        "model": "LogisticRegressionCV(class_weight='balanced', scoring='roc_auc') on hypothesis annotations; fit TRAIN, score TEST",
    }
    return metrics, scores, model.coef_[0].copy()
