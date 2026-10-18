#!/usr/bin/env python
"""Feature baselines (TF-IDF + logistic regression, structured logistic regression, XGBoost) with
bootstrap confidence intervals.

`clip_h_lib.run_feature_baselines` calls this module to produce the feature-baseline rows of the
results table. Data are passed in as arguments, and each task has a leakage rule in
``LEAK_RULES``.

    from baselines_ci import fit_baselines, score_with_ci, LEAK_RULES

    probs = fit_baselines(tr_txt, y_tr, tr_feats, te_txt, te_feats, LEAK_RULES["mortality"])
    for name, p in probs.items():
        r = score_with_ci(y_te, p)
        print(f"{name}  AUROC {r['auroc']:.2f} (±{r['auroc_pm']:.2f})  F1 {r['f1']:.2f} (±{r['f1_pm']:.2f})")

Conventions: seed 42, 10,000 bootstrap draws, 95% percentile intervals ([2.5, 97.5]), encoders
fit on the training split only.

Three choices differ from a naive implementation, and each affects the results:
  * F1 is taken at a prevalence-matched operating point (top-k, k = number of true positives),
    not at 0.5 and not at a train-tuned threshold. The rule needs no calibration, so it applies
    equally to a logistic regression and to an LLM arm whose confidences are coarse
    (0.7/0.8/0.9).
  * structured_lr uses numeric and one-hot categorical columns, not numeric columns alone.
  * xgboost sets scale_pos_weight = neg/pos; without it, XGBoost tends to predict only the
    majority class on imbalanced tasks.
"""
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED, NBOOT = 42, 10000

# ======================================================================================
# Leakage rules: one per task, each returning True for a column that must not be used as a
# feature. A missing rule would let outcome-adjacent columns in unnoticed, so fit_baselines
# refuses to run without one. The test for each column: at the moment of prediction, would the
# clinician know this value?
# ======================================================================================
_IDS = {"idx", "id", "subject_id", "hadm_id", "y", "label"}

LEAK_RULES = {
    # The mortality and discharge cohorts are already restricted to MORTALITY_FEATURE_COLUMNS and
    # DISCHARGE_FEATURE_COLUMNS (mimic_cleaning/src_2/mimic_general.py), and the runner notebooks
    # assert that no excluded column is present, so in practice only the id and label columns are
    # removed here. The outcome-adjacent patterns are a second safeguard in case the feature
    # lists grow.
    "mortality": lambda c: (c.startswith("death") or c.startswith("dod")
                            or c.startswith("discharge") or c.startswith("los_") or c == "los"
                            or c in _IDS | {"expired", "mortality", "hospital_expire_flag",
                                            "deathtime", "dischtime"}),
    "discharge": lambda c: (c.startswith("discharge") or c.startswith("death")
                            or c.startswith("dod") or c.startswith("los_") or c == "los"
                            or c in _IDS | {"discharge_location", "dischtime", "deathtime",
                                            "hospital_expire_flag"}),
}

# Free-text columns. They reach the TF-IDF baseline through the profile text and are excluded
# from the tabular models, which keeps the two kinds of baseline distinct.
FREE_TEXT = {"medications", "diagnoses", "procedures", "culture_sites",
             "positive_culture_sites", "positive_organisms", "vitals", "blood_pressure",
             "chief_complaint"}
CATEG_NAMES = {"gender", "race", "marital_status", "insurance", "language",
               "admission_type", "admission_location", "service", "discharge_location",
               "curr_service"}   # the CLIP-H cohorts name the admitting service `curr_service`


# ======================================================================================
# Operating point + CI
# ======================================================================================
def prevalence_matched(y, prob, seed=SEED):
    """Label the k highest-scoring rows positive, where k is the number of true positives.

    The predicted positive rate therefore matches the base rate, and no threshold is tuned.
    Ties are broken by a seeded random permutation rather than by row order: LLM confidences are
    coarse (0.7/0.8/0.9), so many patients tie, and row-order tie-breaking on a frame sorted by
    outcome would give a model accuracy it did not earn.
    """
    y, prob = np.asarray(y), np.asarray(prob, dtype=float)
    k, n = int(y.sum()), len(prob)
    order = np.lexsort((np.random.default_rng(seed).permutation(n), -prob))
    pred = np.zeros(n, dtype=int)
    pred[order[:k]] = 1
    return pred


def score_with_ci(y, prob, pred=None, n_boot=NBOOT, seed=SEED):
    """AUROC and positive-class F1 with 95% percentile bootstrap CIs, on a 0-100 scale.

    ``pred`` defaults to the prevalence-matched operating point. Returns a dict with ``auroc``,
    ``auroc_ci``, ``auroc_pm``, ``f1``, ``f1_ci`` and ``f1_pm``; ``*_ci`` is the (low, high)
    interval and ``*_pm`` its half-width, the value shown in parentheses in the results table.
    Resamples that contain only one class are skipped, because AUROC is undefined on them.
    """
    y, prob = np.asarray(y), np.asarray(prob, dtype=float)
    pred = prevalence_matched(y, prob, seed) if pred is None else np.asarray(pred)
    rng = np.random.default_rng(seed)
    a, f = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        a.append(roc_auc_score(y[i], prob[i]))
        f.append(f1_score(y[i], pred[i]))
    alo, ahi = np.percentile(a, [2.5, 97.5]) * 100
    flo, fhi = np.percentile(f, [2.5, 97.5]) * 100
    return {"auroc": roc_auc_score(y, prob) * 100, "auroc_ci": (alo, ahi),
            "auroc_pm": (ahi - alo) / 2,
            "f1": f1_score(y, pred) * 100, "f1_ci": (flo, fhi), "f1_pm": (fhi - flo) / 2}


# ======================================================================================
# The three baselines
# ======================================================================================
def _parse_traj(v):
    """Reduce a lab trajectory such as '1.7@16h; 2.8*@21h; 3.4*@22h' to its last value (3.4).

    Values that are not trajectory strings are returned unchanged; an unparseable last entry
    gives NaN.
    """
    if not isinstance(v, str) or "@" not in v:
        return v
    m = re.search(r"(-?\d+\.?\d*)\*?@", v.split(";")[-1])
    return float(m.group(1)) if m else np.nan


def _roles(df, leak):
    """Assign each feature column a role: numeric, categorical or lab trajectory.

    Leaky columns (per ``leak``), free-text columns and ``*trajectory*`` columns are skipped, as
    are non-numeric columns that are neither in ``CATEG_NAMES`` nor have at most 50 distinct
    values. Roles are learned from the training features only and applied unchanged to the test
    features. Returns ``(numeric, categorical, traj)``; trajectory columns are also listed in
    ``numeric``.
    """
    numeric, categorical, traj = [], [], set()
    for c in df.columns:
        if leak(c) or c in FREE_TEXT or "trajectory" in c:
            continue
        col = df[c]
        if col.astype(str).str.contains("@").any():
            traj.add(c); numeric.append(c)
        elif pd.api.types.is_numeric_dtype(col):
            # A column with a numeric dtype is numeric regardless of coverage; the imputer handles
            # missing values. The coverage test below (at least half the values parse as numbers)
            # cannot tell "not a number" from "missing", so a sparse float lab such as
            # lab_albumin_first would otherwise fall through to the categorical branch and be
            # one-hot encoded.
            numeric.append(c)
        elif pd.to_numeric(col, errors="coerce").notna().mean() >= 0.5:
            numeric.append(c)
        elif c in CATEG_NAMES or col.nunique(dropna=True) <= 50:
            categorical.append(c)
    return numeric, categorical, traj


def _frame(df, numeric, categorical, traj):
    """Build the model input frame: numeric columns coerced to float (trajectories reduced to their
    last value) and categorical columns as strings, with missing values marked '__missing__'."""
    out = {c: pd.to_numeric(df[c].map(_parse_traj) if c in traj else df[c], errors="coerce")
           for c in numeric}
    out.update({c: df[c].astype("string").fillna("__missing__").astype(str)
                for c in categorical})
    return pd.DataFrame(out)


def fit_baselines(train_texts, y_train, train_features,
                  test_texts, test_features, leak, seed=SEED, verbose=True):
    """Fit all three baselines and return ``{name: test-set positive-class probabilities}``.

    The names are ``tfidf_lr`` (TF-IDF on ``train_texts``), ``structured_lr`` and ``xgboost``
    (both on ``train_features`` after ``leak`` and the column roles are applied). Encoders are fit
    on the training split and only applied to the test split, so imputation medians and the
    one-hot vocabulary never see test data. Test labels are not passed in; scoring is done
    separately by ``score_with_ci``. With ``verbose``, prints the structured feature counts.

    Raises ValueError if ``leak`` is None or the train or test inputs are not row-aligned.
    """
    if leak is None:
        raise ValueError("leak rule required (see LEAK_RULES). Without one, "
                         "outcome-adjacent columns leak silently.")
    if not (len(train_texts) == len(train_features) == len(y_train)):
        raise ValueError(f"train not row-aligned: {len(train_texts)}/"
                         f"{len(train_features)}/{len(y_train)}")
    if len(test_texts) != len(test_features):
        raise ValueError(f"test not row-aligned: {len(test_texts)}/{len(test_features)}")

    y_train = np.asarray(y_train)
    out = {}

    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    Xtr, Xte = vec.fit_transform(train_texts), vec.transform(test_texts)
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced",
                             random_state=seed).fit(Xtr, y_train)
    out["tfidf_lr"] = clf.predict_proba(Xte)[:, 1]

    numeric, categorical, traj = _roles(train_features, leak)
    ct = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler(with_mean=False))]), numeric),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant",
                                                fill_value="__missing__")),
                          ("oh", OneHotEncoder(handle_unknown="ignore",
                                               min_frequency=2))]), categorical),
    ], sparse_threshold=0.3)
    Xs_tr = ct.fit_transform(_frame(train_features, numeric, categorical, traj))
    Xs_te = ct.transform(_frame(test_features, numeric, categorical, traj))
    if verbose:
        print(f"  structured: {len(numeric)} numeric + {len(categorical)} categorical "
              f"-> {Xs_tr.shape[1]} encoded")

    lr = LogisticRegression(C=1.0, max_iter=5000, solver="liblinear",
                            class_weight="balanced", random_state=seed).fit(Xs_tr, y_train)
    out["structured_lr"] = lr.predict_proba(Xs_te)[:, 1]

    # scale_pos_weight = neg/pos is XGBoost's analogue of class_weight="balanced". Without it,
    # XGBoost tends to predict only the majority class on the imbalanced tasks.
    from xgboost import XGBClassifier
    spw = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1.0)
    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, eval_metric="logloss", random_state=seed,
                        n_jobs=4, scale_pos_weight=spw).fit(Xs_tr, y_train)
    out["xgboost"] = xgb.predict_proba(Xs_te)[:, 1]
    return out
