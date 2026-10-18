# Fits the feature baselines (TF-IDF + LR, structured LR, XGBoost) for one of the five tasks and scores them on the test split. No API calls.
"""Feature baselines for one task, on the same train/test split as CLIP-H.

    python run_baselines.py readmission        # or hosp_los3, icu_los4, mortality, discharge

Reads ../data/<task>/ (written by mimic_cleaning/run/run_<task>.py) and writes
results/<task>/summary.csv and results/<task>/<baseline>/stats.json.

    tfidf_lr        logistic regression on TF-IDF of the profile text
    structured_lr   logistic regression on the feature columns (numeric + one-hot categorical)
    xgboost         gradient boosting on the same feature columns

The models, column rules and scoring are those of clip_h/mortality/mimic_eval/baselines_ci.py,
which the mortality and discharge runner notebooks also call in their Evaluation cell, so all
five tasks' rows come from one method. For mortality and discharge the feature columns are read
from <split>_cohort.parquet, the frame those runners pass to baselines_ci, so this script gives
the same numbers as the runners. Test labels are used only to score and to place the operating
point, never to fit.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# =============================================================================
# SETTINGS
# =============================================================================
TASK = sys.argv[1] if len(sys.argv) > 1 else "readmission"   # readmission | hosp_los3 | icu_los4 | mortality | discharge
SEED = 42                  # tie-break at the operating point, bootstrap, model seeds
NBOOT = 10000              # bootstrap draws for the 95% intervals

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / TASK
OUT = HERE / "results" / TASK

# =============================================================================
# Column rules
# =============================================================================
# Leakage: which columns must not be features depends on when the prediction is made.
#   readmission  predicted at discharge, so discharge and LOS fields are known; only identifiers
#                and the label are dropped.
#   hosp_los3 /  predicted from the first 24 hours, so anything los*/discharge* would leak the
#   icu_los4     answer.
#   mortality /  predicted at admission. Their cohorts are already restricted to the task's feature
#   discharge    columns (mimic_cleaning/src_2/mimic_general.py), so in practice only the IDs and
#                the label are dropped; the death/discharge/LOS patterns are a second safeguard.
#                These two rules are the ones in clip_h/mortality/mimic_eval/baselines_ci.py.
_IDS = {"idx", "id", "subject_id", "hadm_id", "y", "label"}

LEAK_RULES = {
    "readmission": lambda c: c in _IDS | {"readmitted"},
    "hosp_los3": lambda c: (c.startswith("los_") or c == "los" or c.startswith("discharge")
                            or c in _IDS | {"los_gt"}),
    "icu_los4": lambda c: (c.startswith("los_") or c == "los" or c.startswith("discharge")
                           or c in _IDS | {"los_gt"}),
    "mortality": lambda c: (c.startswith("death") or c.startswith("dod")
                            or c.startswith("discharge") or c.startswith("los_") or c == "los"
                            or c in _IDS | {"expired", "mortality", "hospital_expire_flag",
                                            "deathtime", "dischtime"}),
    "discharge": lambda c: (c.startswith("discharge") or c.startswith("death")
                            or c.startswith("dod") or c.startswith("los_") or c == "los"
                            or c in _IDS | {"discharge_location", "dischtime", "deathtime",
                                            "hospital_expire_flag"}),
}

# Tasks whose feature frame is <split>_cohort.parquet (IDs, label and features with their stored
# types), as their runner notebooks read it; the other tasks use <split>_features.csv.
PARQUET_TASKS = {"mortality", "discharge"}

# Free-text columns reach TF-IDF through the profile text and are kept out of the tabular models.
FREE_TEXT = {"medications", "diagnoses", "procedures", "culture_sites", "positive_culture_sites",
             "positive_organisms", "vitals", "blood_pressure", "chief_complaint"}

# Text columns always one-hot encoded, whatever their number of distinct values.
CATEG_NAMES = {"gender", "race", "marital_status", "insurance", "language", "admission_type",
               "admission_location", "service", "discharge_location", "curr_service"}

if TASK not in LEAK_RULES:
    raise SystemExit(f"Unknown task {TASK!r}. Use one of: {', '.join(LEAK_RULES)}")
LEAK = LEAK_RULES[TASK]


# =============================================================================
# Data
# =============================================================================
def load(split):
    """Profile texts, labels and feature frame of one split; the files are row-aligned."""
    path = DATA / f"{split}.jsonl"
    if not path.exists():
        raise SystemExit(f"No {path}. Build it first: python mimic_cleaning/run/run_<task>.py")
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    if TASK in PARQUET_TASKS:
        feats = pd.read_parquet(DATA / f"{split}_cohort.parquet")
    else:
        feats = pd.read_csv(DATA / f"{split}_features.csv", low_memory=False)
    assert len(feats) == len(rows), (
        f"{split}: {len(feats)} feature rows but {len(rows)} profiles; the files are not row-aligned")
    labels = np.array([int(r["label"]) for r in rows])
    if "y" in feats.columns:
        assert np.array_equal(feats["y"].to_numpy().astype(int), labels), (
            f"{split}: labels in {split}.jsonl and the feature file differ")
    return [r["profile"] for r in rows], labels, feats


# =============================================================================
# Feature frame
# =============================================================================
def parse_traj(v):
    """Lab trajectory such as '1.7@16h; 2.8*@21h; 3.4*@22h' -> its last value (3.4)."""
    if not isinstance(v, str) or "@" not in v:
        return v
    m = re.search(r"(-?\d+\.?\d*)\*?@", v.split(";")[-1])
    return float(m.group(1)) if m else np.nan


def detect_roles(df):
    """Sort the train columns into numeric / categorical / trajectory; applied to test unchanged."""
    numeric, categorical, traj = [], [], set()
    for c in df.columns:
        if LEAK(c) or c in FREE_TEXT or "trajectory" in c:
            continue
        col = df[c]
        if col.astype(str).str.contains("@").any():           # lab trajectory string
            traj.add(c)
            numeric.append(c)
        elif pd.api.types.is_numeric_dtype(col):              # numeric however sparse; imputed
            numeric.append(c)
        elif pd.to_numeric(col, errors="coerce").notna().mean() >= 0.5:
            numeric.append(c)                                 # mostly parses as a number
        elif c in CATEG_NAMES or col.nunique(dropna=True) <= 50:
            categorical.append(c)
        # else: high-cardinality text -> dropped
    return numeric, categorical, traj


def build_frame(df, numeric, categorical, traj):
    out = {c: pd.to_numeric(df[c].map(parse_traj) if c in traj else df[c], errors="coerce")
           for c in numeric}
    out.update({c: df[c].astype("string").fillna("__missing__").astype(str) for c in categorical})
    return pd.DataFrame(out)


# =============================================================================
# Scoring
# =============================================================================
def prevalence_matched_pred(y_true, scores):
    """Top-k scores positive, k = number of true positives (500 on the balanced test set).
    Ties at the cut break by a seeded permutation, not by row order."""
    y_true, scores = np.asarray(y_true), np.asarray(scores, dtype=float)
    k, n = int(y_true.sum()), len(scores)
    order = np.lexsort((np.random.default_rng(SEED).permutation(n), -scores))
    pred = np.zeros(n, dtype=int)
    pred[order[:k]] = 1
    return pred


def compute_metrics(y, pred, prob):
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "auroc": float(roc_auc_score(y, prob)),
        "auprc": float(average_precision_score(y, prob)),
        "neg_precision": float(p[0]), "neg_recall": float(r[0]), "neg_f1": float(f[0]),
        "pos_precision": float(p[1]), "pos_recall": float(r[1]), "pos_f1": float(f[1]),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "n_test": int(len(y)), "pos_rate_test": float(np.mean(y)),
    }


def boot_ci(y, prob, pred, n=NBOOT):
    """95% percentile bootstrap over test rows; resamples with one class are skipped."""
    rng = np.random.default_rng(SEED)
    ba, bf = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        ba.append(roc_auc_score(y[i], prob[i]))
        bf.append(f1_score(y[i], pred[i]))
    return np.percentile(ba, [2.5, 97.5]) * 100, np.percentile(bf, [2.5, 97.5]) * 100


def report(name, y, prob):
    """Score one baseline, write its stats.json, return its summary row."""
    pred = prevalence_matched_pred(y, prob)
    m = compute_metrics(y, pred, prob)
    (alo, ahi), (flo, fhi) = boot_ci(y, prob, pred)
    m.update(operating_point="prevalence-matched top-k (seeded tie-break)",
             n_predicted_positive=int(pred.sum()),
             auroc_ci95=[round(alo, 2), round(ahi, 2)], f1_pos_ci95=[round(flo, 2), round(fhi, 2)])
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "stats.json").write_text(json.dumps(m, indent=1))
    print(f"  {name:14s} AUROC {m['auroc']*100:6.2f} [{alo:.2f},{ahi:.2f}]   "
          f"F1pos {m['pos_f1']*100:6.2f} [{flo:.2f},{fhi:.2f}]")
    return dict(task=TASK, baseline=name, auroc=m["auroc"] * 100, auroc_lo=alo, auroc_hi=ahi,
                f1_pos=m["pos_f1"] * 100, f1_lo=flo, f1_hi=fhi)


# =============================================================================
def main():
    print(f"\n===== {TASK} =====   data: {DATA}")
    Xtr_txt, ytr, ftr = load("train")
    Xte_txt, yte, fte = load("test")
    print(f"  train={len(ytr)} (pos {ytr.mean():.3f})  test={len(yte)} (pos {yte.mean():.3f})")
    summary = []

    # 1) TF-IDF + LR on the profile text
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    Xtr, Xte = vec.fit_transform(Xtr_txt), vec.transform(Xte_txt)
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=SEED)
    clf.fit(Xtr, ytr)
    summary.append(report("tfidf_lr", yte, clf.predict_proba(Xte)[:, 1]))

    # 2+3) one structured matrix; the encoder is fit on train and only applied to test
    numeric, categorical, traj = detect_roles(ftr)
    ct = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler(with_mean=False))]), numeric),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="__missing__")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=2))]),
         categorical),
    ], sparse_threshold=0.3)
    Xs_tr = ct.fit_transform(build_frame(ftr, numeric, categorical, traj))
    Xs_te = ct.transform(build_frame(fte, numeric, categorical, traj))
    print(f"  structured: {len(numeric)} numeric + {len(categorical)} categorical "
          f"-> {Xs_tr.shape[1]} encoded")

    lr = LogisticRegression(C=1.0, max_iter=5000, solver="liblinear",
                            class_weight="balanced", random_state=SEED)
    lr.fit(Xs_tr, ytr)
    summary.append(report("structured_lr", yte, lr.predict_proba(Xs_te)[:, 1]))

    # scale_pos_weight = neg/pos is XGBoost's analogue of class_weight="balanced"
    spw = float((ytr == 0).sum()) / max(float((ytr == 1).sum()), 1.0)
    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, eval_metric="logloss", random_state=SEED,
                        n_jobs=4, scale_pos_weight=spw)
    xgb.fit(Xs_tr, ytr)
    summary.append(report("xgboost", yte, xgb.predict_proba(Xs_te)[:, 1]))

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False)
    print(f"\nsummary: {OUT / 'summary.csv'}")


if __name__ == "__main__":
    main()
