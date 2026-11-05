"""Classifier factory and helpers shared by training, evaluation and inference."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

from . import config

MODEL_NAMES = {
    "lr": "Logistic Regression",
    "svm": "Linear SVM",
    "nb": "Complement Naive Bayes",
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "lgbm": "LightGBM",
}
CORE_MODELS = ("lr", "svm", "nb", "rf")
EXTRA_MODELS = ("xgb", "lgbm")

# Grids for --tune (classifier parameters only; the vectoriser is fitted once on the train split)
PARAM_GRIDS = {
    "lr": {"C": [1, 5, 10, 30]},
    "svm": {"C": [0.1, 0.3, 0.5, 1.0]},
    "nb": {"alpha": [0.05, 0.1, 0.3, 1.0]},
    "rf": {"n_estimators": [100, 200], "min_samples_leaf": [1, 2]},
    "xgb": {"max_depth": [4, 6], "learning_rate": [0.1, 0.2]},
    "lgbm": {"num_leaves": [31, 63], "learning_rate": [0.05, 0.1]},
}


def build_classifier(kind: str, seed: int = config.RANDOM_STATE):
    if kind == "lr":
        return LogisticRegression(C=10.0, max_iter=1000, random_state=seed)
    if kind == "svm":
        return LinearSVC(C=0.5, random_state=seed)
    if kind == "nb":
        return ComplementNB(alpha=0.3)
    if kind == "rf":
        return RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed)
    if kind == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError as err:
            raise ImportError("XGBoost is not installed: pip install xgboost") from err
        return XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.2, tree_method="hist",
            n_jobs=-1, random_state=seed, eval_metric="mlogloss",
        )
    if kind == "lgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as err:
            raise ImportError("LightGBM is not installed: pip install lightgbm") from err
        return LGBMClassifier(
            n_estimators=300, learning_rate=0.1, num_leaves=63, n_jobs=-1, random_state=seed,
            verbose=-1,
        )
    raise ValueError(f"Unknown model '{kind}'. Choose from {tuple(MODEL_NAMES)}.")


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def class_probabilities(clf, X) -> np.ndarray:
    """(n_samples, n_classes) probabilities. Margin classifiers (SVM) use a softmax of the margins,
    which is a *relative* score, not a calibrated probability."""
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)
    return _softmax(clf.decision_function(X))


def top_features_per_class(clf, terms: np.ndarray, n: int = 12) -> dict[int, list[tuple[str, float]]] | None:
    """Most indicative terms for every class (linear models and Naive Bayes). Returns ``None`` for
    tree ensembles, whose importances are global rather than per class."""
    if hasattr(clf, "coef_"):
        weights = clf.coef_
    elif hasattr(clf, "feature_log_prob_"):
        logp = clf.feature_log_prob_
        weights = logp - logp.mean(axis=0, keepdims=True)
    else:
        return None
    out = {}
    for k in range(weights.shape[0]):
        idx = np.argsort(weights[k])[::-1][:n]
        out[k] = [(str(terms[i]), float(weights[k][i])) for i in idx]
    return out
