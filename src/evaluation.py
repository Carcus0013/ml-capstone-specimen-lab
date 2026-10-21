"""Metric functions for the three tracks.

All required metrics are computed here so the notebooks, saved result files and the
app report identical numbers.
"""
import time

import numpy as np
from sklearn.metrics import (
    accuracy_score, calinski_harabasz_score, davies_bouldin_score, f1_score,
    mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score,
    roc_auc_score, silhouette_score,
)


# ---------------------------------------------------------------------------
# Regression: R2 (higher is better), RMSE and MAE (lower is better)
# ---------------------------------------------------------------------------
def regression_metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def score_for_roc(model, X):
    """Continuous scores for ROC-AUC: probabilities if the model has them, otherwise
    decision-function values. Returns (scores, source) where scores has one column per
    class (or a 1-D vector for binary decision functions)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X), "predict_proba"
    if hasattr(model, "decision_function"):
        return model.decision_function(X), "decision_function"
    return None, "unavailable"


def roc_auc(y_true, scores, classes):
    """Binary: AUC of the positive class (classes[1]). Multiclass: One-vs-Rest, weighted."""
    if scores is None:
        return float("nan")
    classes = list(classes)
    if len(classes) == 2:
        pos_scores = scores[:, 1] if np.ndim(scores) == 2 else scores
        return roc_auc_score(np.asarray(y_true) == classes[1], pos_scores)
    return roc_auc_score(y_true, scores, multi_class="ovr", average="weighted", labels=classes)


def classification_metrics(y_true, y_pred, scores, classes):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1 (weighted)": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "ROC-AUC": roc_auc(y_true, scores, classes),
    }


# ---------------------------------------------------------------------------
# Clustering: Silhouette (higher better, -1..1), Davies-Bouldin (lower better, >=0),
# Calinski-Harabasz (higher better, >=0)
# ---------------------------------------------------------------------------
def clustering_metrics(X, labels, silhouette_sample=None, random_state=42):
    labels = np.asarray(labels)
    n_clusters = len(np.unique(labels))
    if n_clusters < 2 or n_clusters >= len(labels):
        # All three indices are mathematically undefined in these cases.
        return {"Silhouette": float("nan"), "Davies-Bouldin": float("nan"),
                "Calinski-Harabasz": float("nan"), "n_clusters": n_clusters,
                "note": "undefined: need 2 <= n_clusters < n_samples"}
    return {
        "Silhouette": float(silhouette_score(X, labels, sample_size=silhouette_sample,
                                             random_state=random_state)),
        "Davies-Bouldin": float(davies_bouldin_score(X, labels)),
        "Calinski-Harabasz": float(calinski_harabasz_score(X, labels)),
        "n_clusters": n_clusters,
        "note": "",
    }


# ---------------------------------------------------------------------------
# Practical costs
# ---------------------------------------------------------------------------
def prediction_latency_ms(model, X, repeats=3):
    """Median milliseconds per 1000 predictions (measured, machine-dependent)."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict(X)
        times.append(time.perf_counter() - start)
    return float(np.median(times) / len(X) * 1000 * 1000)
