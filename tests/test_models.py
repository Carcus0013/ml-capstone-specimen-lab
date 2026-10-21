"""Saved pipelines, result files, figures and notebooks produced by the training workflow."""
import json

import joblib
import nbformat
import numpy as np
import pandas as pd
import pytest

REGRESSION_ALGOS = ["Linear Regression", "Ridge Regression", "Lasso Regression", "ElasticNet Regression",
                    "Polynomial Regression", "Decision Tree Regressor", "Random Forest Regressor",
                    "Gradient Boosting Regressor", "Support Vector Regressor", "K-Nearest Neighbors Regressor"]
CLASSIFICATION_ALGOS = ["Logistic Regression", "K-Nearest Neighbors", "Gaussian Naive Bayes", "Decision Tree",
                        "Support Vector Machine (SVC)", "Random Forest", "AdaBoost", "Gradient Boosting",
                        "Bagging (Decision Tree)", "MLP Neural Network"]


def _load_selected(root, track):
    sel = json.loads((root / "models" / track / "selection.json").read_text(encoding="utf-8"))
    return sel, joblib.load(root / "models" / track / sel["file"])


def test_all_regression_algorithms_reported(root):
    res = pd.read_csv(root / "reports/results/regression_results.csv")
    assert sorted(res["Model"]) == sorted(REGRESSION_ALGOS)
    for metric in ["Test R2", "Test RMSE", "Test MAE"]:
        assert res[metric].notna().all(), f"{metric} missing for some model"
    assert res["Test R2"].is_monotonic_decreasing, "table must be ranked by R2"


def test_all_classification_algorithms_reported(root):
    res = pd.read_csv(root / "reports/results/classification_results.csv")
    assert sorted(res["Model"]) == sorted(CLASSIFICATION_ALGOS)
    for metric in ["Accuracy", "Precision", "Recall", "F1 (weighted)", "ROC-AUC"]:
        assert res[metric].between(0, 1).all(), f"{metric} missing or out of range"


def test_clustering_metrics_reported(root):
    res = pd.read_csv(root / "reports/results/clustering_results.csv")
    assert any(res["Algorithm"].str.startswith("K-Means")) and any(res["Algorithm"].str.startswith("Agglomerative"))
    assert res[["Silhouette", "Davies-Bouldin", "Calinski-Harabasz"]].notna().all().all()


def test_cross_validation_results(root):
    reg_cv = json.loads((root / "reports/results/regression_cv.json").read_text(encoding="utf-8"))
    assert len(reg_cv) >= 2 and all(len(v["scores"]) == 5 for v in reg_cv.values())
    clf_cv = json.loads((root / "reports/results/classification_cv.json").read_text(encoding="utf-8"))
    assert len(clf_cv) >= 2 and all(len(v["f1"]) == 5 for v in clf_cv.values())


def test_tuning_results(root):
    hp = json.loads((root / "reports/results/hyperparameter_results.json").read_text(encoding="utf-8"))
    assert len(hp["regression"]) >= 2 and len(hp["classification"]) >= 2


def test_every_saved_pipeline_loads(root):
    files = list((root / "models").rglob("*.joblib"))
    assert len(files) == 22, f"expected 10 + 10 + 2 model files, found {len(files)}"
    for f in files:
        assert joblib.load(f) is not None


def test_regression_prediction_matches_saved_score(root, reg_split):
    _, X_test, _, y_test = reg_split
    sel, model = _load_selected(root, "regression")
    pred = model.predict(X_test)
    r2 = 1 - ((y_test - pred) ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum()
    assert np.isclose(r2, sel["test_metrics"]["R2"]), "reloaded model does not reproduce the saved test R2"


def test_classification_prediction_and_probabilities(root, clf_split):
    _, X_test, _, y_test = clf_split
    sel, model = _load_selected(root, "classification")
    proba = model.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    assert np.allclose(proba.sum(axis=1), 1.0) and ((proba >= 0) & (proba <= 1)).all()
    assert np.isclose((model.predict(X_test) == y_test).mean(), sel["test_metrics"]["Accuracy"])


def test_kmeans_assignment_of_new_rows(root, meteorites):
    from src.preprocessing import clean_meteorites
    model = joblib.load(root / "models/clustering/k_means.joblib")
    clean, _ = clean_meteorites(meteorites)
    saved = pd.read_csv(root / "models/clustering/kmeans_assignments.csv")
    assert (model.predict(clean) == saved["cluster"].to_numpy()).all(), "saved clusters not reproducible"
    new = pd.DataFrame({"reclat": [-76.0, 35.0], "reclong": [156.0, -100.0], "mass": [20.0, 800.0], "year": [1995, 1980]})
    labels = model.predict(new)
    assert labels.shape == (2,) and set(labels) <= set(range(4))


def test_serialization_round_trip(tmp_path, root, reg_split):
    _, X_test, _, _ = reg_split
    _, model = _load_selected(root, "regression")
    joblib.dump(model, tmp_path / "copy.joblib")
    assert np.allclose(joblib.load(tmp_path / "copy.joblib").predict(X_test), model.predict(X_test))


REQUIRED_FIGURES = {
    "regression": ["eda_correlation_heatmap", "eda_feature_target_scatter", "eda_target_distribution",
                   "best_model_predicted_vs_actual", "best_model_residuals", "random_forest_feature_importance",
                   "model_comparison_r2"],
    "classification": ["eda_class_distribution", "confusion_matrices_all", "roc_curves_all", "decision_tree_plot",
                       "random_forest_feature_importance", "model_comparison_metrics"],
    "clustering": ["kmeans_elbow_curve", "agglomerative_dendrogram", "pca_clusters_both", "tsne_clusters_both",
                   "metric_comparison"],
}


@pytest.mark.parametrize("track", list(REQUIRED_FIGURES))
def test_required_figures_exist(root, track):
    missing = [f for f in REQUIRED_FIGURES[track] if not (root / "reports/figures" / track / f"{f}.png").exists()]
    assert not missing, f"missing figures for {track}: {missing}"


@pytest.mark.parametrize("name", ["regression", "classification", "clustering"])
def test_notebooks_fully_executed_without_errors(root, name):
    nb = nbformat.read(root / "notebooks" / f"{name}.ipynb", as_version=4)
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert all(c.get("execution_count") for c in code), f"{name}: some code cells were never run"
    counts = [c["execution_count"] for c in code]
    assert counts == sorted(counts), f"{name}: cells were not run top to bottom"
    errors = [o for c in code for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert not errors, f"{name}: {len(errors)} cell(s) show an error"
