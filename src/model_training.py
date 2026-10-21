"""Model definitions, hyperparameter grids, training/tuning and model persistence.

Each algorithm is described by a ModelSpec: a baseline pipeline (library defaults or a
documented starting value) plus the grid searched by GridSearchCV/RandomizedSearchCV.
Tuning always uses cross-validation on the TRAINING set; the test set is only used to
score the finished models.
"""
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier,
    GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src import config
from src.evaluation import prediction_latency_ms
from src.preprocessing import make_classification_pipeline, make_regression_pipeline
from src.utils import ensure_dir, environment_info, save_json, slugify, timer

RS = config.RANDOM_STATE


@dataclass
class ModelSpec:
    name: str
    pipeline: object                      # unfitted baseline pipeline
    param_grid: dict = field(default_factory=dict)
    search: str = "grid"                  # "grid" or "random"
    n_iter: int = 20                      # used by RandomizedSearchCV only
    baseline_note: str = "library defaults"


def kfold():
    return KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=RS)


def stratified_kfold():
    return StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=RS)


# ---------------------------------------------------------------------------
# Regression: the ten required algorithms
# ---------------------------------------------------------------------------
def regression_specs():
    reg = make_regression_pipeline
    return [
        ModelSpec("Linear Regression", reg(LinearRegression()),
                  baseline_note="ordinary least squares, no hyperparameters to tune"),
        ModelSpec("Ridge Regression", reg(Ridge(random_state=RS)),
                  {"model__alpha": list(np.logspace(-3, 3, 13))}),
        ModelSpec("Lasso Regression", reg(Lasso(random_state=RS, max_iter=50_000)),
                  {"model__alpha": [1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1]}),
        ModelSpec("ElasticNet Regression", reg(ElasticNet(random_state=RS, max_iter=50_000)),
                  {"model__alpha": [1e-4, 1e-3, 1e-2, 0.1],
                   "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]}),
        ModelSpec("Polynomial Regression", reg(LinearRegression(), poly_degree=2),
                  {"preprocess__num__poly__degree": [1, 2, 3]},
                  baseline_note="degree=2 on the numeric block"),
        ModelSpec("Decision Tree Regressor", reg(DecisionTreeRegressor(random_state=RS)),
                  {"model__max_depth": [3, 5, 7, 9, 12, None],
                   "model__min_samples_leaf": [1, 5, 10, 20]}),
        ModelSpec("Random Forest Regressor", reg(RandomForestRegressor(random_state=RS, n_jobs=-1)),
                  {"model__n_estimators": [100, 200, 400],
                   "model__max_depth": [None, 10, 20],
                   "model__min_samples_leaf": [1, 2, 4],
                   "model__max_features": [1.0, "sqrt", 0.5]},
                  search="random", n_iter=20),
        ModelSpec("Gradient Boosting Regressor", reg(GradientBoostingRegressor(random_state=RS)),
                  {"model__learning_rate": [0.03, 0.05, 0.1, 0.2],
                   "model__n_estimators": [100, 300, 500],
                   "model__max_depth": [2, 3, 4],
                   "model__subsample": [0.8, 1.0]},
                  search="random", n_iter=25),
        ModelSpec("Support Vector Regressor", reg(SVR()),
                  {"model__C": [0.1, 1, 10, 100],
                   "model__kernel": ["linear", "rbf"],
                   "model__epsilon": [0.01, 0.05, 0.1]}),
        ModelSpec("K-Nearest Neighbors Regressor", reg(KNeighborsRegressor()),
                  {"model__n_neighbors": list(range(1, 31, 2)),
                   "model__weights": ["uniform", "distance"],
                   "model__p": [1, 2]}),
    ]


# ---------------------------------------------------------------------------
# Classification: the ten required algorithms (Part A = first five)
# ---------------------------------------------------------------------------
def classification_specs():
    clf = make_classification_pipeline
    return [
        ModelSpec("Logistic Regression", clf(LogisticRegression(max_iter=5000, random_state=RS)),
                  {"model__C": [0.001, 0.01, 0.1, 1, 10, 100]}),
        ModelSpec("K-Nearest Neighbors", clf(KNeighborsClassifier()),
                  {"model__n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21],
                   "model__weights": ["uniform", "distance"],
                   "model__metric": ["euclidean", "manhattan"]}),
        ModelSpec("Gaussian Naive Bayes", clf(GaussianNB()),
                  {"model__var_smoothing": list(np.logspace(-12, 0, 13))}),
        ModelSpec("Decision Tree", clf(DecisionTreeClassifier(random_state=RS)),
                  {"model__max_depth": [2, 3, 4, 5, 6, 8, None],
                   "model__criterion": ["gini", "entropy"],
                   "model__min_samples_leaf": [1, 5, 20]}),
        ModelSpec("Support Vector Machine (SVC)", clf(SVC(probability=True, random_state=RS)),
                  {"model__C": [0.01, 0.1, 1, 10],
                   "model__kernel": ["linear", "rbf"]}),
        ModelSpec("Random Forest", clf(RandomForestClassifier(random_state=RS, n_jobs=-1)),
                  {"model__n_estimators": [25, 50, 100, 200],
                   "model__max_depth": [None, 5, 10],
                   "model__max_features": ["sqrt", "log2"]}),
        ModelSpec("AdaBoost", clf(AdaBoostClassifier(random_state=RS)),
                  {"model__n_estimators": [25, 50, 100, 200],
                   "model__learning_rate": [0.1, 0.5, 1.0]}),
        ModelSpec("Gradient Boosting", clf(GradientBoostingClassifier(random_state=RS)),
                  {"model__n_estimators": [50, 100, 200],
                   "model__learning_rate": [0.05, 0.1, 0.2],
                   "model__max_depth": [2, 3]}),
        ModelSpec("Bagging (Decision Tree)",
                  clf(BaggingClassifier(estimator=DecisionTreeClassifier(random_state=RS),
                                        random_state=RS, n_jobs=-1)),
                  {"model__n_estimators": [10, 25, 50, 100],
                   "model__max_samples": [0.5, 1.0],
                   "model__max_features": [0.5, 1.0]},
                  baseline_note="library defaults with a DecisionTreeClassifier base estimator"),
        ModelSpec("MLP Neural Network",
                  clf(MLPClassifier(max_iter=1000, random_state=RS)),
                  {"model__hidden_layer_sizes": [(16,), (64,), (64, 32)],
                   "model__activation": ["relu", "tanh", "logistic"],
                   "model__alpha": [1e-4, 1e-2]},
                  baseline_note="library defaults (100 hidden units, relu) with max_iter=1000"),
    ]


# ---------------------------------------------------------------------------
# Training, tuning and timing
# ---------------------------------------------------------------------------
def model_size_mb(model):
    return len(pickle.dumps(model)) / 1_048_576


def run_model(spec, X_train, y_train, X_test, y_test, metric_fn, cv, scoring):
    """Fit the baseline, tune it on the training set (if it has a grid), score both on
    the shared test set and record costs. Returns a dict with fitted models + numbers."""
    baseline = clone(spec.pipeline)
    with timer() as t:
        baseline.fit(X_train, y_train)
    out = {
        "name": spec.name,
        "baseline_model": baseline,
        "baseline_fit_seconds": t["seconds"],
        "baseline_test": metric_fn(baseline, X_test, y_test),
        "search": None,
    }
    final = baseline
    if spec.param_grid:
        if spec.search == "random":
            search = RandomizedSearchCV(clone(spec.pipeline), spec.param_grid, n_iter=spec.n_iter,
                                        cv=cv, scoring=scoring, n_jobs=-1, random_state=RS)
        else:
            search = GridSearchCV(clone(spec.pipeline), spec.param_grid, cv=cv, scoring=scoring, n_jobs=-1)
        with timer() as t:
            search.fit(X_train, y_train)
        final = search.best_estimator_
        out.update({
            "search": search,
            "search_seconds": t["seconds"],
            "best_params": {k.split("__")[-1]: (v.item() if isinstance(v, np.generic) else v)
                            for k, v in search.best_params_.items()},
            "best_cv_score": search.best_score_,
            "tuned_test": metric_fn(final, X_test, y_test),
        })
    with timer() as t:
        clone(final).fit(X_train, y_train)
    out.update({
        "final_model": final,
        "final_fit_seconds": t["seconds"],
        "final_train": metric_fn(final, X_train, y_train),
        "final_test": metric_fn(final, X_test, y_test),
        "latency_ms_per_1k": prediction_latency_ms(final, X_test),
        "size_mb": model_size_mb(final),
    })
    return out


def cross_validate(model, X_train, y_train, cv, scoring):
    """k-fold CV of an (unfitted clone of a) full pipeline on the training set only."""
    scores = cross_val_score(clone(model), X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    return {"scores": scores.tolist(), "mean": float(scores.mean()), "std": float(scores.std())}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_pipeline(model, track, name, extra_metadata=None):
    """Save a fitted pipeline to models/<track>/<slug>.joblib with a metadata JSON."""
    folder = ensure_dir(config.MODELS_DIR / track)
    slug = slugify(name)
    path = folder / f"{slug}.joblib"
    joblib.dump(model, path, compress=3)
    meta = {
        "name": name,
        "file": path.name,
        "saved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": environment_info(),
    }
    meta.update(extra_metadata or {})
    save_json(meta, folder / f"{slug}.json")
    return path


def load_pipeline(path):
    # joblib/pickle can execute code on load, so only ever call this on model files
    # produced by this project's own notebooks (models/), never on user uploads.
    return joblib.load(Path(path))
