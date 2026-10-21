"""Cleaning rules, train/test splits and leakage-safe preprocessing pipelines.

Leakage rule (guideline 7.1): anything that *learns* from data (imputers, encoders,
scalers, polynomial expansion) lives inside a scikit-learn Pipeline. Calling
pipeline.fit(X_train, ...) therefore fits those steps on training rows only, and
cross-validation refits them inside every training fold.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, PolynomialFeatures, StandardScaler

from src import config
from src.feature_engineering import add_banana_features, add_meteorite_features, add_mushroom_features


# ---------------------------------------------------------------------------
# Regression (banana)
# ---------------------------------------------------------------------------
def regression_xy(df):
    """Select the modelling columns. Leakage / identifier columns never enter X."""
    X = df[config.REG_NUMERIC + config.REG_CATEGORICAL].copy()
    y = df[config.REG_TARGET].copy()
    return X, y


def regression_preprocessor(poly_degree=None):
    numeric = config.REG_NUMERIC + config.REG_ENGINEERED
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if poly_degree is not None:
        # Expansion is applied to the numeric block only, so the one-hot dummies are
        # not multiplied together (that would explode the feature count).
        num_steps.append(("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)))
    num_steps.append(("scale", StandardScaler()))
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        # drop='first' avoids the exact collinearity of a full dummy set in OLS;
        # handle_unknown='ignore' encodes an unseen level as all zeros instead of failing.
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", Pipeline(num_steps), numeric),
        ("cat", categorical, config.REG_CATEGORICAL),
    ])


def make_regression_pipeline(model, poly_degree=None):
    return Pipeline([
        ("features", FunctionTransformer(add_banana_features)),
        ("preprocess", regression_preprocessor(poly_degree)),
        ("model", model),
    ])


# ---------------------------------------------------------------------------
# Classification (mushrooms)
# ---------------------------------------------------------------------------
def classification_xy(df):
    X = df[config.CLF_FEATURES].copy()     # veil-type (constant) is not selected
    y = df[config.CLF_TARGET].copy()
    return X, y


def classification_preprocessor():
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        # Nominal codes have no order, so one-hot encoding is used. '?' in stalk-root is
        # kept as its own level because its absence is itself informative.
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([
        ("cat", categorical, config.CLF_FEATURES),
        ("num", numeric, config.CLF_ENGINEERED),
    ])


def make_classification_pipeline(model):
    return Pipeline([
        ("features", FunctionTransformer(add_mushroom_features)),
        ("preprocess", classification_preprocessor()),
        ("model", model),
    ])


# ---------------------------------------------------------------------------
# Shared split helper
# ---------------------------------------------------------------------------
def train_test(X, y, stratify=False):
    """The single 80:20 split used by every model in a track (random_state=42)."""
    return train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y if stratify else None,
    )


# ---------------------------------------------------------------------------
# Clustering (meteorites)
# ---------------------------------------------------------------------------
def clean_meteorites(df):
    """Apply documented validity rules in order and record how many rows each removes.

    Duplicate-looking rows are deliberately kept: they have different names and ids,
    i.e. they are separate specimens recovered at the same recorded site (common for
    Antarctic expeditions that log one coordinate for a whole search area).
    """
    lo, hi = config.CLU_VALID_YEAR_RANGE
    rules = [
        ("mass missing", lambda d: d["mass"].isna()),
        ("year missing", lambda d: d["year"].isna()),
        ("coordinates missing", lambda d: d["reclat"].isna() | d["reclong"].isna()),
        ("placeholder coordinates (0, 0)", lambda d: (d["reclat"] == 0) & (d["reclong"] == 0)),
        (f"year outside {lo}-{hi}", lambda d: (d["year"] < lo) | (d["year"] > hi)),
        ("longitude outside [-180, 180]", lambda d: (d["reclong"] < -180) | (d["reclong"] > 180)),
        ("mass equal to 0 g", lambda d: d["mass"] <= 0),
    ]
    log = []
    out = df.copy()
    for reason, mask_fn in rules:
        mask = mask_fn(out)
        log.append({"rule": reason, "rows_removed": int(mask.sum()), "rows_remaining": int((~mask).sum())})
        out = out.loc[~mask]
    return out.reset_index(drop=True), pd.DataFrame(log)


def clustering_preprocessor():
    """Feature engineering + scaling. Clustering has no labels and no test set, so the
    scaler is fitted on all cleaned rows; no target or external label is ever used."""
    return Pipeline([
        ("features", FunctionTransformer(add_meteorite_features)),
        ("select_scale", ColumnTransformer([("num", StandardScaler(), config.CLU_MODEL_FEATURES)])),
    ])
