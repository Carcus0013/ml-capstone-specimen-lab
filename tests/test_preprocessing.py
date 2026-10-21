"""Feature engineering, encoding, schema consistency and leakage prevention."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from src.feature_engineering import add_banana_features, add_meteorite_features, add_mushroom_features
from src.preprocessing import make_classification_pipeline, make_regression_pipeline


def test_banana_feature_is_rowwise(banana):
    out = add_banana_features(banana)
    assert np.allclose(out["firmness_squared"], banana["firmness_kgf"] ** 2)
    # row-wise: computing on one row gives the same value as computing on the whole table
    assert add_banana_features(banana.iloc[[5]])["firmness_squared"].iloc[0] == out["firmness_squared"].iloc[5]


def test_mushroom_feature_values(mushrooms):
    out = add_mushroom_features(mushrooms)
    assert set(out["stalk_ring_asymmetry"].unique()) <= {0, 1, 2}


def test_meteorite_unit_sphere():
    df = pd.DataFrame({"reclat": [90.0, 0.0, -70.0, -70.0], "reclong": [0.0, 90.0, -179.9, 179.9], "mass": [1, 10, 100, 1000]})
    out = add_meteorite_features(df)
    radius = np.sqrt(out.geo_x ** 2 + out.geo_y ** 2 + out.geo_z ** 2)
    assert np.allclose(radius, 1.0), "points must lie on the unit sphere"
    chord = np.linalg.norm(out.loc[2, ["geo_x", "geo_y", "geo_z"]].values - out.loc[3, ["geo_x", "geo_y", "geo_z"]].values)
    assert chord < 0.01, "longitudes -179.9 and +179.9 must be neighbours"
    assert np.allclose(out.log10_mass, [0, 1, 2, 3])


def test_scaler_fitted_on_training_rows_only(reg_split, banana, cfg):
    X_train, X_test, y_train, _ = reg_split
    pipe = make_regression_pipeline(LinearRegression()).fit(X_train, y_train)
    scaler = pipe.named_steps["preprocess"].named_transformers_["num"].named_steps["scale"]
    cols = cfg.REG_NUMERIC + cfg.REG_ENGINEERED
    assert np.allclose(scaler.mean_, add_banana_features(X_train)[cols].mean())
    assert not np.allclose(scaler.mean_, add_banana_features(pd.concat([X_train, X_test]))[cols].mean())


def test_encoder_levels_come_from_training_rows(clf_split, cfg):
    X_train, _, y_train, _ = clf_split
    pipe = make_classification_pipeline(LogisticRegression(max_iter=2000)).fit(X_train, y_train)
    enc = pipe.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
    for col, cats in zip(cfg.CLF_FEATURES, enc.categories_):
        assert set(cats) == set(X_train[col])


def test_target_and_leakage_columns_excluded(reg_split, clf_split, cfg):
    X_train, *_ = reg_split
    assert cfg.REG_TARGET not in X_train and "quality_category" not in X_train and "sample_id" not in X_train
    Xc, *_ = clf_split
    assert cfg.CLF_TARGET not in Xc and "veil-type" not in Xc


def test_split_is_fixed_and_stratified(clf_split, mushrooms, cfg):
    _, X_test, _, y_test = clf_split
    assert len(X_test) == round(0.2 * len(mushrooms))
    overall = (mushrooms[cfg.CLF_TARGET] == "p").mean()
    assert abs(y_test.mean() - overall) < 0.002


def test_unseen_category_does_not_crash(clf_split):
    X_train, X_test, y_train, _ = clf_split
    pipe = make_classification_pipeline(LogisticRegression(max_iter=2000)).fit(X_train, y_train)
    row = X_test.iloc[[0]].copy()
    row["odor"] = "zzz"            # never seen in training
    proba = pipe.predict_proba(row)
    assert proba.shape == (1, 2) and np.isclose(proba.sum(), 1.0)


def test_missing_numeric_value_is_imputed(reg_split):
    X_train, X_test, y_train, _ = reg_split
    pipe = make_regression_pipeline(LinearRegression()).fit(X_train, y_train)
    row = X_test.iloc[[0]].copy()
    row["rainfall_mm"] = np.nan
    assert np.isfinite(pipe.predict(row)[0])
