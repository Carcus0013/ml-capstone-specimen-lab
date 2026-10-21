"""Input validation and prediction helpers for the app.

Nothing here imports Streamlit, so every function can be unit-tested directly
(see tests/test_app_logic.py). All predictions go through the saved scikit-learn
pipelines, which contain the exact training-time preprocessing.
"""
from datetime import date

import numpy as np
import pandas as pd

# Cut points of quality_category observed in the banana data (see the regression notebook,
# section 5.1). Used only to put a predicted score into words.
QUALITY_BANDS = [(1.50, "Unripe"), (2.50, "Processing"), (3.50, "Good"), (np.inf, "Premium")]

CLUSTER_COLUMNS = ["reclat", "reclong", "mass", "year"]
MAX_UPLOAD_ROWS = 5_000
MAX_UPLOAD_MB = 5
YEAR_RANGE = (860, date.today().year)


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------
def quality_band(score):
    for upper, name in QUALITY_BANDS:
        if score <= upper:
            return name
    return QUALITY_BANDS[-1][1]


def numeric_bounds(schema, margin=0.05):
    """Hard input limits: the training range widened by 5% on each side (never below 0,
    because every banana measurement is non-negative)."""
    bounds = {}
    for col, s in schema["numeric"].items():
        span = s["max"] - s["min"]
        bounds[col] = (max(0.0, s["min"] - margin * span), s["max"] + margin * span)
    return bounds


def validate_regression_input(values, schema):
    """Check one banana record. Returns (DataFrame or None, errors, warnings).

    Errors block prediction (wrong type, missing, far outside the data). Warnings flag
    values outside the training range, where any prediction is an extrapolation.
    """
    errors, warnings, row = [], [], {}
    bounds = numeric_bounds(schema)
    for col, stats in schema["numeric"].items():
        raw = values.get(col)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors.append(f"{col}: enter a number.")
            continue
        if not np.isfinite(value):
            errors.append(f"{col}: enter a finite number.")
            continue
        lo, hi = bounds[col]
        if value < lo or value > hi:
            errors.append(f"{col} must be between {lo:.2f} and {hi:.2f}.")
        elif value < stats["min"] or value > stats["max"]:
            warnings.append(f"{col} = {value:g} lies outside the training range "
                            f"({stats['min']:g} to {stats['max']:g}), so this prediction is an extrapolation.")
        row[col] = value
    for col, levels in schema["categorical"].items():
        value = values.get(col)
        if value not in levels:
            errors.append(f"{col}: choose one of {', '.join(levels)}.")
        row[col] = value
    if errors:
        return None, errors, warnings
    return pd.DataFrame([row]), errors, warnings


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def validate_mushroom_input(values, schema):
    """Every attribute must be one of the codes seen during training."""
    errors, row = [], {}
    for col, levels in schema["categorical"].items():
        value = values.get(col)
        if value not in levels:
            errors.append(f"{col}: choose one of the listed options.")
        row[col] = value
    if errors:
        return None, errors
    return pd.DataFrame([row]), errors


def readable_feature(name, labels):
    """'cat__odor_n' -> 'odor = none'; 'num__stalk_ring_asymmetry' -> 'stalk ring asymmetry'."""
    if name.startswith("num__"):
        return name[5:].replace("_", " ")
    feature, code = name[5:].rsplit("_", 1)
    return f"{feature} = {labels.get(feature, {}).get(code, code)}"


def logistic_contributions(pipeline, X_row, labels):
    """Per-feature contributions to the log-odds of the positive class for a fitted
    logistic-regression pipeline: coefficient x transformed value. Only active inputs
    (non-zero after encoding) are returned, sorted from most edible to most poisonous."""
    model = pipeline.named_steps["model"]
    transformed = np.asarray(pipeline[:-1].transform(X_row))[0]
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    contrib = pd.Series(model.coef_[0] * transformed, index=[readable_feature(n, labels) for n in names])
    return contrib[contrib != 0].sort_values(), float(model.intercept_[0])


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def describe_cluster(profile_row):
    """Short name for a K-Means cluster, derived by fixed rules from its profile so the
    name follows the data even if cluster numbers change after retraining."""
    if profile_row["pct_antarctic"] >= 50:
        return "Antarctic finds"
    if profile_row["pct_fell"] >= 30:
        return "Historic observed falls"
    if profile_row["median_long"] < -30:
        return "Americas finds"
    return "Africa & Asia finds"


def validate_cluster_frame(df):
    """Validate a table of meteorites for K-Means assignment.

    Returns (valid_rows, rejected_rows_with_reason, errors). `errors` are file-level
    problems (missing columns, too many rows) that stop processing entirely; row-level
    problems only reject that row. Uploaded content is only ever parsed as data.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in CLUSTER_COLUMNS if c not in df.columns]
    if missing:
        return None, None, [f"Missing required column(s): {', '.join(missing)}. "
                            f"The file needs the columns {', '.join(CLUSTER_COLUMNS)}."]
    if df.empty:
        return None, None, ["The file has a header but no data rows."]
    if len(df) > MAX_UPLOAD_ROWS:
        return None, None, [f"The file has {len(df):,} rows; the limit is {MAX_UPLOAD_ROWS:,}."]

    reasons = pd.Series("", index=df.index)
    work = df.copy()
    for col in CLUSTER_COLUMNS:
        work[col] = pd.to_numeric(df[col], errors="coerce")
        reasons[work[col].isna()] += f"{col} is missing or not a number; "
    checks = [
        (~work["reclat"].between(-90, 90), "reclat must be between -90 and 90; "),
        (~work["reclong"].between(-180, 180), "reclong must be between -180 and 180; "),
        (~(work["mass"] > 0), "mass must be greater than 0 g; "),
        (~work["year"].between(*YEAR_RANGE), f"year must be between {YEAR_RANGE[0]} and {YEAR_RANGE[1]}; "),
        ((work["reclat"] == 0) & (work["reclong"] == 0), "(0, 0) is the dataset's placeholder for an unknown location; "),
    ]
    numeric_ok = reasons == ""
    for mask, message in checks:
        reasons[numeric_ok & mask] += message
    ok = reasons == ""
    rejected = df.loc[~ok].assign(reason=reasons[~ok].str.rstrip("; "))
    return work.loc[ok].reset_index(drop=True), rejected, []
