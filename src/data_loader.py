"""Loading the three assigned datasets and auditing their quality.

The raw CSV files in data/raw/ are only ever read, never written.
"""
import numpy as np
import pandas as pd

from src import config

# Codes that are sometimes used in place of a real missing value.
SENTINEL_VALUES = ["?", "NA", "N/A", "na", "n/a", "-", "--", "", "null", "None", "unknown"]


def load_banana(path=None):
    """Regression dataset: one row per banana sample."""
    return pd.read_csv(path or config.DATASET_FILES["banana"])


def load_mushrooms(path=None):
    """Classification dataset. Every column is a one-letter category code.

    keep_default_na=False stops pandas from turning letter codes into NaN, and keeps
    the '?' code in stalk-root visible so it can be handled deliberately.
    """
    return pd.read_csv(path or config.DATASET_FILES["mushrooms"], dtype=str, keep_default_na=False)


def load_meteorites(path=None):
    """Clustering dataset: NASA Meteorite Landings extract."""
    return pd.read_csv(path or config.DATASET_FILES["meteorites"])


LOADERS = {"banana": load_banana, "mushrooms": load_mushrooms, "meteorites": load_meteorites}


def load_dataset(name):
    return LOADERS[name]()


def column_audit(df):
    """One row per column: dtype, missing values, cardinality and an example value."""
    rows = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        if s.dtype == object or str(s.dtype).startswith("str"):
            sentinels = int(s.astype(str).str.strip().isin(SENTINEL_VALUES).sum())
        else:
            sentinels = 0
        n_unique = int(s.nunique(dropna=True))
        rows.append({
            "column": col,
            "dtype": str(s.dtype),
            "kind": "numeric" if pd.api.types.is_numeric_dtype(s) else "categorical/text",
            "missing": int(s.isna().sum()),
            "missing_pct": round(100 * s.isna().mean(), 2),
            "sentinel_coded": sentinels,
            "n_unique": n_unique,
            "unique_pct": round(100 * n_unique / n, 2) if n else 0.0,
            "constant": n_unique <= 1,
            "identifier_like": n_unique == n and n > 1,
            "example": s.dropna().iloc[0] if s.notna().any() else None,
        })
    return pd.DataFrame(rows)


def dataset_audit(df, name, id_columns=()):
    """Dataset-level audit summary used for reports/results/dataset_audit.json."""
    cols = column_audit(df)
    content = df.drop(columns=[c for c in id_columns if c in df.columns])
    numeric = df.select_dtypes(include=np.number)
    return {
        "dataset": name,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "numeric_columns": list(numeric.columns),
        "categorical_columns": [c for c in df.columns if c not in numeric.columns],
        "missing_cells": int(df.isna().sum().sum()),
        "sentinel_coded_cells": int(cols["sentinel_coded"].sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_rows_ignoring_ids": int(content.duplicated().sum()),
        "constant_columns": cols.loc[cols["constant"], "column"].tolist(),
        "identifier_like_columns": cols.loc[cols["identifier_like"], "column"].tolist(),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1_048_576, 3),
        "column_details": cols.to_dict(orient="records"),
    }
