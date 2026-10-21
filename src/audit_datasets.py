"""Audit all three assigned datasets and write the machine-readable audit files.

Run from the project root:  python -m src.audit_datasets
Outputs: reports/results/dataset_audit.json and reports/results/dataset_audit.csv
"""
import numpy as np
import pandas as pd

from src import config
from src.data_loader import dataset_audit, load_banana, load_meteorites, load_mushrooms
from src.preprocessing import clean_meteorites
from src.utils import save_json


def banana_checks(df):
    """Target candidates and the leakage evidence for the regression dataset."""
    bins = (df.groupby("quality_category")[config.REG_TARGET]
              .agg(["min", "max", "count"]).sort_values("min"))
    ripeness = (df.groupby("ripeness_category")["ripeness_index"]
                  .agg(["min", "max", "count"]).sort_values("min"))
    harvest = pd.to_datetime(df["harvest_date"])
    day_index = (harvest - harvest.min()).dt.days
    return {
        "target_candidates": {
            "quality_score": "continuous (float); chosen regression target",
            "quality_category": "4-level label; exact binning of quality_score (leakage if used as a feature)",
        },
        "target_summary": df[config.REG_TARGET].describe().round(4).to_dict(),
        "target_skew": round(float(df[config.REG_TARGET].skew()), 4),
        "quality_category_ranges": bins.round(2).reset_index().to_dict(orient="records"),
        "ripeness_category_ranges": ripeness.round(2).reset_index().to_dict(orient="records"),
        "harvest_date_range": [str(harvest.min().date()), str(harvest.max().date())],
        "harvest_day_corr_with_target": round(float(np.corrcoef(day_index, df[config.REG_TARGET])[0, 1]), 4),
        "suitable_tracks": ["regression (quality_score)", "classification (quality_category) - not used"],
    }


def mushroom_checks(df):
    counts = df[config.CLF_TARGET].value_counts()
    q_mask = df["stalk-root"] == "?"
    return {
        "target_candidates": {"class": "binary label e (edible) / p (poisonous); chosen target"},
        "target_distribution": counts.to_dict(),
        "target_distribution_pct": (counts / len(df) * 100).round(2).to_dict(),
        "stalk_root_question_mark_rows": int(q_mask.sum()),
        "stalk_root_question_mark_pct": round(100 * q_mask.mean(), 2),
        "poisonous_rate_when_stalk_root_missing_pct": round(100 * (df.loc[q_mask, "class"] == "p").mean(), 2),
        "poisonous_rate_when_stalk_root_known_pct": round(100 * (df.loc[~q_mask, "class"] == "p").mean(), 2),
        "duplicate_feature_rows_with_conflicting_labels": int(
            df.groupby(config.CLF_FEATURES + ["veil-type"])["class"].nunique().gt(1).sum()),
        "suitable_tracks": ["classification (class)"],
        "regression_possible": False,
        "reason_no_regression": "every column is a categorical letter code; there is no continuous outcome",
    }


def meteorite_checks(df):
    clean, log = clean_meteorites(df)
    return {
        "target_candidates": {
            "mass": "continuous, but predicting mass from year/location/class has no clear purpose and is not a "
                    "defensible regression problem for this project",
            "fall": f"binary Fell/Found, extremely imbalanced ({(df['fall'] == 'Fell').sum()} Fell vs "
                    f"{(df['fall'] == 'Found').sum()} Found); used only for post hoc cluster interpretation",
            "recclass": f"{df['recclass'].nunique()} meteorite classes; used only for post hoc interpretation",
        },
        "invalid_values": {
            "placeholder_0_0_coordinates": int(((df["reclat"] == 0) & (df["reclong"] == 0)).sum()),
            "year_outside_860_2016": int(((df["year"] < 860) | (df["year"] > 2016)).sum()),
            "longitude_outside_-180_180": int(((df["reclong"] < -180) | (df["reclong"] > 180)).sum()),
            "mass_zero": int((df["mass"] == 0).sum()),
        },
        "geolocation_is_redundant": "GeoLocation is the string '(reclat, reclong)'; it duplicates two columns",
        "mass_quantiles_g": df["mass"].quantile([0, 0.25, 0.5, 0.75, 0.99, 1]).round(2).to_dict(),
        "cleaning_log": log.to_dict(orient="records"),
        "rows_after_cleaning": int(len(clean)),
        "antarctic_share_after_cleaning_pct": round(100 * (clean["reclat"] < -60).mean(), 2),
        "suitable_tracks": ["clustering (geography + log mass + year)"],
    }


def main():
    datasets = [
        ("banana_quality_dataset.csv", load_banana(), ["sample_id"], banana_checks, "Regression"),
        ("mushrooms.csv", load_mushrooms(), [], mushroom_checks, "Classification"),
        ("meteorite-landings.csv", load_meteorites(), ["id", "name"], meteorite_checks, "Clustering"),
    ]
    report, column_rows = {}, []
    for fname, df, ids, checks, track in datasets:
        audit = dataset_audit(df, fname, id_columns=ids)
        audit["assigned_track"] = track
        audit["specific_checks"] = checks(df)
        report[fname] = audit
        for row in audit["column_details"]:
            column_rows.append({"dataset": fname, "track": track, **row})
    save_json(report, config.RESULTS_DIR / "dataset_audit.json")
    pd.DataFrame(column_rows).to_csv(config.RESULTS_DIR / "dataset_audit.csv", index=False)
    for fname, audit in report.items():
        print(f"{fname}: {audit['rows']} rows x {audit['columns']} columns, "
              f"missing cells={audit['missing_cells']}, duplicates={audit['duplicate_rows']} "
              f"({audit['duplicate_rows_ignoring_ids']} ignoring ids), "
              f"constant={audit['constant_columns']}, id-like={audit['identifier_like_columns']}")


if __name__ == "__main__":
    main()
