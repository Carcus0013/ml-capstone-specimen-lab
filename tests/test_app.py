"""App input validation, CSV upload validation and Streamlit smoke tests."""
import re

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from core import logic
from core.data import ROOT, read_json

REG_SCHEMA = read_json(ROOT / "models/regression/feature_schema.json")
CLF_SCHEMA = read_json(ROOT / "models/classification/feature_schema.json")


def _valid_banana():
    values = {c: s["median"] for c, s in REG_SCHEMA["numeric"].items()}
    values.update({c: levels[0] for c, levels in REG_SCHEMA["categorical"].items()})
    return values


# --- regression input -------------------------------------------------------
def test_regression_valid_input():
    row, errors, warnings = logic.validate_regression_input(_valid_banana(), REG_SCHEMA)
    assert errors == [] and warnings == [] and row.shape == (1, 11)


@pytest.mark.parametrize("bad", ["abc", None, float("nan"), float("inf")])
def test_regression_rejects_non_numbers(bad):
    values = _valid_banana() | {"sugar_content_brix": bad}
    row, errors, _ = logic.validate_regression_input(values, REG_SCHEMA)
    assert row is None and any("sugar_content_brix" in e for e in errors)


def test_regression_range_checks():
    s = REG_SCHEMA["numeric"]["firmness_kgf"]
    _, errors, _ = logic.validate_regression_input(_valid_banana() | {"firmness_kgf": s["max"] * 10}, REG_SCHEMA)
    assert errors, "a value far outside the data must be rejected"
    lo, hi = logic.numeric_bounds(REG_SCHEMA)["firmness_kgf"]
    row, errors, warnings = logic.validate_regression_input(_valid_banana() | {"firmness_kgf": (s["max"] + hi) / 2}, REG_SCHEMA)
    assert not errors and warnings, "a value just outside training range should warn, not block"


def test_regression_rejects_unknown_category():
    _, errors, _ = logic.validate_regression_input(_valid_banana() | {"variety": "Unknown"}, REG_SCHEMA)
    assert any("variety" in e for e in errors)


def test_quality_band_cut_points():
    assert [logic.quality_band(v) for v in (1.0, 1.5, 2.0, 3.0, 3.6)] == ["Unripe", "Unripe", "Processing", "Good", "Premium"]


# --- classification input -----------------------------------------------------
def test_mushroom_validation():
    example = {k: v for k, v in CLF_SCHEMA["examples"][0].items() if k != "actual_class"}
    row, errors = logic.validate_mushroom_input(example, CLF_SCHEMA)
    assert errors == [] and row.shape == (1, 21)
    _, errors = logic.validate_mushroom_input(example | {"odor": "zzz"}, CLF_SCHEMA)
    assert errors


# --- clustering CSV upload ----------------------------------------------------
def test_cluster_csv_missing_columns():
    valid, rejected, errors = logic.validate_cluster_frame(pd.DataFrame({"lat": [1], "lon": [2]}))
    assert valid is None and errors and "reclat" in errors[0]


def test_cluster_csv_row_level_rejections():
    df = pd.DataFrame({"reclat": [-76, 95, 10, 0, 20], "reclong": [156, 0, 10, 0, 20],
                       "mass": [20, 5, -1, 3, "heavy"], "year": [1995, 2000, 2000, 2000, 1990]})
    valid, rejected, errors = logic.validate_cluster_frame(df)
    assert errors == [] and len(valid) == 1 and len(rejected) == 4
    reasons = " ".join(rejected["reason"])
    for expected in ["reclat must be between", "mass must be greater", "placeholder", "mass is missing or not a number"]:
        assert expected in reasons


def test_cluster_csv_limits():
    too_many = pd.DataFrame({c: np.ones(logic.MAX_UPLOAD_ROWS + 1) for c in logic.CLUSTER_COLUMNS})
    assert logic.validate_cluster_frame(too_many)[2]
    assert logic.validate_cluster_frame(pd.DataFrame(columns=logic.CLUSTER_COLUMNS))[2]


def test_describe_cluster_rules():
    assert logic.describe_cluster({"pct_antarctic": 97, "pct_fell": 0, "median_long": 150}) == "Antarctic finds"
    assert logic.describe_cluster({"pct_antarctic": 0, "pct_fell": 52, "median_long": 12}) == "Historic observed falls"
    assert logic.describe_cluster({"pct_antarctic": 0, "pct_fell": 5, "median_long": -100}) == "Americas finds"


# --- Streamlit smoke tests ------------------------------------------------------
PAGE_SCRIPT = """
import sys
sys.path.insert(0, {app!r}); sys.path.insert(0, {root!r})
from views import {page}
{page}.render()
"""


def _page(page):
    return AppTest.from_string(PAGE_SCRIPT.format(app=str(ROOT / "app"), root=str(ROOT), page=page), default_timeout=180)


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_app_entry_point_starts():
    at = AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=180).run()
    _ok(at)
    assert any("Three specimens" in m.value for m in at.markdown)


@pytest.mark.parametrize("page", ["comparison", "explorer", "about", "clustering"])
def test_pages_render(page):
    _ok(_page(page).run())


def test_regression_page_predicts_with_real_model():
    at = _page("regression").run()
    at.button[0].click().run()
    [b for b in at.button if "Predict" in str(b.label)][0].click().run()
    _ok(at)
    panel = next(m.value for m in at.markdown if 'class="result"' in m.value)
    value = float(re.search(r'result-value">([\d.]+)<', panel).group(1))
    actual = REG_SCHEMA["examples"][0]["actual_quality_score"]
    assert abs(value - actual) < 0.1, "prediction for a held-out banana should be close to its measured score"


def test_classification_page_predicts_both_classes():
    for i, expected in [(0, "Edible"), (1, "Poisonous")]:
        at = _page("classification").run()
        at.button[i].click().run()
        [b for b in at.button if "Classify" in str(b.label)][0].click().run()
        _ok(at)
        assert any(f'result-value">{expected}<' in m.value for m in at.markdown)


def test_clustering_page_assigns_new_meteorite():
    at = _page("clustering").run()
    [b for b in at.button if "Assign to a cluster" in str(b.label)][0].click().run()
    _ok(at)
    assert any("Assigned cluster" in m.value for m in at.markdown)


def test_no_absolute_paths_in_app_or_src():
    pattern = re.compile(r"[A-Za-z]:\\\\|C:/Users|/home/|/Users/")
    for folder in ("app", "src"):
        for f in (ROOT / folder).rglob("*.py"):
            assert not pattern.search(f.read_text(encoding="utf-8")), f"absolute path found in {f}"
