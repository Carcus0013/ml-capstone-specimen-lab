"""Dataset loading, schemas and the documented data-quality facts."""
import hashlib

import pytest

from src.preprocessing import clean_meteorites

EXPECTED_SHAPES = {"banana": (1000, 16), "mushrooms": (8124, 23), "meteorites": (45716, 10)}
ORIGINAL_MD5 = {
    "banana_quality_dataset.csv": "6cbe6bbc9f8315a0def709047b2282be",
    "mushrooms.csv": "20e36f0bcc0fec359eb394ee947237dd",
    "meteorite-landings.csv": "f0967aac3c7ca2e725f752938d2a3f82",
}


def test_raw_files_unchanged(cfg):
    """The assigned datasets must stay byte-for-byte identical to the originals."""
    for path in cfg.DATASET_FILES.values():
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        assert digest == ORIGINAL_MD5[path.name], f"{path.name} has been modified"


@pytest.mark.parametrize("name", ["banana", "mushrooms", "meteorites"])
def test_shapes(name, banana, mushrooms, meteorites):
    df = {"banana": banana, "mushrooms": mushrooms, "meteorites": meteorites}[name]
    assert df.shape == EXPECTED_SHAPES[name], f"{name} has shape {df.shape}"


def test_banana_schema(banana, cfg):
    expected = set(cfg.REG_NUMERIC + cfg.REG_CATEGORICAL + [cfg.REG_TARGET]) | set(cfg.REG_DROPPED)
    assert expected == set(banana.columns)
    assert banana.isna().sum().sum() == 0
    assert banana[cfg.REG_NUMERIC].dtypes.map(lambda d: d.kind == "f").all()


def test_quality_category_is_binned_target(banana):
    """Evidence for the leakage decision: the category bins never overlap."""
    ranges = banana.groupby("quality_category")["quality_score"].agg(["min", "max"]).sort_values("min")
    assert (ranges["min"].iloc[1:].values > ranges["max"].iloc[:-1].values).all()


def test_mushroom_schema(mushrooms, cfg):
    assert set(cfg.CLF_FEATURES) | {"veil-type", cfg.CLF_TARGET} == set(mushrooms.columns)
    assert set(mushrooms[cfg.CLF_TARGET]) == {"e", "p"}
    assert (mushrooms["stalk-root"] == "?").sum() == 2480, "'?' must be read as a category, not NaN"
    assert mushrooms["veil-type"].nunique() == 1


def test_meteorite_cleaning(meteorites):
    clean, log = clean_meteorites(meteorites)
    assert len(clean) == 31911
    assert int(log["rows_removed"].sum()) == len(meteorites) - len(clean)
    assert not ((clean.reclat == 0) & (clean.reclong == 0)).any()
    assert clean.reclong.between(-180, 180).all() and clean.year.between(860, 2016).all()
    assert (clean.mass > 0).all() and clean[["mass", "year", "reclat", "reclong"]].notna().all().all()
