"""Central configuration: paths, random seed and dataset / feature definitions.

Every notebook, the Streamlit app and the tests import from here, so a path or a
feature list is defined exactly once. All paths are built relative to the project
root, so nothing depends on a particular machine or operating system.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility and evaluation protocol (guidelines 7.1 and 7.2)
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20          # 80:20 train/test split in every supervised track
CV_FOLDS = 5              # 5-fold cross-validation

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"
COMPARISON_DIR = REPORTS_DIR / "model_comparison"

DATASET_FILES = {
    "banana": DATA_RAW / "banana_quality_dataset.csv",
    "mushrooms": DATA_RAW / "mushrooms.csv",
    "meteorites": DATA_RAW / "meteorite-landings.csv",
}

# ---------------------------------------------------------------------------
# Regression track: banana_quality_dataset.csv
# ---------------------------------------------------------------------------
REG_TARGET = "quality_score"
REG_NUMERIC = [
    "ripeness_index", "sugar_content_brix", "firmness_kgf", "length_cm", "weight_g",
    "tree_age_years", "altitude_m", "rainfall_mm", "soil_nitrogen_ppm",
]
REG_CATEGORICAL = ["variety", "region"]
REG_ENGINEERED = ["firmness_squared"]
REG_DROPPED = {
    "sample_id": "row identifier with no predictive content",
    "quality_category": "TARGET LEAKAGE: an exact binning of quality_score",
    "ripeness_category": "coarser duplicate of ripeness_index",
    "harvest_date": "single 61-day season; no measurable relationship with the target",
}

# ---------------------------------------------------------------------------
# Classification track: mushrooms.csv (UCI Mushroom data, all categorical)
# ---------------------------------------------------------------------------
CLF_TARGET = "class"
CLF_POSITIVE = "p"        # poisonous: the safety-critical class
CLF_CLASS_LABELS = {"e": "edible", "p": "poisonous"}
CLF_DROPPED = {
    "veil-type": "constant column: every row has the value 'p'",
}
CLF_FEATURES = [
    "cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment",
    "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root",
    "stalk-surface-above-ring", "stalk-surface-below-ring", "stalk-color-above-ring",
    "stalk-color-below-ring", "veil-color", "ring-number", "ring-type",
    "spore-print-color", "population", "habitat",
]
CLF_ENGINEERED = ["stalk_ring_asymmetry"]

# Attribute codes from the UCI Mushroom data dictionary, used to show readable labels.
MUSHROOM_CODE_LABELS = {
    "cap-shape": {"b": "bell", "c": "conical", "x": "convex", "f": "flat", "k": "knobbed", "s": "sunken"},
    "cap-surface": {"f": "fibrous", "g": "grooves", "y": "scaly", "s": "smooth"},
    "cap-color": {"n": "brown", "b": "buff", "c": "cinnamon", "g": "gray", "r": "green",
                  "p": "pink", "u": "purple", "e": "red", "w": "white", "y": "yellow"},
    "bruises": {"t": "bruises", "f": "no bruises"},
    "odor": {"a": "almond", "l": "anise", "c": "creosote", "y": "fishy", "f": "foul",
             "m": "musty", "n": "none", "p": "pungent", "s": "spicy"},
    "gill-attachment": {"a": "attached", "d": "descending", "f": "free", "n": "notched"},
    "gill-spacing": {"c": "close", "w": "crowded", "d": "distant"},
    "gill-size": {"b": "broad", "n": "narrow"},
    "gill-color": {"k": "black", "n": "brown", "b": "buff", "h": "chocolate", "g": "gray",
                   "r": "green", "o": "orange", "p": "pink", "u": "purple", "e": "red",
                   "w": "white", "y": "yellow"},
    "stalk-shape": {"e": "enlarging", "t": "tapering"},
    "stalk-root": {"b": "bulbous", "c": "club", "u": "cup", "e": "equal",
                   "z": "rhizomorphs", "r": "rooted", "?": "missing (not recorded)"},
    "stalk-surface-above-ring": {"f": "fibrous", "y": "scaly", "k": "silky", "s": "smooth"},
    "stalk-surface-below-ring": {"f": "fibrous", "y": "scaly", "k": "silky", "s": "smooth"},
    "stalk-color-above-ring": {"n": "brown", "b": "buff", "c": "cinnamon", "g": "gray",
                               "o": "orange", "p": "pink", "e": "red", "w": "white", "y": "yellow"},
    "stalk-color-below-ring": {"n": "brown", "b": "buff", "c": "cinnamon", "g": "gray",
                               "o": "orange", "p": "pink", "e": "red", "w": "white", "y": "yellow"},
    "veil-type": {"p": "partial", "u": "universal"},
    "veil-color": {"n": "brown", "o": "orange", "w": "white", "y": "yellow"},
    "ring-number": {"n": "none", "o": "one", "t": "two"},
    "ring-type": {"c": "cobwebby", "e": "evanescent", "f": "flaring", "l": "large",
                  "n": "none", "p": "pendant", "s": "sheathing", "z": "zone"},
    "spore-print-color": {"k": "black", "n": "brown", "b": "buff", "h": "chocolate", "r": "green",
                          "o": "orange", "u": "purple", "w": "white", "y": "yellow"},
    "population": {"a": "abundant", "c": "clustered", "n": "numerous", "s": "scattered",
                   "v": "several", "y": "solitary"},
    "habitat": {"g": "grasses", "l": "leaves", "m": "meadows", "p": "paths", "u": "urban",
                "w": "waste", "d": "woods"},
}

# ---------------------------------------------------------------------------
# Clustering track: meteorite-landings.csv
# ---------------------------------------------------------------------------
CLU_RAW_FEATURES = ["reclat", "reclong", "mass", "year"]
CLU_MODEL_FEATURES = ["geo_x", "geo_y", "geo_z", "log10_mass", "year"]
CLU_EXTERNAL_LABELS = ["fall", "recclass", "nametype"]   # post hoc interpretation only
CLU_VALID_YEAR_RANGE = (860, 2016)   # documented valid range for this NASA extract
CLU_AGGLO_SAMPLE_SIZE = 10_000       # agglomerative clustering needs O(n^2) memory
CLU_TSNE_SAMPLE_SIZE = 4_000
