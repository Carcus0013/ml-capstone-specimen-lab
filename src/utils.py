"""Small helpers shared by the notebooks, the app and the tests."""
import json
import math
import platform
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np


def ensure_dir(path):
    """Create a directory (and parents) if needed and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_jsonable(value):
    """Convert NumPy / pandas scalars and NaN into plain JSON-friendly values."""
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_to_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return value.as_posix()
    return value


def save_json(data, path):
    """Write data to a UTF-8 JSON file, creating the folder if necessary."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_to_jsonable(data), fh, indent=2, ensure_ascii=False)
    return path


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@contextmanager
def timer():
    """Context manager that records elapsed wall-clock seconds in result['seconds']."""
    result = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["seconds"] = time.perf_counter() - start


def file_size_mb(path):
    return Path(path).stat().st_size / 1_048_576


def environment_info():
    """Library versions recorded next to every saved model, for reproducibility."""
    import pandas
    import sklearn

    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "pandas": pandas.__version__,
        "numpy": np.__version__,
    }


def slugify(name):
    """'Random Forest Regressor' -> 'random_forest_regressor' (used for file names)."""
    keep = [c.lower() if c.isalnum() else "_" for c in name]
    slug = "".join(keep)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")
