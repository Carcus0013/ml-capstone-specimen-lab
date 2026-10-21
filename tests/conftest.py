"""Shared fixtures. Tests run from the repository root:  python -m pytest"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT / "app"), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from src import config  # noqa: E402
from src.data_loader import load_banana, load_meteorites, load_mushrooms  # noqa: E402
from src.preprocessing import classification_xy, regression_xy, train_test  # noqa: E402


@pytest.fixture(scope="session")
def banana():
    return load_banana()


@pytest.fixture(scope="session")
def mushrooms():
    return load_mushrooms()


@pytest.fixture(scope="session")
def meteorites():
    return load_meteorites()


@pytest.fixture(scope="session")
def reg_split(banana):
    return train_test(*regression_xy(banana))


@pytest.fixture(scope="session")
def clf_split(mushrooms):
    return train_test(*classification_xy(mushrooms), stratify=True)


@pytest.fixture(scope="session")
def root():
    return ROOT


@pytest.fixture(scope="session")
def cfg():
    return config
