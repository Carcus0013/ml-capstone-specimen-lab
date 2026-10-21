"""Cached access to the saved models, result files and datasets.

Models are loaded once per server process (st.cache_resource) and tables once per
file (st.cache_data), so page interactions never retrain or reload anything.
All paths are relative to the repository, so the app runs the same locally and in
the cloud.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
RESULTS = ROOT / "reports" / "results"
FIGURES = ROOT / "reports" / "figures"

TRACKS = ("regression", "classification", "clustering")

REQUIRED_FILES = [
    MODELS / "regression" / "selection.json",
    MODELS / "regression" / "feature_schema.json",
    MODELS / "classification" / "selection.json",
    MODELS / "classification" / "feature_schema.json",
    MODELS / "clustering" / "k_means.joblib",
    MODELS / "clustering" / "pca_projection.joblib",
    MODELS / "clustering" / "clustering_summary.json",
    MODELS / "clustering" / "kmeans_assignments.csv",
    MODELS / "clustering" / "agglomerative_sample_assignments.csv",
    RESULTS / "regression_results.csv",
    RESULTS / "classification_results.csv",
    RESULTS / "clustering_results.csv",
    RESULTS / "dataset_audit.json",
]


def missing_files():
    """Startup check: required artefacts that are not present."""
    missing = [p for p in REQUIRED_FILES if not p.exists()]
    for track in ("regression", "classification"):
        sel = MODELS / track / "selection.json"
        if sel.exists() and not (MODELS / track / read_json(sel)["file"]).exists():
            missing.append(MODELS / track / read_json(sel)["file"])
    return [p.relative_to(ROOT).as_posix() for p in missing]


def read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def load_json(path_str):
    return read_json(Path(path_str))


@st.cache_data(show_spinner=False)
def load_csv(path_str):
    return pd.read_csv(path_str)


@st.cache_resource(show_spinner="Loading the trained model...")
def load_model(path_str):
    # Only files from this repository's models/ folder are ever loaded here.
    return joblib.load(path_str)


def selection(track):
    return load_json(str(MODELS / track / "selection.json"))


def schema(track):
    return load_json(str(MODELS / track / "feature_schema.json"))


def selected_model(track):
    sel = selection(track)
    return sel, load_model(str(MODELS / track / sel["file"]))


def all_models(track):
    """(name, metadata, model) for every saved supervised pipeline of a track."""
    out = []
    for meta_path in sorted((MODELS / track).glob("*.json")):
        meta = load_json(str(meta_path))
        if meta.get("task") == track and "file" in meta:
            out.append((meta["name"], meta, load_model(str(MODELS / track / meta["file"]))))
    return out


def results(name):
    return load_csv(str(RESULTS / name))


def audit():
    return load_json(str(RESULTS / "dataset_audit.json"))


def clustering_summary():
    return load_json(str(MODELS / "clustering" / "clustering_summary.json"))


def cluster_assignments(kind="kmeans"):
    fname = "kmeans_assignments.csv" if kind == "kmeans" else "agglomerative_sample_assignments.csv"
    return load_csv(str(MODELS / "clustering" / fname))


@st.cache_data(show_spinner="Reading the dataset...")
def raw_dataset(name):
    from src.data_loader import load_dataset
    return load_dataset(name)


def figure(track, name):
    path = FIGURES / track / f"{name}.png"
    return str(path) if path.exists() else None
