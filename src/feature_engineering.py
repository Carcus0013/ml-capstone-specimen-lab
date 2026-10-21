"""Engineered features for each track.

Each function is pure row-wise arithmetic: the value for a row depends only on that
row, never on statistics pooled across rows. That makes them leakage-free by
construction, so they are safe to apply before the train/test split and they are
placed as the first step of every saved pipeline (via FunctionTransformer). The app
therefore sends raw inputs and gets exactly the same features as training did.
"""
import numpy as np
import pandas as pd


def add_banana_features(df):
    """firmness_squared lets linear models represent the inverted-U effect of firmness
    on quality (too soft = overripe, too hard = underripe)."""
    out = pd.DataFrame(df).copy()
    out["firmness_squared"] = out["firmness_kgf"].astype(float) ** 2
    return out


def add_mushroom_features(df):
    """stalk_ring_asymmetry counts (0-2) how many stalk attributes differ above vs
    below the ring. No single one-hot column can express this relationship."""
    out = pd.DataFrame(df).copy()
    surface_differs = out["stalk-surface-above-ring"] != out["stalk-surface-below-ring"]
    color_differs = out["stalk-color-above-ring"] != out["stalk-color-below-ring"]
    out["stalk_ring_asymmetry"] = surface_differs.astype(int) + color_differs.astype(int)
    return out


def add_meteorite_features(df):
    """Convert latitude/longitude to 3-D unit-sphere coordinates and mass to log10.

    * Raw longitude wraps around: -179.9 and +179.9 are neighbours on the globe but
      far apart numerically. On the unit sphere, straight-line distance between two
      points grows monotonically with their great-circle distance, so Euclidean
      clustering algorithms see true geographic proximity.
    * Mass spans 0.01 g to 60 tonnes (heavily right-skewed). log10 turns that into a
      roughly symmetric scale, so a few giant meteorites cannot dominate distances.
    """
    out = pd.DataFrame(df).copy()
    lat = np.radians(out["reclat"].astype(float))
    lon = np.radians(out["reclong"].astype(float))
    out["geo_x"] = np.cos(lat) * np.cos(lon)
    out["geo_y"] = np.cos(lat) * np.sin(lon)
    out["geo_z"] = np.sin(lat)
    out["log10_mass"] = np.log10(out["mass"].astype(float))
    return out
