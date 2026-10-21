"""Plot helpers with a consistent, colour-blind-friendly style (guideline 7.3).

Every helper gives its figure a title, labelled axes and a legend where one is
meaningful, and save_fig() writes it with bbox_inches='tight' so labels are not clipped.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

from src import config
from src.utils import ensure_dir

PALETTE = sns.color_palette("colorblind")


def set_plot_style():
    sns.set_theme(style="whitegrid", palette="colorblind")
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 120,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
    })


def save_fig(fig, track, name):
    """Save to reports/figures/<track>/<name>.png and return the path."""
    folder = ensure_dir(config.FIGURES_DIR / track)
    path = folder / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    return path


def plot_predicted_vs_actual(y_true, y_pred, title):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y_true, y_pred, s=18, alpha=0.6, color=PALETTE[0], label="test samples")
    lims = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
    ax.plot(lims, lims, "--", color=PALETTE[3], linewidth=2, label="perfect prediction (y = x)")
    ax.set_title(title)
    ax.set_xlabel("Actual value")
    ax.set_ylabel("Predicted value")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_residuals(y_true, y_pred, title):
    """Two panels: residuals vs predicted (should be patternless) and their histogram."""
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].scatter(y_pred, residuals, s=18, alpha=0.6, color=PALETTE[0], label="test samples")
    axes[0].axhline(0, color=PALETTE[3], linestyle="--", linewidth=2, label="zero residual")
    axes[0].set_title(f"{title}: residuals vs predicted")
    axes[0].set_xlabel("Predicted value")
    axes[0].set_ylabel("Residual (actual - predicted)")
    axes[0].legend()
    axes[1].hist(residuals, bins=30, color=PALETTE[0], edgecolor="white", label="residuals")
    axes[1].axvline(0, color=PALETTE[3], linestyle="--", linewidth=2, label="zero")
    axes[1].set_title(f"{title}: residual distribution")
    axes[1].set_xlabel("Residual (actual - predicted)")
    axes[1].set_ylabel("Number of test samples")
    axes[1].legend()
    fig.tight_layout()
    return fig


def plot_feature_importance(names, importances, title, top_n=15, xlabel="Importance"):
    s = pd.Series(importances, index=names).sort_values(ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(s) + 1.5))
    ax.barh(s.index, s.values, color=PALETTE[0], label=xlabel.lower())
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Feature")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_metric_bars(df, metrics, title, name_col="Model", xlim=None):
    """Grouped horizontal bar chart comparing models on several metrics."""
    data = df.set_index(name_col)[metrics]
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(data) * max(1, len(metrics) / 2) + 1.5))
    data.plot.barh(ax=ax, color=PALETTE[: len(metrics)], width=0.8)
    ax.set_title(title)
    ax.set_xlabel("Score")
    ax.set_ylabel("Model")
    if xlim:
        ax.set_xlim(*xlim)
    ax.legend(title="Metric", loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def plot_confusion_grid(cms, class_names, ncols=5, title="Confusion matrices (test set)"):
    """cms: dict model name -> 2-D confusion matrix (rows = actual, columns = predicted)."""
    n = len(cms)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.3 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, cm) in zip(axes, cms.items()):
        ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Actual label")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_elbow(ks, inertias, chosen_k, title="Elbow curve: K-Means inertia vs k"):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ks, inertias, marker="o", color=PALETTE[0], label="inertia (within-cluster SS)")
    ax.axvline(chosen_k, color=PALETTE[3], linestyle="--", label=f"chosen k = {chosen_k}")
    ax.set_title(title)
    ax.set_xlabel("Number of clusters k")
    ax.set_ylabel("Inertia")
    ax.set_xticks(list(ks))
    ax.legend()
    fig.tight_layout()
    return fig


def plot_clusters_2d(embedding, labels, title, xlabel, ylabel, ax=None):
    """Scatter of a 2-D projection coloured by cluster label."""
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(7.5, 6))
    labels = np.asarray(labels)
    for i, lab in enumerate(np.unique(labels)):
        mask = labels == lab
        ax.scatter(embedding[mask, 0], embedding[mask, 1], s=6, alpha=0.55,
                   color=PALETTE[i % len(PALETTE)], label=f"cluster {lab} (n={mask.sum():,})")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(markerscale=3, fontsize=8, loc="best")
    if created:
        fig.tight_layout()
        return fig
    return ax.figure
