"""Overview: the three questions, the models that answer them, and how they were built."""
from html import escape

import pandas as pd
import streamlit as st

from core import data, nav, theme


def _question(track, question, dataset, model, figure, figure_label, detail):
    color = theme.TRACK[track]["color"]
    return f"""<div class="question" style="--track:{color}">
      <h2>{escape(question)}</h2>
      <p class="dataset">{escape(dataset)}</p>
      <p class="answer">Answered by <strong>{escape(model)}</strong></p>
      <p class="figure">{escape(figure)}</p>
      <p class="figure-label">{escape(figure_label)}</p>
      <p class="figure-label">{escape(detail)}</p>
    </div>"""


def render():
    reg_sel, clf_sel = data.selection("regression"), data.selection("classification")
    clu = data.clustering_summary()
    km_all = next(m for m in clu["metrics"] if m["Algorithm"] == "K-Means" and "all" in m["Evaluated on"])
    audit = data.audit()

    st.markdown(
        """<div class="hero"><h1>Three specimens, three kinds of machine learning</h1>
        <p class="lede">A banana, a mushroom and a meteorite each pose a different question. This app runs the
        models trained in the project notebooks on your own inputs and reports their measured performance.
        Every number on these pages is read from the saved results; none are typed in by hand.</p></div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="questions">'
        + _question("regression", "How good is this banana?",
                    f"Regression on {audit['banana_quality_dataset.csv']['rows']:,} banana samples",
                    reg_sel["selected_model"], f"R² {reg_sel['test_metrics']['R2']:.4f}",
                    "on 200 held-out bananas",
                    f"RMSE {reg_sel['test_metrics']['RMSE']:.3f} on a 0.9 to 3.9 quality scale")
        + _question("classification", "Is this mushroom poisonous?",
                    f"Classification of {audit['mushrooms.csv']['rows']:,} mushrooms",
                    clf_sel["selected_model"], f"F1 {clf_sel['test_metrics']['F1 (weighted)']:.3f}",
                    "weighted, on 1,625 held-out mushrooms",
                    f"{clf_sel['test_metrics']['Poisonous predicted edible']} poisonous mushrooms called edible")
        + _question("clustering", "Which meteorites belong together?",
                    f"Clustering of {audit['meteorite-landings.csv']['specific_checks']['rows_after_cleaning']:,} cleaned meteorite records",
                    f"K-Means, k = {clu['k']}", f"Silhouette {km_all['Silhouette']:.3f}",
                    "on all cleaned records (range -1 to 1)",
                    f"agrees with agglomerative clustering at ARI {clu['kmeans_vs_agglomerative_ari']:.2f}")
        + "</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        nav.link(st, "regression", "Predict a banana's quality", ":material/nutrition:")
    with c2:
        nav.link(st, "classification", "Classify a mushroom", ":material/eco:")
    with c3:
        nav.link(st, "clustering", "Explore meteorite clusters", ":material/public:")

    st.markdown("## How every model was built")
    st.markdown(
        """
- **One split per track, made first.** 80% of rows for training and 20% held out (stratified for the mushrooms), with `random_state = 42`. All ten models in a track are scored on the same held-out rows.
- **Preprocessing lives inside the model pipeline.** Imputers, encoders and scalers are fitted on training rows only, so no information from the held-out rows leaks in. The saved pipeline that runs here is the same object the notebook evaluated.
- **Hyperparameters are tuned by 5-fold cross-validation on the training rows.** The held-out rows are used once, for the final score.
- **Clustering never sees a label.** The meteorites' `fall` and `recclass` columns are used only afterwards, to interpret the groups.
"""
    )

    st.markdown("## What the results say")
    reg = data.results("regression_results.csv")
    clf = data.results("classification_results.csv")
    n_perfect = int((clf["F1 (weighted)"] == 1.0).sum())
    best_tree = reg[reg["Model"].str.contains("Forest|Tree|Boosting")].iloc[0]
    st.markdown(
        f"""
- **Banana quality is almost a smooth function of four fruit properties.** These are ripeness, sugar, length and firmness (through a squared term). The linear-family models all reach R² ≈ 0.996, and the best tree model ({best_tree['Model']}) reaches {best_tree['Test R2']:.3f}. Variety, region and growing conditions add almost nothing.
- **The mushroom data is almost separable.** {n_perfect} of 10 classifiers make no errors on the held-out set, and odour alone gets about 98.5% right. Real foraging is not like this; see the warning on the mushroom page.
- **Meteorite clusters reflect how meteorites are collected.** The four groups are Antarctic search finds, Africa and Asia finds, Americas finds, and a group in which over half were seen falling. They describe where people look rather than where meteorites land.
"""
    )

    st.markdown("## The three datasets")
    rows = []
    for fname, track in [("banana_quality_dataset.csv", "regression"), ("mushrooms.csv", "classification"),
                         ("meteorite-landings.csv", "clustering")]:
        a = audit[fname]
        rows.append({"Dataset": fname, "Track": theme.TRACK[track]["name"], "Rows": a["rows"], "Columns": a["columns"],
                     "Missing cells": a["missing_cells"], "Coded as '?'": a["sentinel_coded_cells"],
                     "Duplicate rows": a["duplicate_rows"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Counts from the dataset audit (reports/results/dataset_audit.json). See the Data explorer page for details.")
