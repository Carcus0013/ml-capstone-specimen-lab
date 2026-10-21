"""About: method summary, artefact provenance, AI-use disclosure and limitations."""
import pandas as pd
import streamlit as st

from core import data, theme


def render():
    theme.plain_header("About this project", "23CSE301 Machine Learning capstone: how the work was done and what "
                                             "the results can and cannot support.")
    st.markdown(
        """
### Method in brief
- **Regression:** 10 algorithms on `banana_quality_dataset.csv`, predicting `quality_score`. `quality_category` was removed as target leakage, and `firmness_squared` was engineered from a residual diagnostic on the training set.
- **Classification:** 10 algorithms on `mushrooms.csv`, with a stratified split. `?` in `stalk-root` is kept as its own category, and `stalk_ring_asymmetry` was engineered.
- **Clustering:** K-Means and agglomerative clustering on cleaned meteorite records, using unit-sphere coordinates, log10 mass and year. The `fall` and `recclass` labels were used only afterwards, to interpret the clusters.
- Every learned preprocessing step sits inside a scikit-learn `Pipeline`, so it is fitted on training rows only. The app loads those exact saved pipelines and never retrains.

### Where the numbers come from
Notebooks in `notebooks/` write every metric to `reports/results/` and every model to `models/`. This app only reads
those files. If a model file is missing, the affected page says so instead of inventing an answer.
"""
    )
    rows = []
    for track in ("regression", "classification"):
        sel = data.selection(track)
        meta = data.load_json(str(data.MODELS / track / sel["file"].replace(".joblib", ".json")))
        rows.append({"Track": track, "Selected model": sel["selected_model"], "File": f"models/{track}/{sel['file']}",
                     "Saved (UTC)": meta["saved_utc"], "scikit-learn": meta["environment"]["scikit-learn"],
                     "Python": meta["environment"]["python"]})
    meta = data.load_json(str(data.MODELS / "clustering" / "k_means.json"))
    rows.append({"Track": "clustering", "Selected model": f"K-Means (k = {meta['k']})", "File": "models/clustering/k_means.joblib",
                 "Saved (UTC)": meta["saved_utc"], "scikit-learn": meta["environment"]["scikit-learn"],
                 "Python": meta["environment"]["python"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.markdown(
        """
### Use of AI assistance
The code scaffolding, the shared `src/` package, this app and the first drafts of the notebook commentary were
produced with an AI assistant (Anthropic's Claude), working on the team's datasets and earlier Review 1 notebooks.
The course guidelines allow AI for code scaffolding but not for analysis or interpretation. Every interpretive
notebook cell is therefore marked **DRAFT** for the team to verify against the outputs and rewrite in their own words.

### Limitations
- The banana data appears synthetic, and the mushroom data is nearly separable. Their near-perfect scores reflect the data more than real-world skill.
- The meteorite clusters mirror where meteorites are searched for (Antarctica, hot deserts), not where they fall.
- Agglomerative clustering was fitted on a 10,000-row random sample, because the full pairwise-distance matrix would need about 4 GB.
- The free hosting plan may put the app to sleep after a period without visitors; the first visit after that takes a little longer to load.
"""
    )
