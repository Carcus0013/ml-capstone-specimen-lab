"""Data explorer: audit summaries, column distributions and EDA figures for the three datasets."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from core import data, theme

DATASETS = {
    "Banana quality (regression)": ("banana", "banana_quality_dataset.csv", "regression"),
    "Mushrooms (classification)": ("mushrooms", "mushrooms.csv", "classification"),
    "Meteorite landings (clustering)": ("meteorites", "meteorite-landings.csv", "clustering"),
}
EDA_FIGURES = {
    "regression": [("eda_target_distribution", "Target distribution"), ("eda_correlation_heatmap", "Correlation heatmap"),
                   ("eda_feature_target_scatter", "Feature vs target"), ("fe_firmness_residual_diagnostic",
                                                                         "Why firmness² was engineered")],
    "classification": [("eda_class_distribution", "Class balance"), ("eda_mutual_information", "Information about the class"),
                       ("eda_cramers_v_heatmap", "Association between columns (Cramér's V)"),
                       ("eda_feature_distributions_by_class", "Every feature, split by class")],
    "clustering": [("eda_world_scatter", "Where meteorites are recorded"), ("eda_missing_values", "Missing-value pattern"),
                   ("eda_mass_year_distributions", "Mass and year"), ("eda_correlation_heatmap", "Correlations")],
}
EXPLAIN = {
    "regression": "Each row is one banana. The model learns to predict the continuous quality_score from the other measurements.",
    "classification": "Each row is one mushroom described by 22 one-letter codes. The model learns the edible/poisonous label.",
    "clustering": "Each row is one meteorite. No label is predicted: the algorithms group similar records on their own.",
}


def render():
    theme.plain_header("Data explorer", "What each dataset contains, how clean it is, and what the exploratory "
                                        "analysis found. Figures are those produced by the notebooks.")
    choice = st.radio("Dataset", list(DATASETS), horizontal=True)
    key, fname, track = DATASETS[choice]
    audit = data.audit()[fname]
    st.markdown(EXPLAIN[track])

    c = st.columns(5)
    c[0].metric("Rows", f"{audit['rows']:,}")
    c[1].metric("Columns", audit["columns"])
    c[2].metric("Missing cells", f"{audit['missing_cells']:,}")
    c[3].metric("Coded missing ('?')", f"{audit['sentinel_coded_cells']:,}")
    c[4].metric("Duplicate rows", f"{audit['duplicate_rows']:,}")

    st.markdown("#### Columns")
    cols = pd.DataFrame(audit["column_details"])[["column", "dtype", "kind", "missing", "missing_pct", "sentinel_coded",
                                                  "n_unique", "constant", "identifier_like"]]
    st.dataframe(cols, hide_index=True, width="stretch")

    st.markdown("#### Look at one column")
    df = data.raw_dataset(key)
    col = st.selectbox("Column", [c for c in df.columns if c not in ("sample_id", "id", "name", "GeoLocation")])
    s = df[col].dropna()
    if pd.api.types.is_numeric_dtype(s):
        log = col == "mass" and st.checkbox("Use a log10 scale (mass spans eight orders of magnitude)", value=True)
        values = np.log10(s[s > 0]) if log else s
        fig = px.histogram(values, nbins=50, color_discrete_sequence=[theme.TRACK[track]["color"]])
        fig.update_layout(title=f"Distribution of {'log10 ' if log else ''}{col}", xaxis_title=f"{'log10 ' if log else ''}{col}",
                          yaxis_title="Count", showlegend=False)
        st.caption(f"min {s.min():,.3f}, median {s.median():,.3f}, max {s.max():,.3f}, missing {df[col].isna().sum():,}")
    else:
        counts = s.value_counts().head(25).rename_axis(col).reset_index(name="count")
        fig = px.bar(counts, x=col, y="count", color_discrete_sequence=[theme.TRACK[track]["color"]])
        fig.update_layout(title=f"Most frequent values of {col} (up to 25 shown)", xaxis_title=col, yaxis_title="Count")
        st.caption(f"{s.nunique():,} distinct values")
    st.plotly_chart(theme.style(fig, 360), width="stretch")

    st.markdown("#### Findings from the exploratory analysis")
    figs = EDA_FIGURES[track]
    pair = st.columns(2)
    for i, (name, caption) in enumerate(figs):
        path = data.figure(track, name)
        if path:
            pair[i % 2].image(path, caption=caption)

    with st.expander("Preview the first 8 rows"):
        st.dataframe(df.head(8), hide_index=True, width="stretch")
