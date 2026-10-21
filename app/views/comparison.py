"""Model comparison: the saved evaluation results for every algorithm in every track."""
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import data, theme


def _params(value):
    if isinstance(value, str):
        return value
    return json.dumps(value)


def _bar(df, x, title, track, selected, xrange=None):
    colors = [theme.TRACK[track]["color"] if m == selected else "#B8C4BF" for m in df["Model"]]
    fig = go.Figure(go.Bar(x=df[x], y=df["Model"], orientation="h", marker_color=colors,
                           text=[f"{v:.4f}" for v in df[x]], textposition="outside"))
    fig.update_layout(title=title, xaxis_title=x, yaxis=dict(autorange="reversed", title=""))
    if xrange:
        fig.update_xaxes(range=xrange)
    return theme.style(fig, 60 + 34 * len(df))


def _inspect(df, key, metrics):
    name = st.selectbox("Inspect one model", df["Model"], key=key)
    row = df[df["Model"] == name].iloc[0]
    cols = st.columns(len(metrics))
    for c, m in zip(cols, metrics):
        c.metric(m, f"{row[m]:.4f}" if isinstance(row[m], float) else row[m])
    st.markdown(f"**Tuned hyperparameters:** `{row['Best params']}`")


def _figures(track, items):
    cols = st.columns(2)
    for i, (name, caption) in enumerate(items):
        path = data.figure(track, name)
        if path:
            cols[i % 2].image(path, caption=caption)


def render():
    theme.plain_header("Model comparison",
                       "All ten regression models, all ten classifiers and both clustering algorithms, as evaluated "
                       "in the notebooks. Every value is read from reports/results.")
    hp = data.load_json(str(data.RESULTS / "hyperparameter_results.json"))
    tab_r, tab_c, tab_k = st.tabs(["Regression", "Classification", "Clustering"])

    with tab_r:
        reg = data.results("regression_results.csv")
        sel = data.selection("regression")
        st.plotly_chart(_bar(reg, "Test R2", "Test R² of every regression model (selected model highlighted)",
                             "regression", sel["selected_model"], [max(0, reg["Test R2"].min() - 0.05), 1.01]), width="stretch")
        show = reg.drop(columns=["Best params"])
        st.dataframe(show, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.5f") for c in
                                    ["Test R2", "Test RMSE", "Test MAE", "Train R2", "CV R2 (5-fold, train)", "Baseline test R2"]})
        st.caption(f"Ranked by test R² on the same 200 held-out bananas. Selected: {sel['selected_model']} ({sel['rule']}).")
        _inspect(reg, "inspect_reg", ["Test R2", "Test RMSE", "Test MAE", "Train R2"])
        st.markdown("#### Effect of hyperparameter tuning")
        tune = pd.DataFrame(hp["regression"]).T.reset_index(names="Model")
        tune["Best params"] = tune["Best params"].map(_params)
        st.dataframe(tune, hide_index=True, width="stretch")
        cv = data.load_json(str(data.RESULTS / "regression_cv.json"))
        st.markdown("#### 5-fold cross-validated R² of the two best models")
        st.dataframe(pd.DataFrame({m: {"mean": v["mean"], "std": v["std"], **{f"fold {i + 1}": s for i, s in enumerate(v["scores"])}}
                                   for m, v in cv.items()}).T, width="stretch")
        _figures("regression", [("best_model_predicted_vs_actual", "Selected model: predicted vs actual"),
                                ("best_model_residuals", "Selected model: residuals"),
                                ("random_forest_feature_importance", "Random Forest feature importance"),
                                ("best_model_permutation_importance", "Selected model: permutation importance")])

    with tab_c:
        clf = data.results("classification_results.csv")
        sel = data.selection("classification")
        metrics = ["Accuracy", "Precision", "Recall", "F1 (weighted)", "ROC-AUC"]
        long = clf.melt(id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Score")
        fig = px.bar(long, x="Score", y="Model", color="Metric", barmode="group", orientation="h",
                     color_discrete_sequence=theme.SERIES)
        fig.update_layout(title="Five required metrics for all ten classifiers (test set)", yaxis=dict(autorange="reversed", title=""),
                          xaxis=dict(range=[max(0, clf["Accuracy"].min() - 0.01), 1.001]))
        st.plotly_chart(theme.style(fig, 640), width="stretch")
        danger = clf[["Model", "Poisonous predicted edible"]]
        fig = go.Figure(go.Bar(x=danger["Poisonous predicted edible"], y=danger["Model"], orientation="h",
                               marker_color=[theme.DANGER if v else "#B8C4BF" for v in danger["Poisonous predicted edible"]]))
        fig.update_layout(title="Poisonous mushrooms predicted edible (test set, lower is safer)",
                          xaxis=dict(title="count", dtick=1), yaxis=dict(autorange="reversed", title=""))
        st.plotly_chart(theme.style(fig, 400), width="stretch")
        st.dataframe(clf.drop(columns=["Best params"]), hide_index=True, width="stretch")
        st.caption(f"Selected: {sel['selected_model']} ({sel['rule']}).")
        _inspect(clf, "inspect_clf", metrics + ["Poisonous predicted edible"])
        st.markdown("#### Effect of hyperparameter tuning")
        tune = pd.DataFrame(hp["classification"]).T.reset_index(names="Model")
        tune["Best params"] = tune["Best params"].map(_params)
        st.dataframe(tune, hide_index=True, width="stretch")
        cv = data.load_json(str(data.RESULTS / "classification_cv.json"))
        st.markdown("#### Stratified 5-fold cross-validation (training set)")
        st.dataframe(pd.DataFrame({m: {k: pd.Series(v).mean() for k, v in s.items()} for m, s in cv.items()}).T
                     .rename(columns={"log_loss": "neg log-loss"}), width="stretch")
        _figures("classification", [("confusion_matrices_all", "Confusion matrices"), ("roc_curves_all", "ROC curves"),
                                    ("decision_tree_plot", "Tuned decision tree (top levels)"),
                                    ("random_forest_feature_importance", "Random Forest feature importance")])

    with tab_k:
        res = data.results("clustering_results.csv")
        st.dataframe(res, hide_index=True, width="stretch")
        st.caption("Silhouette (higher is better), Davies-Bouldin (lower is better), Calinski-Harabasz (higher is "
                   "better, and only comparable on the same rows).")
        ksel = data.results("clustering_k_selection.csv")
        fig = go.Figure()
        for metric, color in [("Silhouette", theme.SERIES[0]), ("Davies-Bouldin", theme.SERIES[3])]:
            fig.add_trace(go.Scatter(x=ksel["k"], y=ksel[metric], mode="lines+markers", name=metric, line_color=color))
        fig.update_layout(title="K-Means: internal indices for k = 2 to 10", xaxis_title="k", yaxis_title="index value")
        st.plotly_chart(theme.style(fig, 360), width="stretch")
        st.markdown("#### Agglomerative linkage comparison (10,000-row sample, k = 4)")
        st.dataframe(data.results("clustering_linkage_comparison.csv"), hide_index=True, width="stretch")
        st.caption("Single, complete and average linkage isolate a few outliers and put almost everything else in "
                   "one cluster. Their high silhouette shows why indices must be read alongside cluster sizes.")
        _figures("clustering", [("pca_clusters_both", "PCA projection, both algorithms"),
                                ("tsne_clusters_both", "t-SNE, both algorithms"),
                                ("agglomerative_dendrogram", "Dendrogram"), ("metric_comparison", "Metric comparison")])
