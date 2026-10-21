"""Clustering: explore the meteorite clusters and assign new meteorites with K-Means."""
import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import data, logic, theme

MAP_SAMPLE = 8_000


def _profiles(summary, key):
    prof = pd.DataFrame(summary[key])
    cluster_col = "kmeans" if "kmeans" in prof.columns else "agglo"
    prof = prof.rename(columns={cluster_col: "cluster"})
    if "pct_antarctic" in prof:
        prof["name"] = prof.apply(logic.describe_cluster, axis=1)
    return prof


def _named(assign, prof):
    names = dict(zip(prof["cluster"], prof["name"])) if "name" in prof else {}
    return assign.assign(group=assign["cluster"].map(lambda c: f"{c}: {names.get(c, 'cluster')}"))


def _world(df, color, title, extra=None, order=None):
    fig = px.scatter_geo(df, lat="reclat", lon="reclong", color=color, opacity=0.55,
                         category_orders={color: order} if order else None,
                         hover_data={"name": True, "mass": ":,.1f", "year": ":.0f", "reclat": False, "reclong": False})
    fig.update_traces(marker=dict(size=4))
    if extra is not None and len(extra):
        fig.add_trace(go.Scattergeo(lat=extra["reclat"], lon=extra["reclong"], mode="markers", name="your meteorite(s)",
                                    marker=dict(size=13, symbol="star", color=theme.INK, line=dict(color="white", width=1))))
    fig.update_geos(projection_type="natural earth", showcountries=True, countrycolor="#C9D2CE", showland=True,
                    landcolor="#EEF2EF", oceancolor="#FFFFFF", showocean=True, coastlinecolor="#AAB6B1")
    fig.update_layout(title=title, legend_title_text="cluster", margin=dict(l=0, r=0, t=50, b=0))
    return theme.style(fig, 470)


def _predict(df_valid, model, prof):
    clusters = model.predict(df_valid[logic.CLUSTER_COLUMNS])
    distances = model.transform(df_valid[logic.CLUSTER_COLUMNS])
    names = dict(zip(prof["cluster"], prof["name"]))
    out = df_valid.copy()
    out["cluster"] = clusters
    out["cluster name"] = [names.get(c, "") for c in clusters]
    out["distance to centroid"] = distances[np.arange(len(out)), clusters]
    return out


def render():
    theme.page_header("clustering", "Meteorite landings",
                      "Groups found without any labels, using only where, when and how heavy each meteorite was.")
    summary = data.clustering_summary()
    model = data.load_model(str(data.MODELS / "clustering" / "k_means.joblib"))
    pca = data.load_model(str(data.MODELS / "clustering" / "pca_projection.joblib"))
    prof = _profiles(summary, "kmeans_profile")
    assign = _named(data.cluster_assignments("kmeans"), prof)
    order = sorted(assign["group"].unique())

    tab_map, tab_one, tab_csv, tab_agglo, tab_how = st.tabs(
        ["Cluster map", "Assign one meteorite", "Assign a CSV file", "Agglomerative clusters", "How this works"])

    with tab_map:
        display = assign.sample(min(MAP_SAMPLE, len(assign)), random_state=42).sort_values("group")
        st.plotly_chart(_world(display, "group", f"K-Means clusters (k = {summary['k']})", order=order), width="stretch")
        st.caption(f"For speed the map draws a random {len(display):,} of the {len(assign):,} clustered meteorites; "
                   "the table below uses all of them.")
        table = prof[["cluster", "name", "n", "median_lat", "median_long", "median_mass_g", "median_year",
                      "pct_antarctic", "pct_fell", "top_recclass"]].rename(columns={
            "n": "meteorites", "median_lat": "median latitude", "median_long": "median longitude",
            "median_mass_g": "median mass (g)", "median_year": "median year", "pct_antarctic": "% in Antarctica",
            "pct_fell": "% seen falling", "top_recclass": "most common classes"})
        st.dataframe(table, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.1f") for c in
                                    ["median latitude", "median longitude", "% in Antarctica", "% seen falling"]})
        st.caption("Cluster names are given by fixed rules on these profiles (for example, over half in Antarctica "
                   "gives 'Antarctic finds'). '% seen falling' uses the fall label, which was hidden from the "
                   "algorithm and is only used here to interpret the groups.")
        st.markdown("#### Quality of the clustering")
        metrics = data.results("clustering_results.csv")
        st.dataframe(metrics, hide_index=True, width="stretch",
                     column_config={"Silhouette": st.column_config.NumberColumn(format="%.3f"),
                                    "Davies-Bouldin": st.column_config.NumberColumn(format="%.3f"),
                                    "Calinski-Harabasz": st.column_config.NumberColumn(format="%.0f")})
        st.caption("Silhouette: -1 to 1, higher is better. Davies-Bouldin: 0 or more, lower is better. "
                   "Calinski-Harabasz: higher is better, and only comparable on the same rows.")

    with tab_one:
        st.markdown("Enter one meteorite. K-Means assigns it to the nearest cluster centre, after the same "
                    "transformation used in training: unit-sphere coordinates, log10 mass and scaling.")
        with st.form("cluster_one"):
            a, b = st.columns(2)
            lat = a.number_input("Latitude (degrees, -90 to 90)", -90.0, 90.0, value=-76.0, step=0.5)
            lon = b.number_input("Longitude (degrees, -180 to 180)", -180.0, 180.0, value=156.0, step=0.5)
            mass = a.number_input("Mass (grams)", min_value=0.01, max_value=1e8, value=25.0, step=1.0)
            year = b.number_input(f"Year of fall or find ({logic.YEAR_RANGE[0]} to {logic.YEAR_RANGE[1]})",
                                  logic.YEAR_RANGE[0], logic.YEAR_RANGE[1], value=1995, step=1)
            go_one = st.form_submit_button("Assign to a cluster", type="primary")
        if go_one:
            valid, rejected, errors = logic.validate_cluster_frame(
                pd.DataFrame([{"reclat": lat, "reclong": lon, "mass": mass, "year": year}]))
            if errors or valid is None or valid.empty:
                for msg in errors or rejected["reason"].tolist():
                    st.error(msg)
            else:
                try:
                    res = _predict(valid, model, prof).iloc[0]
                except Exception:
                    st.error("This meteorite could not be assigned. Check the values and try again.")
                    return
                c = int(res["cluster"])
                p = prof.set_index("cluster").loc[c]
                theme.result_panel("clustering", "Assigned cluster", f"{c}: {res['cluster name']}",
                                   f"Members typically: latitude {p['median_lat']:.0f}°, mass {p['median_mass_g']:,.0f} g, "
                                   f"year {p['median_year']:.0f}; {p['pct_fell']:.1f}% were seen falling.")
                dists = model.transform(valid[logic.CLUSTER_COLUMNS])[0]
                st.dataframe(pd.DataFrame({"cluster": [f"{k}: {n}" for k, n in zip(prof["cluster"], prof["name"])],
                                           "distance to centre (scaled units)": dists[prof["cluster"].to_numpy()]}),
                             hide_index=True, width="stretch")
                st.plotly_chart(_world(display, "group", "Your meteorite on the cluster map",
                                       extra=valid, order=order), width="stretch")

    with tab_csv:
        st.markdown(f"Upload a CSV with the columns **reclat, reclong, mass, year**; any other columns are kept "
                    f"and ignored. Limits: {logic.MAX_UPLOAD_ROWS:,} rows, {logic.MAX_UPLOAD_MB} MB. "
                    "The file is only read as data.")
        template = pd.DataFrame({"name": ["example A", "example B"], "reclat": [-72.5, 35.2],
                                 "reclong": [75.3, -101.8], "mass": [12.4, 5300.0], "year": [1988, 1930]})
        st.download_button("Download a template CSV", template.to_csv(index=False), "meteorites_template.csv", "text/csv")
        upload = st.file_uploader("Choose a CSV file", type=["csv"])
        if upload is not None:
            if upload.size > logic.MAX_UPLOAD_MB * 1_048_576:
                st.error(f"The file is {upload.size / 1_048_576:.1f} MB; the limit is {logic.MAX_UPLOAD_MB} MB.")
            else:
                try:
                    frame = pd.read_csv(io.BytesIO(upload.getvalue()))
                except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError):
                    st.error("The file could not be read as a comma-separated table. Save it as CSV (UTF-8) and try again.")
                    frame = None
                if frame is not None:
                    valid, rejected, errors = logic.validate_cluster_frame(frame)
                    for e in errors:
                        st.error(e)
                    if not errors:
                        st.markdown(f"**{len(valid):,}** rows assigned, **{len(rejected):,}** rejected.")
                        if len(valid):
                            res = _predict(valid, model, prof)
                            counts = res["cluster name"].value_counts().rename_axis("cluster").reset_index(name="rows")
                            st.dataframe(counts, hide_index=True)
                            st.dataframe(res, hide_index=True, width="stretch")
                            st.download_button("Download the assignments", res.to_csv(index=False),
                                               "meteorite_cluster_assignments.csv", "text/csv")
                            pts = pca.transform(model[:-1].transform(valid[logic.CLUSTER_COLUMNS]))
                            fig = px.scatter(display, x="pc1", y="pc2", color="group", opacity=0.35,
                                             category_orders={"group": order}, render_mode="webgl")
                            fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="markers", name="uploaded rows",
                                                     marker=dict(size=11, symbol="star", color=theme.INK)))
                            fig.update_layout(title="Uploaded rows on the PCA projection", xaxis_title="PC1",
                                              yaxis_title="PC2", legend_title_text="cluster")
                            st.plotly_chart(theme.style(fig, 420), width="stretch")
                        if len(rejected):
                            st.markdown("Rejected rows and why:")
                            st.dataframe(rejected, hide_index=True, width="stretch")

    with tab_agglo:
        agg_prof = _profiles(summary, "agglomerative_profile")
        agg = data.cluster_assignments("agglomerative")
        agg = agg.assign(group=agg["cluster"].map(lambda c: f"cluster {c}"))
        theme.note("Agglomerative clustering builds a hierarchy over the points it was given. It has no rule for "
                   "placing a new point, so scikit-learn's AgglomerativeClustering has no <code>predict</code> method. "
                   "This tab therefore shows the clusters it found in its 10,000-meteorite training sample, and new "
                   "meteorites are assigned with K-Means only.", "warn")
        st.plotly_chart(_world(agg.sort_values("group"), "group",
                               f"Agglomerative clustering ({summary['linkage']} linkage) on the 10,000-row sample",
                               order=sorted(agg["group"].unique())), width="stretch")
        st.dataframe(agg_prof.rename(columns={"n": "meteorites"}), hide_index=True, width="stretch")
        st.caption(f"Agreement with K-Means on the same sample: adjusted Rand index "
                   f"{summary['kmeans_vs_agglomerative_ari']:.3f} (1 = identical partitions).")

    with tab_how:
        st.markdown(
            f"""
**Unsupervised learning** looks for structure without any answer key. Nobody told the algorithms which
meteorites belong together; they grouped records that are close in a 5-number description:

1. latitude and longitude converted to a point on a unit sphere, so that -179.9° and +179.9° count as neighbours,
2. log10 of the mass, so that a 60-tonne meteorite does not dominate,
3. the year,

with each number rescaled to mean 0 and standard deviation 1.

**Choosing k = {summary['k']}.** The silhouette and Davies-Bouldin indices both pointed to {summary['k']}
(votes: {summary['k_votes']}), the elbow curve bends there, and refitting on bootstrap resamples reproduced the
same clusters (mean adjusted Rand index {summary['stability']['bootstrap_ari_mean']:.3f}).

**Limits.** The clusters mirror where meteorites are *searched for*: Antarctic ice and hot deserts. The PCA picture
keeps only {sum(summary['pca_explained_variance']):.0%} of the variation, so clusters can look more mixed there than
they really are.
"""
        )
        for name, caption in [("kmeans_elbow_curve", "Elbow curve"), ("kmeans_metrics_vs_k", "Internal indices for k = 2 to 10"),
                              ("agglomerative_dendrogram", "Dendrogram (Ward linkage, 10,000-row sample)")]:
            path = data.figure("clustering", name)
            if path:
                st.image(path, caption=caption)
