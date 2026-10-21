"""Regression: predict a banana's quality score with the selected saved pipeline."""
import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import data, logic, theme

LABELS = {
    "ripeness_index": "Ripeness index (1 to 7)",
    "sugar_content_brix": "Sugar content (°Brix)",
    "firmness_kgf": "Firmness (kgf)",
    "length_cm": "Length (cm)",
    "weight_g": "Weight (g)",
    "tree_age_years": "Tree age (years)",
    "altitude_m": "Altitude (m)",
    "rainfall_mm": "Annual rainfall (mm)",
    "soil_nitrogen_ppm": "Soil nitrogen (ppm)",
    "variety": "Variety",
    "region": "Growing region",
}
FRUIT = ["ripeness_index", "sugar_content_brix", "firmness_kgf", "length_cm", "weight_g"]
GROWING = ["tree_age_years", "altitude_m", "rainfall_mm", "soil_nitrogen_ppm"]


def _step(lo, hi):
    return float(10 ** math.floor(math.log10(max(hi - lo, 1e-9) / 100)))


def _fill(values, bounds):
    """Callback: copy an example (or the medians) into the form widgets."""
    filled = {}
    for col, value in values.items():
        if col == "actual_quality_score":
            continue
        if col in bounds:
            lo, hi = bounds[col]
            value = min(max(float(value), lo), hi)
        st.session_state[f"reg_{col}"] = value
        filled[col] = value
    st.session_state["reg_example"] = {"values": filled, "actual": values.get("actual_quality_score")}


def render():
    theme.page_header("regression", "Banana quality",
                      "Enter a banana's measurements. The selected model predicts its quality score, "
                      "using the same preprocessing it was trained with.")
    sel, model = data.selected_model("regression")
    schema = data.schema("regression")
    bounds = logic.numeric_bounds(schema)
    medians = {c: s["median"] for c, s in schema["numeric"].items()}
    medians.update({c: levels[0] for c, levels in schema["categorical"].items()})
    if "reg_ripeness_index" not in st.session_state:
        _fill(medians, bounds)

    st.markdown("Start from a banana in the held-out test set, or type your own values.")
    cols = st.columns(len(schema["examples"]) + 1)
    for i, ex in enumerate(schema["examples"]):
        cols[i].button(f"Test banana {i + 1}", on_click=_fill, args=(ex, bounds), width="stretch",
                       help=f"A real held-out sample; its measured score is {ex['actual_quality_score']:.2f}")
    cols[-1].button("Reset to medians", on_click=_fill, args=(medians, bounds), width="stretch")

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        with st.form("regression_form"):
            st.markdown("### The fruit")
            a, b = st.columns(2)
            for i, col in enumerate(FRUIT):
                lo, hi = bounds[col]
                (a if i % 2 == 0 else b).number_input(LABELS[col], min_value=lo, max_value=hi, step=_step(lo, hi),
                                                      key=f"reg_{col}", format="%.2f")
            st.markdown("### Where and how it grew")
            a, b = st.columns(2)
            a.selectbox(LABELS["variety"], schema["categorical"]["variety"], key="reg_variety")
            b.selectbox(LABELS["region"], schema["categorical"]["region"], key="reg_region")
            for i, col in enumerate(GROWING):
                lo, hi = bounds[col]
                (a if i % 2 == 0 else b).number_input(LABELS[col], min_value=lo, max_value=hi, step=_step(lo, hi),
                                                      key=f"reg_{col}", format="%.2f")
            submitted = st.form_submit_button("Predict quality score", type="primary", width="stretch")
            st.caption("Inputs are limited to the range seen in training plus a 5% margin.")

    with right:
        if submitted:
            values = {c: st.session_state.get(f"reg_{c}") for c in list(schema["numeric"]) + list(schema["categorical"])}
            row, errors, warnings = logic.validate_regression_input(values, schema)
            if errors:
                for e in errors:
                    st.error(e)
                return
            try:
                prediction = float(model.predict(row)[0])
            except Exception:  # never show a raw stack trace to the user
                st.error("The model could not produce a prediction for these inputs. Check the values and try again.")
                return
            st.session_state["reg_last"] = {"row": row, "prediction": prediction, "warnings": warnings}

        last = st.session_state.get("reg_last")
        if not last:
            theme.note("Fill in the form and press <b>Predict quality score</b>. The prediction appears here.")
            return
        prediction = last["prediction"]
        band = logic.quality_band(prediction)
        theme.result_panel("regression", "Predicted quality score", f"{prediction:.2f}",
                           f"In the dataset's <b>{band}</b> band. Model: {sel['selected_model']}, whose "
                           f"error on 200 held-out bananas averages {sel['test_metrics']['MAE']:.3f} (MAE).")
        example = st.session_state.get("reg_example") or {}
        if example.get("actual") is not None and last["row"].iloc[0].to_dict() == example["values"]:
            st.markdown(f"This test banana's measured score is **{example['actual']:.2f}**, so the prediction is "
                        f"off by {abs(prediction - example['actual']):.3f}.")
        for w in last["warnings"]:
            theme.note(w, "warn")

        scores = data.raw_dataset("banana")["quality_score"]
        fig = go.Figure(go.Histogram(x=scores, nbinsx=40, marker_color=theme.TRACK["regression"]["soft"],
                                     marker_line=dict(color=theme.TRACK["regression"]["color"], width=1),
                                     name="all 1,000 bananas"))
        fig.add_vline(x=prediction, line_width=3, line_color=theme.INK,
                      annotation_text=f"this banana: {prediction:.2f}", annotation_position="top")
        fig.update_layout(title="Where the prediction sits among the 1,000 bananas", xaxis_title="quality_score",
                          yaxis_title="Number of bananas", showlegend=False)
        st.plotly_chart(theme.style(fig, 300), width="stretch")

    if st.session_state.get("reg_last"):
        row = st.session_state["reg_last"]["row"]
        with st.expander("What every trained model predicts for this banana"):
            rows = []
            for name, meta, m in data.all_models("regression"):
                rows.append({"Model": name, "Prediction": float(m.predict(row)[0]),
                             "Held-out R²": meta["test_metrics"]["R2"], "Selected": "yes" if meta["selected"] else ""})
            table = pd.DataFrame(rows).sort_values("Held-out R²", ascending=False)
            st.dataframe(table, hide_index=True, width="stretch",
                         column_config={"Prediction": st.column_config.NumberColumn(format="%.3f"),
                                        "Held-out R²": st.column_config.NumberColumn(format="%.4f")})
            st.caption("All ten models were tuned in the regression notebook. Models with lower held-out R² "
                       "(KNN, the single decision tree) tend to disagree most with the rest.")
        with st.expander("Inputs used"):
            st.dataframe(row.T.rename(columns={0: "value"}).rename(index=LABELS).astype(str), width="stretch")

    st.markdown("### Limitations")
    st.markdown(
        "- The training data appears synthetic, with uniform features and an almost deterministic target, so "
        "real bananas would not be predicted this accurately.\n"
        "- The model learned from a single harvest season (September to October 2023) and eight varieties and regions.\n"
        "- The quality band uses the dataset's own cut points (1.50, 2.50 and 3.50), not an industry standard.\n"
        "- The model shows associations, not causes: raising sugar content will not by itself make a banana better."
    )
