"""Classification: is a mushroom edible or poisonous? Uses the selected saved pipeline."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression

from core import data, logic, theme

GROUPS = {
    "Cap": ["cap-shape", "cap-surface", "cap-color", "bruises", "odor"],
    "Gills": ["gill-attachment", "gill-spacing", "gill-size", "gill-color"],
    "Stalk": ["stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring",
              "stalk-color-above-ring", "stalk-color-below-ring"],
    "Veil, ring and spores": ["veil-color", "ring-number", "ring-type", "spore-print-color"],
    "Where it grows": ["population", "habitat"],
}


def _fill(values):
    for col, value in values.items():
        if col != "actual_class":
            st.session_state[f"clf_{col}"] = value
    st.session_state["clf_example"] = values


def render():
    theme.page_header("classification", "Mushroom safety",
                      "Describe a mushroom's features. The selected model estimates whether it is edible or poisonous.")
    theme.note("<b>Never use this to decide whether a real mushroom is safe to eat.</b> The training data are 8,124 "
               "hypothetical records based on 23 species in a 1981 field guide. A model that is perfect on this data "
               "can still be badly wrong on mushrooms it has never seen.", "danger")

    sel, model = data.selected_model("classification")
    schema = data.schema("classification")
    labels = schema["labels"]
    if "clf_odor" not in st.session_state:
        _fill(schema["examples"][0])

    st.markdown("Start from a mushroom in the held-out test set, or change any feature.")
    c1, c2, _ = st.columns([1, 1, 2])
    c1.button("Test edible mushroom", on_click=_fill, args=(schema["examples"][0],), width="stretch")
    c2.button("Test poisonous mushroom", on_click=_fill, args=(schema["examples"][1],), width="stretch")

    with st.form("classification_form"):
        for group, features in GROUPS.items():
            st.markdown(f"### {group}")
            cols = st.columns(3)
            for i, col in enumerate(features):
                cols[i % 3].selectbox(col.replace("-", " ").capitalize(), schema["categorical"][col], key=f"clf_{col}",
                                      format_func=lambda code, c=col: f"{labels[c].get(code, code)} ({code})")
        submitted = st.form_submit_button("Classify mushroom", type="primary", width="stretch")

    if submitted:
        values = {c: st.session_state.get(f"clf_{c}") for c in schema["categorical"]}
        row, errors = logic.validate_mushroom_input(values, schema)
        if errors:
            for e in errors:
                st.error(e)
            return
        try:
            proba = model.predict_proba(row)[0]
            pred = int(model.predict(row)[0])
        except Exception:
            st.error("The model could not classify this mushroom. Check the selections and try again.")
            return
        st.session_state["clf_last"] = {"row": row, "proba": proba.tolist(), "pred": pred}

    last = st.session_state.get("clf_last")
    if not last:
        return
    classes = schema["classes"]
    name = classes[str(last["pred"])]
    p_poison = last["proba"][list(model.classes_).index(schema["positive_class"])]
    left, right = st.columns([1, 1.1], gap="large")
    with left:
        theme.result_panel("classification", "Predicted class", name.capitalize(),
                           f"Model's estimated probability of poisonous: <b>{p_poison:.1%}</b>. "
                           f"Model: {sel['selected_model']}.")
        example = st.session_state.get("clf_example") or {}
        if example.get("actual_class") and all(last["row"].iloc[0][c] == example[c] for c in last["row"].columns):
            st.markdown(f"This test mushroom is actually **{example['actual_class']}**.")
        fig = go.Figure(go.Bar(x=[1 - p_poison, p_poison], y=["edible", "poisonous"], orientation="h",
                               marker_color=[theme.SAFE, theme.DANGER], text=[f"{1 - p_poison:.1%}", f"{p_poison:.1%}"],
                               textposition="auto"))
        fig.update_layout(title="Estimated class probabilities", xaxis=dict(range=[0, 1], title="probability", tickformat=".0%"),
                          yaxis_title="class")
        st.plotly_chart(theme.style(fig, 230), width="stretch")
        st.caption("A probability is the model's own confidence, learned from this dataset. It is not a guarantee "
                   "that the prediction is correct, and on near-separable data models are often over-confident.")
    with right:
        if isinstance(model.named_steps["model"], LogisticRegression):
            contrib, intercept = logic.logistic_contributions(model, last["row"], labels)
            shown = pd.concat([contrib.head(5), contrib.tail(5)]).drop_duplicates()
            fig = go.Figure(go.Bar(x=shown.values, y=shown.index, orientation="h",
                                   marker_color=[theme.DANGER if v > 0 else theme.SAFE for v in shown.values]))
            fig.update_layout(title="What pushed this prediction (log-odds contributions)",
                              xaxis_title="towards edible  |  towards poisonous", yaxis_title="")
            st.plotly_chart(theme.style(fig, 380), width="stretch")
            st.caption(f"For logistic regression, log-odds of poisonous = {intercept:.2f} (intercept) plus the sum of "
                       "these contributions. Features not shown contribute less.")

    with st.expander("What every trained model says about this mushroom"):
        rows = []
        for mname, meta, m in data.all_models("classification"):
            p = m.predict_proba(last["row"])[0][list(m.classes_).index(schema["positive_class"])]
            rows.append({"Model": mname, "Prediction": classes[str(int(m.predict(last["row"])[0]))],
                         "P(poisonous)": p, "Held-out F1": meta["test_metrics"]["F1 (weighted)"],
                         "Selected": "yes" if meta["selected"] else ""})
        table = pd.DataFrame(rows).sort_values("Held-out F1", ascending=False)
        st.dataframe(table, hide_index=True, width="stretch",
                     column_config={"P(poisonous)": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1),
                                    "Held-out F1": st.column_config.NumberColumn(format="%.4f")})
        n_disagree = int((table["Prediction"] != name).sum())
        st.caption(f"{n_disagree} of {len(table)} models disagree with the selected model on this mushroom.")

    st.markdown("### Limitations")
    st.markdown(
        "- Seven of the ten models are perfect on the 1,625 held-out mushrooms, but that set comes from the same 23 "
        "species as the training data.\n"
        "- Features such as odour are human judgements, and their reliability in the field is unknown.\n"
        "- Zero errors on 1,625 test cases is still consistent with an error rate of up to about 0.2%."
    )
