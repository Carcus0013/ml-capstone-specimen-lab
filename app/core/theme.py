"""Visual system: one colour per track, shared typography, and small HTML helpers."""
from html import escape
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

INK = "#18262F"
MUTED = "#52636B"
RULE = "#D3DCD8"
PAPER = "#F3F6F4"
SURFACE = "#FFFFFF"
DANGER = "#B3263E"
SAFE = "#2F7D5B"

TRACK = {
    "regression": {"color": "#B07D0A", "soft": "#F6ECD2", "name": "Regression", "subject": "Banana quality"},
    "classification": {"color": "#6D4C8A", "soft": "#ECE4F2", "name": "Classification", "subject": "Mushroom safety"},
    "clustering": {"color": "#2B6A8F", "soft": "#DFEBF2", "name": "Clustering", "subject": "Meteorite landings"},
}
# Okabe-Ito colour-blind-safe palette for categorical series (clusters, models)
SERIES = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]

_template = go.layout.Template()
_template.layout = go.Layout(
    font=dict(family="Atkinson Hyperlegible, Arial, sans-serif", color=INK, size=13),
    title=dict(font=dict(family="Bricolage Grotesque, Arial, sans-serif", size=16), x=0, xanchor="left"),
    paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, colorway=SERIES,
    xaxis=dict(gridcolor="#E7ECEA", zerolinecolor=RULE, linecolor=RULE),
    yaxis=dict(gridcolor="#E7ECEA", zerolinecolor=RULE, linecolor=RULE),
    margin=dict(l=10, r=10, t=50, b=10), legend=dict(bgcolor="rgba(255,255,255,0.8)"),
)
pio.templates["specimen"] = _template
pio.templates.default = "simple_white+specimen"


def inject_css():
    css = (Path(__file__).resolve().parents[1] / "assets" / "style.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def page_header(track, title, question):
    """Page title block: the question in the track colour's rule, with a short subtitle."""
    t = TRACK[track]
    st.markdown(
        f"""<div class="page-head" style="--track:{t['color']}">
              <p class="page-kicker">{escape(t['name'])} track</p>
              <h1>{escape(title)}</h1>
              <p class="page-question">{escape(question)}</p>
            </div>""",
        unsafe_allow_html=True,
    )


def plain_header(title, subtitle):
    st.markdown(f"""<div class="page-head plain"><h1>{escape(title)}</h1>
                    <p class="page-question">{escape(subtitle)}</p></div>""", unsafe_allow_html=True)


def result_panel(track, label, value, detail):
    """The one emphasised output on a prediction page."""
    t = TRACK[track]
    st.markdown(
        f"""<div class="result" style="--track:{t['color']};--soft:{t['soft']}">
              <p class="result-label">{escape(label)}</p>
              <p class="result-value">{escape(value)}</p>
              <p class="result-detail">{detail}</p>
            </div>""",
        unsafe_allow_html=True,
    )


def note(text, kind="info"):
    """Inline note. kind: info | warn | danger."""
    st.markdown(f'<div class="note note-{kind}">{text}</div>', unsafe_allow_html=True)


def style(fig, height=None):
    fig.update_layout(template="simple_white+specimen")
    if height:
        fig.update_layout(height=height)
    return fig
