"""Streamlit entry point for the 23CSE301 ML capstone.

Run from the repository root:   streamlit run app/app.py

The app only *loads* the pipelines and result files produced by the notebooks; it never
retrains a model. Paths are resolved relative to this file, so it runs unchanged on any
machine or hosting service.
"""
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
for path in (str(APP_DIR), str(ROOT)):          # ROOT ends up first: `src` and `app/core` both importable
    if path not in sys.path:
        sys.path.insert(0, path)

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Specimen Lab: ML capstone", page_icon=":material/science:", layout="wide")

from core import data, nav, theme  # noqa: E402
from views import about, classification, clustering, comparison, explorer, home, regression  # noqa: E402

theme.inject_css()

pages = {
    "home": st.Page(home.render, title="Overview", icon=":material/home:", url_path="overview", default=True),
    "regression": st.Page(regression.render, title="Banana quality", icon=":material/nutrition:", url_path="regression"),
    "classification": st.Page(classification.render, title="Mushroom safety", icon=":material/eco:", url_path="classification"),
    "clustering": st.Page(clustering.render, title="Meteorite clusters", icon=":material/public:", url_path="clustering"),
    "comparison": st.Page(comparison.render, title="Model comparison", icon=":material/leaderboard:", url_path="comparison"),
    "explorer": st.Page(explorer.render, title="Data explorer", icon=":material/table_view:", url_path="data"),
    "about": st.Page(about.render, title="About", icon=":material/info:", url_path="about"),
}
nav.PAGES.update(pages)
current = st.navigation(list(pages.values()), position="top")

# Startup check: fail clearly (not with a stack trace) if any model or result file is absent.
missing = data.missing_files()
if missing:
    st.error("Some trained models or result files are missing, so the app cannot show real predictions. "
             "Run the three notebooks to regenerate them. Missing: " + ", ".join(missing))
    st.stop()

current.run()
