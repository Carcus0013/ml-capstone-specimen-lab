"""Registry of the app's pages, filled in by app.py so views can link to each other."""
PAGES = {}


def link(st, key, label, icon=None):
    if key in PAGES:
        st.page_link(PAGES[key], label=label, icon=icon)
