"""
Production Streamlit shell for the split-module rock genetics game.

Run locally with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import streamlit as st

from Rock_Streamlit.app_state import init_session_state
from Rock_Streamlit.sections import breeding, debug, home, market, save_load, tree


PAGES = {
    "Home": home.render,
    "Market": market.render,
    "Breeding": breeding.render,
    "Tree": tree.render,
    "Save / Load": save_load.render,
    "Debug": debug.render,
}


def render_with_page_navigation() -> bool:
    """
    Prefer modern Streamlit navigation when available.

    Returns True when the app was rendered through st.navigation.
    """
    if not hasattr(st, "Page") or not hasattr(st, "navigation"):
        return False

    try:
        pages = [
            st.Page(page_fn, title=title)
            for title, page_fn in PAGES.items()
        ]
        selected_page = st.navigation(pages)
    except (AttributeError, TypeError):
        return False

    selected_page.run()
    return True


def render_with_sidebar_navigation() -> None:
    st.sidebar.title("Rock Game")
    page_title = st.sidebar.radio("Navigate", list(PAGES), label_visibility="collapsed")
    PAGES[page_title]()


def main() -> None:
    st.set_page_config(
        page_title="Rock Genetics Game",
        page_icon="🪨",
        layout="wide",
    )
    init_session_state()

    if render_with_page_navigation():
        return

    render_with_sidebar_navigation()


main()
