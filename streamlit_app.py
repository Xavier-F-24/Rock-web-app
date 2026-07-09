"""
Production Streamlit shell for the split-module rock genetics game.

Run locally with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import streamlit as st

from Rock_Streamlit.app_state import get_game_state, has_game_state, init_session_state
from Rock_Streamlit.sections import breeding, debug, ending, home, market, save_load, start
from Rock_Streamlit.ui_components import apply_cozy_lab_style


PAGES = {
    "Home": ("home", home.render),
    "Market": ("market", market.render),
    "The Rock Farm": ("rock-farm", breeding.render),
    "Save / Load": ("save-load", save_load.render),
    "Debug": ("debug", debug.render),
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
            st.Page(page_fn, title=title, url_path=url_path)
            for title, (url_path, page_fn) in PAGES.items()
        ]
        selected_page = st.navigation(pages)
    except (AttributeError, TypeError):
        return False

    selected_page.run()
    return True


def render_with_sidebar_navigation() -> None:
    st.sidebar.title("Rock Game")
    page_title = st.sidebar.radio("Navigate", list(PAGES), label_visibility="collapsed")
    _, page_fn = PAGES[page_title]
    page_fn()


def main() -> None:
    st.set_page_config(page_title="Rock Genetics Game", page_icon="R", layout="wide")
    init_session_state()
    apply_cozy_lab_style()

    if not has_game_state():
        start.render()
        return

    if get_game_state().game_over:
        ending.render()
        return

    if render_with_page_navigation():
        return

    render_with_sidebar_navigation()


main()
