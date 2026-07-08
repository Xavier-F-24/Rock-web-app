"""Tree page for lineage visualization."""

from __future__ import annotations

import streamlit as st

from Rock_Streamlit.app_state import get_game_state
from Rock_Streamlit.ui_components import page_header, section
from rockgame_ui import game_controller


def render() -> None:
    game = get_game_state()

    page_header("Family Tree", "Pan, zoom, and inspect the full rock lineage.")

    with section("Tree Controls"):
        col_a, col_b, col_c = st.columns(3)
        canvas_width = col_a.slider("Width", 800, 1800, 1200, step=100)
        canvas_height = col_b.slider("Height", 500, 1200, 800, step=50)
        debug_connectors = col_c.checkbox("Debug connectors", value=False)

    if not game_controller.has_rocks(game):
        st.info("No rocks to draw yet.")
        return

    with section("Lineage"):
        fig = game_controller.render_tree_for_streamlit(
            game=game,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            debug_connectors=debug_connectors,
        )
        st.plotly_chart(fig, width="stretch", config={"scrollZoom": True})
