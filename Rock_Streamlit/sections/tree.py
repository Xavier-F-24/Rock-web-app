"""Tree page scaffold."""

from __future__ import annotations

import streamlit as st

from Rock_Drawing.rock_lineage_drawing_helper import TreeDrawer
from Rock_Streamlit.app_state import get_game_state


def render() -> None:
    game = get_game_state()

    st.title("Family Tree")
    st.caption("Placeholder shell for lineage visualization and tree controls.")

    col_a, col_b, col_c = st.columns(3)
    canvas_width = col_a.slider("Width", 800, 1800, 1200, step=100)
    canvas_height = col_b.slider("Height", 500, 1200, 800, step=50)
    debug_connectors = col_c.checkbox("Debug connectors", value=False)

    if not game.rocks:
        st.info("No rocks to draw yet.")
        return

    fig = TreeDrawer(
        game=game,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        debug_connectors=debug_connectors,
    ).draw()
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})
