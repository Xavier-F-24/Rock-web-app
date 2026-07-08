"""Debug page scaffold."""

from __future__ import annotations

import streamlit as st

from Rock_Streamlit.app_state import get_game_state
from rockgame_ui import game_controller


def render() -> None:
    game = get_game_state()

    st.title("Debug")
    st.caption("Safe raw state summary for development.")

    st.json(game_controller.get_raw_state_summary(game))

    with st.expander("Recent Events", expanded=True):
        if game.events:
            for event in game.events[-25:]:
                st.write(f"- {event}")
        else:
            st.write("No events yet.")
