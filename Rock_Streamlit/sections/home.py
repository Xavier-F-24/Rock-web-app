"""Home page for the Streamlit rock game shell."""

from __future__ import annotations

import streamlit as st

from Rock_Streamlit.app_state import clear_game_state, get_game_state
from Rock_Streamlit.ui_components import metric_strip, page_header, section
from rockgame_ui import game_controller


def render() -> None:
    page_header("Rock Genetics Game", "Breed, trade, and trace a very serious lineage of very serious rocks.")

    if st.button("New Game", type="primary"):
        clear_game_state()
        st.rerun()

    game = get_game_state()
    summary = game_controller.get_game_summary(game)
    score = game_controller.get_final_score_summary(game)

    metric_strip(
        [
            ("Generation", f"{summary['generation']} / {summary['max_generation']}"),
            ("Money", f"${summary['money']}"),
            ("Active Value", f"${score['active_rock_score']}"),
            ("Projected Score", f"${score['final_score']}"),
        ]
    )

    with section("Current Rocks"):
        st.dataframe(game_controller.get_rock_rows(game), width="stretch", hide_index=True)

    with section("Recent Events"):
        events = game_controller.get_recent_events(game, limit=8)
        if events:
            for event in events:
                st.write(f"- {event}")
        else:
            st.write("No events yet.")
