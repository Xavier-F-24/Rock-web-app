"""Home page for the Streamlit rock game shell."""

from __future__ import annotations

import streamlit as st

from Rock_Streamlit.app_state import get_game_state, reset_game
from rockgame_ui import game_controller


def render() -> None:
    st.title("Rock Genetics Game")
    st.caption("Breed, trade, and trace a very serious lineage of very serious rocks.")

    if st.button("New Game", type="primary"):
        reset_game()
        st.rerun()

    game = get_game_state()
    summary = game_controller.get_game_summary(game)

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Generation", f"{summary['generation']} / {summary['max_generation']}")
    col_b.metric("Money", f"${summary['money']}")
    col_c.metric("Rocks", summary["rock_count"])
    col_d.metric("Queued", summary["queued_pairs"])

    st.subheader("Current Rocks")
    st.dataframe(game_controller.get_rock_rows(game), width="stretch", hide_index=True)

    st.subheader("Recent Events")
    events = game_controller.get_recent_events(game, limit=8)
    if events:
        for event in events:
            st.write(f"- {event}")
    else:
        st.write("No events yet.")
