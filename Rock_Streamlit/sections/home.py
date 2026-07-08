"""Home page for the Streamlit rock game shell."""

from __future__ import annotations

import streamlit as st

from Rock_Streamlit.app_state import get_game_state, reset_game
from Rock_Streamlit.view_models import game_summary, rock_table_rows


def render() -> None:
    st.title("Rock Genetics Game")
    st.caption("Breed, trade, and trace a very serious lineage of very serious rocks.")

    if st.button("New Game", type="primary"):
        reset_game()
        st.rerun()

    game = get_game_state()
    summary = game_summary(game)

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Generation", f"{summary['generation']} / {game.max_generation}")
    col_b.metric("Money", f"${summary['money']}")
    col_c.metric("Rocks", summary["rock_count"])
    col_d.metric("Queued", summary["queued_pairs"])

    st.subheader("Current Rocks")
    st.dataframe(rock_table_rows(game), use_container_width=True, hide_index=True)

    st.subheader("Recent Events")
    if game.events:
        for event in game.events[-8:]:
            st.write(f"- {event}")
    else:
        st.write("No events yet.")
