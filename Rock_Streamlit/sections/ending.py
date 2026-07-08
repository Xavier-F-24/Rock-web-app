"""Ending/results screen for completed games."""

from __future__ import annotations

import streamlit as st

from Rock_Streamlit.app_state import get_game_state, reset_game
from Rock_Streamlit.ui_components import metric_strip, page_header, section
from rockgame_ui import game_controller


def render() -> None:
    game = get_game_state()
    score = game_controller.get_final_score_summary(game)

    page_header(
        "Final Farm Ledger",
        "The last generation is complete. Time to see what the rocks made of themselves.",
    )

    metric_strip(
        [
            ("Active Rock Value", f"${score['active_rock_score']}"),
            ("Money Left", f"${score['money']}"),
            ("Farm Cost", f"-${score['rock_farm_cost']}"),
            ("Final Score", f"${score['final_score']}"),
        ]
    )

    with section("Final Save"):
        save_json = game_controller.serialize_game(game)
        st.download_button(
            "Download Final Save",
            data=save_json,
            file_name=f"rock_game_final_gen{game.generation}.json",
            mime="application/json",
            type="primary",
        )
        if st.button("Start New Farm", key="ending_new_game"):
            reset_game()
            st.rerun()

    with section("Final Rocks"):
        st.dataframe(game_controller.get_rock_rows(game), width="stretch", hide_index=True)

    with section("Final Tree"):
        fig = game_controller.render_tree_for_streamlit(
            game=game,
            canvas_width=1300,
            canvas_height=800,
        )
        st.plotly_chart(fig, width="stretch", config={"scrollZoom": True})
