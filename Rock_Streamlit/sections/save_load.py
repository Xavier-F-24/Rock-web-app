"""Save/load page scaffold."""

from __future__ import annotations

import streamlit as st

from Rock_Streamlit.app_state import get_game_state, set_game_state
from rockgame_ui import game_controller


def render() -> None:
    game = get_game_state()

    st.title("Save / Load")
    st.caption("Download and upload split-module game saves.")

    save_json = game_controller.serialize_game(game)
    st.download_button(
        "Download Save",
        data=save_json,
        file_name=f"rock_game_gen{game.generation}.json",
        mime="application/json",
    )

    uploaded = st.file_uploader("Upload Save", type=["json"])
    if uploaded is not None:
        if st.button("Load Uploaded Save", type="primary"):
            try:
                loaded_game = game_controller.load_game_from_json(uploaded.getvalue().decode("utf-8"))
                set_game_state(loaded_game)
                st.success("Save loaded.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
