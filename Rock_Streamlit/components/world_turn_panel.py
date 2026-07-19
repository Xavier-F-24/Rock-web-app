"""Reusable explicit persistent-world turn controls."""

import streamlit as st

from rockgame_ui import game_controller


WORLD_ACTION_KEY = "world_last_action"


def render_world_turn_panel(game, *, key_prefix: str):
    summary = game_controller.get_world_summary(game)
    columns = st.columns([1, 1, 1, 1.2])
    columns[0].metric("World Turn", summary["turn"])
    columns[1].metric("NPC Farms", summary["npc_count"])
    columns[2].metric("Unread", summary["unread_messages"])
    if columns[3].button("End World Turn", type="primary", width="stretch", key=f"{key_prefix}_end_world_turn"):
        st.session_state[WORLD_ACTION_KEY] = game_controller.end_world_turn(game)
        st.rerun()
    result = st.session_state.pop(WORLD_ACTION_KEY, None)
    if result is not None:
        (st.success if result.ok else st.error)(result.message)
    return summary
