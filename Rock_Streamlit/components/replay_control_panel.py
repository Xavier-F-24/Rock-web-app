"""Snapshot-backed replay navigation controls."""

from __future__ import annotations

import streamlit as st


def render_replay_controls(replay, seek_callback) -> None:
    st.warning("Replay mode: controls below navigate recorded snapshots and do not mutate the live session.")
    columns = st.columns(4)
    if columns[0].button("First", key="ai_obs_replay_first"):
        seek_callback("first")
    if columns[1].button("Previous", key="ai_obs_replay_previous", disabled=replay.position <= 0):
        seek_callback(max(0, replay.position - 1))
    if columns[2].button("Next", key="ai_obs_replay_next", disabled=replay.position >= len(replay.frames) - 1):
        seek_callback(min(len(replay.frames) - 1, replay.position + 1))
    if columns[3].button("Last", key="ai_obs_replay_last"):
        seek_callback("last")
    position = st.slider(
        "Replay cursor",
        0,
        max(0, len(replay.frames) - 1),
        replay.position,
        key="ai_obs_replay_cursor",
    )
    if position != replay.position:
        seek_callback(position)
    st.caption(f"Frame {replay.position} of {len(replay.frames) - 1}")
    if not replay.validation.valid:
        st.error(f"Replay divergence detected in {len(replay.validation.divergences)} checks.")
