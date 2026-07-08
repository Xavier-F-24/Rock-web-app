"""Session-state helpers for the Streamlit rock game UI."""

from __future__ import annotations

import streamlit as st

from Rock_GameState.rock_game_state_helper import GameMaster


GAME_STATE_KEY = "rock_game"


def make_new_game(seed: int | None = None) -> GameMaster:
    return GameMaster(seed=seed)


def init_session_state() -> None:
    if GAME_STATE_KEY not in st.session_state:
        st.session_state[GAME_STATE_KEY] = make_new_game()


def get_game_state() -> GameMaster:
    init_session_state()
    return st.session_state[GAME_STATE_KEY]


def reset_game(seed: int | None = None) -> GameMaster:
    st.session_state[GAME_STATE_KEY] = make_new_game(seed=seed)
    return st.session_state[GAME_STATE_KEY]


def set_game_state(game: GameMaster) -> None:
    st.session_state[GAME_STATE_KEY] = game
