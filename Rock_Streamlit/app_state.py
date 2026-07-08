"""Session-state helpers for the Streamlit rock game UI."""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

from rockgame_ui.game_controller import GameStartSettings, start_new_game

if TYPE_CHECKING:
    from Rock_GameState.rock_game_state_helper import GameMaster


GAME_STATE_KEY = "rock_game"


def make_new_game(
    seed: int | None = None,
    settings: GameStartSettings | dict | None = None,
) -> GameMaster:
    return start_new_game(seed=seed, settings=settings)


def init_session_state() -> None:
    if GAME_STATE_KEY not in st.session_state:
        st.session_state[GAME_STATE_KEY] = None


def has_game_state() -> bool:
    init_session_state()
    return st.session_state[GAME_STATE_KEY] is not None


def get_game_state() -> GameMaster:
    init_session_state()
    game = st.session_state[GAME_STATE_KEY]
    if game is None:
        raise RuntimeError("No active game. Start or load a game first.")
    return game


def reset_game(
    seed: int | None = None,
    settings: GameStartSettings | dict | None = None,
) -> GameMaster:
    st.session_state[GAME_STATE_KEY] = make_new_game(seed=seed, settings=settings)
    return st.session_state[GAME_STATE_KEY]


def set_game_state(game: GameMaster) -> None:
    st.session_state[GAME_STATE_KEY] = game
