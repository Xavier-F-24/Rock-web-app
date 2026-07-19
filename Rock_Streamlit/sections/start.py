"""Start screen for the Streamlit rock game."""

from __future__ import annotations

import streamlit as st

from Rock_Streamlit.app_state import reset_game, set_game_state
from Rock_Streamlit.ui_components import page_header, section
from rockgame_ui import game_controller


def _world_settings(prefix: str):
    mode_label = st.segmented_control(
        "World farmer size", ("Random (3-8)", "Choose a number"),
        default="Random (3-8)", key=f"{prefix}_world_size_mode",
    )
    mode = "random" if mode_label == "Random (3-8)" else "fixed"
    count = st.number_input(
        "NPC farmers", min_value=2, max_value=12, value=3, step=1,
        disabled=mode == "random", key=f"{prefix}_world_farmer_count",
    )
    return mode, int(count)


def _dev_settings_form() -> None:
    with st.expander("Dev Mode", expanded=False):
        with st.form("dev_start_settings"):
            seed_enabled = st.checkbox("Use fixed seed", value=False)
            seed = st.number_input("Seed", min_value=0, value=101, step=1)
            starting_money = st.number_input("Starting money", min_value=0, value=10, step=1)
            max_generation = st.number_input("Max generation", min_value=1, value=7, step=1)
            max_pairs = st.number_input("Max pairs per generation", min_value=1, value=3, step=1)
            farm_cost = st.number_input("Rock farm cost", min_value=0, value=75, step=1)
            world_mode, world_count = _world_settings("dev")
            neural_farmers = st.checkbox("Allow compatible neural farmers", value=True)

            if st.form_submit_button("Start Dev Game", type="primary"):
                settings = game_controller.GameStartSettings(
                    seed=int(seed) if seed_enabled else None,
                    starting_money=int(starting_money),
                    max_generation=int(max_generation),
                    max_pairs_per_generation=int(max_pairs),
                    rock_farm_cost=int(farm_cost),
                    world_size_mode=world_mode,
                    world_farmer_count=world_count,
                    allow_neural_farmers=neural_farmers,
                )
                reset_game(settings=settings)
                st.rerun()


def _load_save_control() -> None:
    uploaded = st.file_uploader("Load Save", type=["json"], key="start_load_save")
    if uploaded is None:
        return

    if st.button("Load Game", type="primary", key="start_load_game"):
        try:
            loaded_game = game_controller.load_game_from_json(uploaded.getvalue().decode("utf-8"))
            set_game_state(loaded_game)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))


def render() -> None:
    page_header(
        "Rock Genetics Game",
        "Breed a tiny lineage, manage the market, and see whether the farm pays for itself.",
    )

    col_start, col_load = st.columns([1.2, 1])

    with col_start:
        with section("Start Farm", "Begin with the standard rules and a fresh starter set."):
            st.write("Default game: $10 starting money, 7 generations, 3 breeding pairs per generation.")
            world_mode, world_count = _world_settings("standard")
            if st.button("Start Game", type="primary", key="start_standard_game", width="stretch"):
                reset_game(settings=game_controller.GameStartSettings(
                    world_size_mode=world_mode, world_farmer_count=world_count,
                ))
                st.rerun()
            _dev_settings_form()

    with col_load:
        with section("Continue", "Upload a previous JSON save."):
            _load_save_control()
