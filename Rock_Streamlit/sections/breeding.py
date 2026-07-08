"""Breeding page scaffold."""

from __future__ import annotations

import streamlit as st

from Rock_Streamlit.app_state import get_game_state
from rockgame_ui import game_controller


def render() -> None:
    game = get_game_state()
    breeders = game_controller.get_breedable_rocks(game)

    st.title("Breeding")
    st.caption("Placeholder shell for parent selection, relatedness preview, and queue actions.")

    if len(breeders) < 2:
        st.info("Not enough active rocks to breed.")
        return

    options = [rock.id for rock in breeders]
    labels = {rock.id: game_controller.rock_label(rock) for rock in breeders}

    col_a, col_b, col_c = st.columns(3)
    parent_a_id = col_a.selectbox("Parent A", options, format_func=lambda rock_id: labels[rock_id])
    parent_b_id = col_b.selectbox("Parent B", options, format_func=lambda rock_id: labels[rock_id])
    potion_key = col_c.selectbox(
        "Potion",
        game_controller.get_potion_options(game),
        format_func=lambda key: "None" if key is None else key,
    )

    validation = game_controller.validate_breeding_pair(game, parent_a_id, parent_b_id)

    if validation["warnings"]:
        for warning in validation["warnings"]:
            st.warning(warning)
    if validation["errors"]:
        for error in validation["errors"]:
            st.error(error)

    st.subheader("Action Area")
    st.write("Queueing and generation-advance controls will be wired here in the next pass.")
    queue_summary = game_controller.get_queue_summary(game)
    st.write(f"Current queue: {queue_summary['queued_pairs']} / {queue_summary['max_pairs']}")

    if st.button("Queue Selected Pair", disabled=not validation["valid"]):
        result = game_controller.breed_pair(
            game,
            parent_a_id,
            parent_b_id,
            options={"potion_key": potion_key},
        )
        if result.ok:
            st.success(result.message)
            st.rerun()
        else:
            st.error(result.message)
