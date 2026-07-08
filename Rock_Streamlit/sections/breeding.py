"""Breeding page scaffold."""

from __future__ import annotations

import streamlit as st

from Rock_Streamlit.app_state import get_game_state
from Rock_Streamlit.view_models import active_rocks, rock_label


def render() -> None:
    game = get_game_state()
    breeders = active_rocks(game)

    st.title("Breeding")
    st.caption("Placeholder shell for parent selection, relatedness preview, and queue actions.")

    if len(breeders) < 2:
        st.info("Not enough active rocks to breed.")
        return

    options = [rock.id for rock in breeders]
    labels = {rock.id: rock_label(rock) for rock in breeders}

    col_a, col_b, col_c = st.columns(3)
    parent_a_id = col_a.selectbox("Parent A", options, format_func=lambda rock_id: labels[rock_id])
    parent_b_id = col_b.selectbox("Parent B", options, format_func=lambda rock_id: labels[rock_id])
    potion_key = col_c.selectbox(
        "Potion",
        [None] + sorted(game.potions),
        format_func=lambda key: "None" if key is None else f"{key} ({game.potions[key]})",
    )

    parent_a = game.get_rock(parent_a_id)
    parent_b = game.get_rock(parent_b_id)
    validation = game.breeding_master.validate_breeding_pair(parent_a, parent_b, game=game)

    if validation["warnings"]:
        for warning in validation["warnings"]:
            st.warning(warning)
    if validation["errors"]:
        for error in validation["errors"]:
            st.error(error)

    st.subheader("Action Area")
    st.write("Queueing and generation-advance controls will be wired here in the next pass.")
    st.write(f"Current queue: {len(game.breeding_queue)} / {game.max_pairs_per_generation}")

    if st.button("Queue Selected Pair", disabled=not validation["valid"]):
        try:
            game.add_pair_to_queue(parent_a_id, parent_b_id, potion_key=potion_key)
            st.success("Pair queued.")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
