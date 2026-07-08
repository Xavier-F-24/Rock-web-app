"""Breeding page scaffold."""

from __future__ import annotations

import streamlit as st

from Rock_Streamlit.app_state import get_game_state
from rockgame_ui import game_controller


BREEDING_MESSAGE_KEY = "breeding_last_action"


def _show_last_action_message() -> None:
    result = st.session_state.pop(BREEDING_MESSAGE_KEY, None)
    if result is None:
        return

    if result.ok:
        st.success(result.message)
        if result.payload:
            st.subheader("Children Created")
            st.dataframe(result.payload, use_container_width=True, hide_index=True)
    else:
        st.error(result.message)


def _store_action_and_rerun(result: game_controller.ActionResult) -> None:
    st.session_state[BREEDING_MESSAGE_KEY] = result
    st.rerun()


def _parent_b_options(parent_a_id: int, candidates):
    parent_a = next((rock for rock in candidates if rock.id == parent_a_id), None)
    other_candidates = [rock for rock in candidates if rock.id != parent_a_id]
    if parent_a is None:
        return other_candidates

    opposite_sex = [rock for rock in other_candidates if rock.sex != parent_a.sex]
    return opposite_sex or other_candidates


def render() -> None:
    game = get_game_state()
    summary = game_controller.get_game_summary(game)
    queue_summary = game_controller.get_queue_summary(game)
    candidates = game_controller.get_available_breeding_candidates(game)

    st.title("Breeding")
    st.caption("Select parents, queue breeding pairs, then advance the generation.")
    _show_last_action_message()

    col_gen, col_slots = st.columns(2)
    col_gen.metric("Generation", f"{summary['generation']} / {summary['max_generation']}")
    col_slots.metric(
        "Breeding slots",
        f"{queue_summary['queued_pairs']} / {queue_summary['max_pairs']}",
    )

    queue_rows = game_controller.get_breeding_queue_rows(game)
    st.subheader("Queued Pairs")
    if queue_rows:
        st.dataframe(queue_rows, use_container_width=True, hide_index=True)
    else:
        st.info("No pairs queued yet.")

    if len(candidates) < 2:
        st.info("Not enough unqueued active rocks to add another pair.")
        if queue_rows and st.button("Breed queued pairs", key="breeding_advance_generation"):
            result = game_controller.advance_breeding_generation(game)
            _store_action_and_rerun(result)
        return

    options = [rock.id for rock in candidates]
    labels = {rock.id: game_controller.rock_label(rock) for rock in candidates}

    st.subheader("Select Parents")
    col_a, col_b, col_c = st.columns(3)
    parent_a_id = col_a.selectbox(
        "Parent A",
        options,
        format_func=lambda rock_id: labels[rock_id],
        key="breeding_parent_a_select",
    )
    parent_b_candidates = _parent_b_options(parent_a_id, candidates)
    parent_b_options = [rock.id for rock in parent_b_candidates]
    parent_b_id = col_b.selectbox(
        "Parent B",
        parent_b_options,
        format_func=lambda rock_id: labels[rock_id],
        key="breeding_parent_b_select",
    )
    potion_key = col_c.selectbox(
        "Potion",
        game_controller.get_potion_options(game),
        format_func=lambda key: "None" if key is None else key,
        key="breeding_potion_select",
    )

    validation = game_controller.validate_breeding_pair(game, parent_a_id, parent_b_id)

    if validation["warnings"]:
        for warning in validation["warnings"]:
            st.warning(warning)
    if validation["errors"]:
        for error in validation["errors"]:
            st.error(error)

    st.subheader("Action Area")
    queue_full = queue_summary["queued_pairs"] >= queue_summary["max_pairs"]
    col_queue, col_breed = st.columns(2)

    if col_queue.button(
        "Queue selected pair",
        key="breeding_queue_selected_pair",
        disabled=queue_full or not validation["valid"],
    ):
        result = game_controller.breed_pair(
            game,
            parent_a_id,
            parent_b_id,
            options={"potion_key": potion_key},
        )
        _store_action_and_rerun(result)

    if col_breed.button(
        "Breed queued pairs",
        key="breeding_advance_generation",
        disabled=queue_summary["queued_pairs"] == 0,
        type="primary",
    ):
        result = game_controller.advance_breeding_generation(game)
        _store_action_and_rerun(result)
