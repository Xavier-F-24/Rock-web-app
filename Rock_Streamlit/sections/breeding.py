"""Breeding page for parent selection and generation advancement."""

from __future__ import annotations

import streamlit as st

from Rock_Streamlit.app_state import get_game_state
from Rock_Streamlit.ui_components import metric_strip, page_header, rock_card, section
from rockgame_ui import game_controller


BREEDING_MESSAGE_KEY = "breeding_last_action"
PARENT_A_KEY = "breeding_parent_a_id"
PARENT_B_KEY = "breeding_parent_b_id"


def _show_last_action_message() -> None:
    result = st.session_state.pop(BREEDING_MESSAGE_KEY, None)
    if result is None:
        return

    if result.ok:
        st.success(result.message)
        if result.payload:
            st.subheader("Children Created")
            st.dataframe(result.payload, width="stretch", hide_index=True)
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


def _sync_selected_parent_ids(candidate_ids: set[int]) -> None:
    for key in (PARENT_A_KEY, PARENT_B_KEY):
        if st.session_state.get(key) not in candidate_ids:
            st.session_state[key] = None


def _render_parent_cards(title: str, cards: list[dict], selected_id: int | None, state_key: str) -> None:
    with section(title):
        if not cards:
            st.info("No available rocks for this slot.")
            return

        cols = st.columns(min(4, len(cards)))
        for index, card in enumerate(cards):
            with cols[index % len(cols)]:
                if rock_card(
                    card,
                    key_prefix=f"{state_key}_card",
                    selected=card["id"] == selected_id,
                ):
                    st.session_state[state_key] = card["id"]
                    st.rerun()


def render() -> None:
    game = get_game_state()
    summary = game_controller.get_game_summary(game)
    queue_summary = game_controller.get_queue_summary(game)
    candidates = game_controller.get_available_breeding_candidates(game)
    candidate_cards = game_controller.get_breeding_candidate_cards(game)
    card_by_id = {card["id"]: card for card in candidate_cards}
    _sync_selected_parent_ids(set(card_by_id))

    page_header("Breeding", "Select parents, queue breeding pairs, then advance the generation.")
    _show_last_action_message()

    metric_strip(
        [
            ("Generation", f"{summary['generation']} / {summary['max_generation']}"),
            ("Breeding Slots", f"{queue_summary['queued_pairs']} / {queue_summary['max_pairs']}"),
        ]
    )

    queue_rows = game_controller.get_breeding_queue_rows(game)
    with section("Queued Pairs"):
        if queue_rows:
            st.dataframe(queue_rows, width="stretch", hide_index=True)
        else:
            st.info("No pairs queued yet.")

    if len(candidates) < 2:
        st.info("Not enough unqueued active rocks to add another pair.")
        if queue_rows and st.button("Breed queued pairs", key="breeding_advance_generation"):
            result = game_controller.advance_breeding_generation(game)
            _store_action_and_rerun(result)
        return

    parent_a_id = st.session_state.get(PARENT_A_KEY)
    parent_b_candidate_rocks = _parent_b_options(parent_a_id, candidates) if parent_a_id else candidates
    parent_b_candidate_ids = {rock.id for rock in parent_b_candidate_rocks}
    if st.session_state.get(PARENT_B_KEY) not in parent_b_candidate_ids:
        st.session_state[PARENT_B_KEY] = None
    parent_b_id = st.session_state.get(PARENT_B_KEY)

    col_a, col_b = st.columns(2)
    with col_a:
        _render_parent_cards("Parent A", candidate_cards, parent_a_id, PARENT_A_KEY)
    with col_b:
        parent_b_cards = [card_by_id[rock.id] for rock in parent_b_candidate_rocks if rock.id in card_by_id]
        _render_parent_cards("Parent B", parent_b_cards, parent_b_id, PARENT_B_KEY)

    labels = {rock.id: game_controller.rock_label(rock) for rock in candidates}
    options = [rock.id for rock in candidates]
    blank_parent_options = [None, *options]
    with st.expander("Fallback selectors"):
        fallback_a = st.selectbox(
            "Parent A",
            blank_parent_options,
            index=blank_parent_options.index(parent_a_id) if parent_a_id in blank_parent_options else 0,
            format_func=lambda rock_id: "Choose parent A" if rock_id is None else labels[rock_id],
            key="breeding_parent_a_select_fallback",
        )
        if fallback_a != parent_a_id:
            st.session_state[PARENT_A_KEY] = fallback_a
            st.rerun()

        fallback_b_options = [
            rock.id for rock in (_parent_b_options(st.session_state.get(PARENT_A_KEY), candidates) or candidates)
        ]
        blank_parent_b_options = [None, *fallback_b_options]
        fallback_b = st.selectbox(
            "Parent B",
            blank_parent_b_options,
            index=blank_parent_b_options.index(parent_b_id) if parent_b_id in blank_parent_b_options else 0,
            format_func=lambda rock_id: "Choose parent B" if rock_id is None else labels[rock_id],
            key="breeding_parent_b_select_fallback",
        )
        if fallback_b != parent_b_id:
            st.session_state[PARENT_B_KEY] = fallback_b
            st.rerun()

    parent_a_id = st.session_state.get(PARENT_A_KEY)
    parent_b_id = st.session_state.get(PARENT_B_KEY)
    validation = {"valid": False, "errors": ["Choose two parents."], "warnings": []}
    if parent_a_id is not None and parent_b_id is not None:
        validation = game_controller.validate_breeding_pair(game, parent_a_id, parent_b_id)

    with section("Breeding Action"):
        potion_key = st.selectbox(
            "Potion",
            game_controller.get_potion_options(game),
            format_func=lambda key: "None" if key is None else key,
            key="breeding_potion_select",
        )

        if validation["warnings"]:
            for warning in validation["warnings"]:
                st.warning(warning)
        if validation["errors"]:
            for error in validation["errors"]:
                st.error(error)

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
            st.session_state[PARENT_A_KEY] = None
            st.session_state[PARENT_B_KEY] = None
            _store_action_and_rerun(result)

        if col_breed.button(
            "Breed queued pairs",
            key="breeding_advance_generation",
            disabled=queue_summary["queued_pairs"] == 0,
            type="primary",
        ):
            result = game_controller.advance_breeding_generation(game)
            _store_action_and_rerun(result)

    with section("Tree Reference"):
        fig = game_controller.render_tree_for_streamlit(game, canvas_width=1100, canvas_height=520)
        st.plotly_chart(fig, width="stretch", config={"scrollZoom": True})
