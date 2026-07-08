"""Breeding page for parent selection and generation advancement."""

from __future__ import annotations

import streamlit as st

from Rock_Streamlit.app_state import get_game_state
from Rock_Streamlit.ui_components import metric_strip, page_header, section
from rockgame_ui import game_controller


BREEDING_MESSAGE_KEY = "breeding_last_action"
PARENT_A_KEY = "breeding_parent_a_id"
PARENT_B_KEY = "breeding_parent_b_id"
CHECKBOX_PREFIX = "breeding_candidate_checkbox_"
CLEAR_PARENT_SELECTION_KEY = "breeding_clear_parent_selection_next_run"
MANUAL_PARENT_SELECTION_KEY = "breeding_manual_parent_selection_next_run"
BREED_MODE_KEY = "breeding_tree_breed_mode"
POTION_MULTISELECT_KEY = "breeding_potion_multiselect"
CLEAR_POTION_SELECTION_KEY = "breeding_clear_potion_selection_next_run"


def _show_last_action_message() -> None:
    result = st.session_state.pop(BREEDING_MESSAGE_KEY, None)
    if result is None:
        return

    if result.ok:
        st.success(result.message)
        if isinstance(result.payload, list) and result.payload:
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


def candidate_checkbox_key(rock_id: int) -> str:
    return f"{CHECKBOX_PREFIX}{int(rock_id)}"


def resolve_checkbox_parent_ids(checked_ids: list[int]) -> tuple[int | None, int | None]:
    parent_a_id = checked_ids[0] if len(checked_ids) >= 1 else None
    parent_b_id = checked_ids[1] if len(checked_ids) >= 2 else None
    return parent_a_id, parent_b_id


def _checked_candidate_ids(candidate_ids: set[int]) -> list[int]:
    return [
        rock_id
        for rock_id in sorted(candidate_ids)
        if st.session_state.get(candidate_checkbox_key(rock_id), False)
    ]


def _clear_parent_selection(candidate_ids: set[int]) -> None:
    st.session_state[PARENT_A_KEY] = None
    st.session_state[PARENT_B_KEY] = None
    for rock_id in candidate_ids:
        st.session_state[candidate_checkbox_key(rock_id)] = False


def _sync_selected_parent_ids(candidate_ids: set[int]) -> None:
    for key in (PARENT_A_KEY, PARENT_B_KEY):
        if st.session_state.get(key) not in candidate_ids:
            st.session_state[key] = None


def _selected_parent_label(parent_id: int | None, labels: dict[int, str]) -> str:
    if parent_id is None:
        return "None selected"
    return labels.get(parent_id, f"Rock #{parent_id}")


def render() -> None:
    game = get_game_state()
    summary = game_controller.get_game_summary(game)
    queue_summary = game_controller.get_queue_summary(game)
    candidates = game_controller.get_available_breeding_candidates(game)
    candidate_ids = {rock.id for rock in candidates}
    labels = {rock.id: game_controller.rock_label(rock) for rock in candidates}
    if st.session_state.pop(CLEAR_PARENT_SELECTION_KEY, False):
        _clear_parent_selection(candidate_ids)
    if st.session_state.pop(CLEAR_POTION_SELECTION_KEY, False):
        st.session_state[POTION_MULTISELECT_KEY] = []
    manual_selection = st.session_state.pop(MANUAL_PARENT_SELECTION_KEY, None)
    if manual_selection is not None:
        manual_a, manual_b = manual_selection
        _clear_parent_selection(candidate_ids)
        if manual_a in candidate_ids:
            st.session_state[candidate_checkbox_key(manual_a)] = True
        if manual_b in candidate_ids:
            st.session_state[candidate_checkbox_key(manual_b)] = True
    _sync_selected_parent_ids(candidate_ids)

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
            remove_cols = st.columns(len(queue_rows))
            for column, row in zip(remove_cols, queue_rows):
                with column:
                    if st.button(
                        f"Remove slot {row['slot']}",
                        key=f"breeding_remove_queue_slot_{row['slot']}",
                    ):
                        result = game_controller.remove_queued_pair(game, row["slot"])
                        _store_action_and_rerun(result)
        else:
            st.info("No pairs queued yet.")

    if len(candidates) < 2:
        st.info("Not enough unqueued active rocks to add another pair.")
        if queue_rows and st.button("Breed queued pairs", key="breeding_advance_generation"):
            result = game_controller.advance_breeding_generation(game)
            _store_action_and_rerun(result)
        return

    with section("Tree Reference", "Turn on Breed? to show parent boxes under available rocks."):
        breed_mode = st.checkbox(
            "Breed?",
            key=BREED_MODE_KEY,
            help="Show parent-picking boxes under available rocks in the lineage tree.",
        )
        if not breed_mode:
            _clear_parent_selection(candidate_ids)

        checked_before = _checked_candidate_ids(candidate_ids)
        parent_a_id, parent_b_id = resolve_checkbox_parent_ids(checked_before)
        st.session_state[PARENT_A_KEY] = parent_a_id
        st.session_state[PARENT_B_KEY] = parent_b_id

        highlighted_ids = tuple(
            rock_id for rock_id in (parent_a_id, parent_b_id) if rock_id is not None
        )
        fig = game_controller.render_tree_for_streamlit(
            game,
            canvas_width=1100,
            canvas_height=560,
            highlighted_rock_ids=highlighted_ids,
            rock_badges=game_controller.get_queued_parent_badges(game),
            tree_checkbox_ids=tuple(sorted(candidate_ids)) if breed_mode else (),
            tree_checked_ids=tuple(checked_before) if breed_mode else (),
        )
        st.plotly_chart(
            fig,
            width="stretch",
            config={"scrollZoom": True},
            key="breeding_tree_parent_selector",
        )

    if breed_mode:
        with section("Tree Box Controls", "These controls update the boxes shown on the tree. Pick exactly two."):
            if st.button("Clear parent checkboxes", key="breeding_clear_parent_checkboxes"):
                _clear_parent_selection(candidate_ids)
                st.rerun()

            checked_before = _checked_candidate_ids(candidate_ids)
            checked_limit_reached = len(checked_before) >= 2
            columns_per_row = 6
            for row_start in range(0, len(candidates), columns_per_row):
                row_rocks = candidates[row_start : row_start + columns_per_row]
                columns = st.columns(columns_per_row)
                for column, rock in zip(columns, row_rocks):
                    with column:
                        st.checkbox(
                            f"#{rock.id}",
                            key=candidate_checkbox_key(rock.id),
                            disabled=checked_limit_reached and rock.id not in checked_before,
                            help=labels[rock.id],
                        )

            checked_ids = _checked_candidate_ids(candidate_ids)
            if len(checked_ids) > 2:
                st.warning("Only the first two checked rocks will be used for this pair.")
            parent_a_id, parent_b_id = resolve_checkbox_parent_ids(checked_ids)
            st.session_state[PARENT_A_KEY] = parent_a_id
            st.session_state[PARENT_B_KEY] = parent_b_id

            selected_a, selected_b = st.columns(2)
            selected_a.info(f"Parent A: {_selected_parent_label(parent_a_id, labels)}")
            selected_b.info(f"Parent B: {_selected_parent_label(parent_b_id, labels)}")
    else:
        parent_a_id = None
        parent_b_id = None
        st.info("Turn on Breed? to pick two parents from the tree.")

    with st.expander("Manual fallback selectors"):
        options = [rock.id for rock in candidates]
        blank_parent_options = [None, *options]
        manual_a = st.selectbox(
            "Parent A",
            blank_parent_options,
            index=blank_parent_options.index(parent_a_id) if parent_a_id in blank_parent_options else 0,
            format_func=lambda rock_id: "Choose parent A" if rock_id is None else labels[rock_id],
            key="breeding_parent_a_select_fallback",
        )
        manual_b_options = [
            rock.id for rock in (_parent_b_options(manual_a, candidates) if manual_a else candidates)
        ]
        blank_parent_b_options = [None, *manual_b_options]
        manual_b = st.selectbox(
            "Parent B",
            blank_parent_b_options,
            index=blank_parent_b_options.index(parent_b_id) if parent_b_id in blank_parent_b_options else 0,
            format_func=lambda rock_id: "Choose parent B" if rock_id is None else labels[rock_id],
            key="breeding_parent_b_select_fallback",
        )
        if st.button("Use manual parent selectors", key="breeding_use_manual_parent_selectors"):
            st.session_state[MANUAL_PARENT_SELECTION_KEY] = (manual_a, manual_b)
            st.rerun()

    parent_a_id = st.session_state.get(PARENT_A_KEY)
    parent_b_id = st.session_state.get(PARENT_B_KEY)
    validation = {"valid": False, "errors": ["Choose two parents."], "warnings": []}
    if parent_a_id is not None and parent_b_id is not None:
        validation = game_controller.validate_breeding_pair(game, parent_a_id, parent_b_id)

    with section("Breeding Action"):
        potion_keys = st.multiselect(
            "Potions",
            [key for key in game_controller.get_potion_options(game) if key is not None],
            key=POTION_MULTISELECT_KEY,
            help="Attach any combination of owned potion types to this breeding pair.",
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
                options={"potion_keys": potion_keys},
            )
            st.session_state[PARENT_A_KEY] = None
            st.session_state[PARENT_B_KEY] = None
            st.session_state[CLEAR_PARENT_SELECTION_KEY] = True
            st.session_state[CLEAR_POTION_SELECTION_KEY] = True
            _store_action_and_rerun(result)

        if col_breed.button(
            "Breed queued pairs",
            key="breeding_advance_generation",
            disabled=queue_summary["queued_pairs"] == 0,
            type="primary",
        ):
            result = game_controller.advance_breeding_generation(game)
            _store_action_and_rerun(result)
