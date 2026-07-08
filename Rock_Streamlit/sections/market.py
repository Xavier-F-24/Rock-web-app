"""Market page for imports, potions, pods, and selling."""

from __future__ import annotations

import streamlit as st

from Rock_Streamlit.app_state import get_game_state
from Rock_Streamlit.ui_components import metric_strip, page_header, section
from rockgame_ui import game_controller


MARKET_MESSAGE_KEY = "market_last_action"


def _show_last_action_message() -> None:
    result = st.session_state.pop(MARKET_MESSAGE_KEY, None)
    if result is None:
        return

    if result.ok:
        st.success(result.message)
    else:
        st.error(result.message)


def _store_action_and_rerun(result: game_controller.ActionResult) -> None:
    st.session_state[MARKET_MESSAGE_KEY] = result
    st.rerun()


def _render_potion_shop(game) -> None:
    with section("Potion Shop", "Choose quantities, then purchase them together."):
        potion_rows = game_controller.get_potion_rows(game)
        if not potion_rows:
            st.info("No potions available.")
            return

        columns = st.columns(len(potion_rows))
        quantities = {}
        for column, row in zip(columns, potion_rows):
            with column:
                st.markdown(f"**{row['name']}**")
                st.caption(row["description"])
                st.write(f"${row['cost']} each")
                st.write(f"Owned: {row['owned']}")
                quantities[row["key"]] = st.number_input(
                    "Qty",
                    min_value=0,
                    max_value=99,
                    value=0,
                    step=1,
                    key=f"market_potion_qty_{row['key']}",
                )

        total_cost = sum(row["cost"] * quantities[row["key"]] for row in potion_rows)
        st.write(f"Potion cart total: ${total_cost}")
        if st.button(
            "Purchase selected potions",
            key="market_purchase_potions",
            disabled=total_cost <= 0,
            type="primary",
        ):
            result = game_controller.buy_potions(game, quantities)
            _store_action_and_rerun(result)


def _render_market_pods(game) -> None:
    with section("Market Pods", "Buy a guest-family pod, then keep one child."):
        pending_rows = game_controller.get_pending_market_pod_rows(game)
        if pending_rows:
            st.warning("Choose one child from the pending market pod before buying another pod.")
            st.dataframe(pending_rows, width="stretch", hide_index=True)

            child_options = {
                f"{row['index']}: {row['name']} ({row['sex']}, value ${row['value']})": row["index"]
                for row in pending_rows
            }
            selected_child_label = st.selectbox(
                "Child to keep",
                list(child_options),
                key="market_pending_pod_child_select",
            )
            if st.button("Keep selected child", key="market_keep_pod_child", type="primary"):
                result = game_controller.choose_market_pod_child(
                    game,
                    child_options[selected_child_label],
                )
                _store_action_and_rerun(result)
            return

        market_rows = game_controller.get_market_pod_rows(game)
        if not market_rows:
            st.info("No market pods available yet.")
            return

        available_rows = [row for row in market_rows if not row["used"]]
        if not available_rows:
            st.info("All current market pods have been used.")
            return

        pod_cols = st.columns(len(available_rows))
        for column, row in zip(pod_cols, available_rows):
            with column:
                st.markdown(f"**{row['name']}**")
                st.caption(row["tagline"])
                st.write(f"Tier: {row['tier']}")
                st.write(f"Price: ${row['price']}")
                if st.button("Purchase", key=f"market_purchase_pod_{row['offer_id']}"):
                    result = game_controller.buy_market_pod(game, row["offer_id"])
                    _store_action_and_rerun(result)


def render() -> None:
    game = get_game_state()
    summary = game_controller.get_game_summary(game)

    page_header("Market", "Imports, selling, potions, and market pods.")
    _show_last_action_message()

    metric_strip([("Money", f"${summary['money']}")])

    active_rows = game_controller.get_active_rock_rows(game)
    sellable_rows = game_controller.get_sellable_rock_rows(game)

    with section("Imports"):
        if st.button("Buy random import rock", key="market_buy_random_import", type="primary"):
            result = game_controller.buy_random_rock(game)
            _store_action_and_rerun(result)

    _render_potion_shop(game)
    _render_market_pods(game)

    with section("Sell Rocks"):
        if sellable_rows:
            sell_options = {
                f"#{row['id']} {row['name']} - ${row['sell_value']}": row["id"]
                for row in sellable_rows
            }
            selected_sell_label = st.selectbox(
                "Eligible rock",
                list(sell_options),
                key="market_sell_rock_select",
            )
            selected_sell_id = sell_options[selected_sell_label]

            if st.button(
                "Sell selected rock",
                key=f"market_sell_rock_{selected_sell_id}",
                type="primary",
            ):
                result = game_controller.sell_rock(game, selected_sell_id)
                _store_action_and_rerun(result)
        else:
            st.info("No rocks are currently eligible to sell.")

    with section("Owned Active Rocks"):
        if active_rows:
            st.dataframe(active_rows, width="stretch", hide_index=True)
        else:
            st.info("No active rocks owned yet.")
