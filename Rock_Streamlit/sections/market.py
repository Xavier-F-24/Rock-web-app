"""Market page scaffold."""

from __future__ import annotations

import streamlit as st

from Rock_Streamlit.app_state import get_game_state
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
    st.subheader("Potion Shop")
    potion_rows = game_controller.get_potion_rows(game)
    if not potion_rows:
        st.info("No potions available.")
        return

    st.dataframe(potion_rows, use_container_width=True, hide_index=True)

    st.caption("Choose quantities, then purchase them together.")
    columns = st.columns(len(potion_rows))
    quantities = {}
    for column, row in zip(columns, potion_rows):
        with column:
            st.write(row["name"])
            st.caption(f"${row['cost']} each | owned: {row['owned']}")
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
    st.subheader("Market Pods")

    pending_rows = game_controller.get_pending_market_pod_rows(game)
    if pending_rows:
        st.warning("Choose one child from the pending market pod before buying another pod.")
        st.dataframe(pending_rows, use_container_width=True, hide_index=True)

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

    st.dataframe(market_rows, use_container_width=True, hide_index=True)
    available_rows = [row for row in market_rows if not row["used"]]
    if not available_rows:
        st.info("All current market pods have been used.")
        return

    pod_options = {
        f"{row['name']} ({row['tier']}) - ${row['price']}": row["offer_id"]
        for row in available_rows
    }
    selected_pod_label = st.selectbox(
        "Pod to buy",
        list(pod_options),
        key="market_pod_select",
    )
    if st.button("Purchase selected pod", key="market_purchase_pod", type="primary"):
        result = game_controller.buy_market_pod(game, pod_options[selected_pod_label])
        _store_action_and_rerun(result)


def render() -> None:
    game = get_game_state()
    summary = game_controller.get_game_summary(game)

    st.title("Market")
    st.caption("Imports, selling, potions, and market pods.")
    _show_last_action_message()

    st.metric("Money", f"${summary['money']}")

    active_rows = game_controller.get_active_rock_rows(game)
    sellable_rows = game_controller.get_sellable_rock_rows(game)

    st.subheader("Owned Active Rocks")
    if active_rows:
        st.dataframe(active_rows, use_container_width=True, hide_index=True)
    else:
        st.info("No active rocks owned yet.")

    st.subheader("Sell Rocks")
    if sellable_rows:
        st.dataframe(sellable_rows, use_container_width=True, hide_index=True)

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

    st.subheader("Imports")
    if st.button("Buy random import rock", key="market_buy_random_import"):
        result = game_controller.buy_random_rock(game)
        _store_action_and_rerun(result)

    _render_potion_shop(game)
    _render_market_pods(game)
