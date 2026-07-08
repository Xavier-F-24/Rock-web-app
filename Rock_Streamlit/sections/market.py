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

    st.subheader("Potion Shop")
    st.dataframe(game_controller.get_potion_rows(game), use_container_width=True, hide_index=True)

    st.subheader("Market Pods")
    market_rows = game_controller.get_market_pod_rows(game)
    if market_rows:
        st.dataframe(
            market_rows,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No market pods available yet.")
