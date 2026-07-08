"""Market page scaffold."""

from __future__ import annotations

import streamlit as st

from Rock_Streamlit.app_state import get_game_state
from rockgame_ui import game_controller


def render() -> None:
    game = get_game_state()

    st.title("Market")
    st.caption("Placeholder shell for potions, imports, selling, and market pods.")

    st.subheader("Available Actions")
    st.write("- Buy potions")
    st.write("- Import a random rock")
    st.write("- Request a defined-trait rock")
    st.write("- Buy a breeding pod and keep one child")
    st.write("- Sell rocks")

    st.subheader("Potion Shop")
    st.dataframe(game_controller.get_potion_rows(game), use_container_width=True, hide_index=True)

    st.subheader("Quick Actions")
    if st.button("Buy Random Rock"):
        result = game_controller.buy_random_rock(game)
        if result.ok:
            st.success(result.message)
            st.rerun()
        else:
            st.error(result.message)

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
