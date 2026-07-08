"""Market page scaffold."""

from __future__ import annotations

import streamlit as st

from Rock_Market.rock_market_helper import POTION_SHOP
from Rock_Streamlit.app_state import get_game_state


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
    potion_rows = [
        {
            "key": key,
            "name": potion["name"],
            "cost": potion["cost"],
            "description": potion["description"],
            "owned": game.potions.get(key, 0),
        }
        for key, potion in POTION_SHOP.items()
    ]
    st.dataframe(potion_rows, use_container_width=True, hide_index=True)

    st.subheader("Market Pods")
    if game.market_pods:
        st.dataframe(
            [
                {
                    "offer_id": offer.offer_id,
                    "name": offer.name,
                    "tier": offer.tier,
                    "price": offer.price,
                    "used": offer.used,
                    "tagline": offer.tagline,
                }
                for offer in game.market_pods
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No market pods available yet.")
