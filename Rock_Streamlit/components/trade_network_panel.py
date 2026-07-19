import streamlit as st

from Rock_AI.visualization.trade_network_visualizer import trade_network_rows


def render_trade_network(world):
    rows = trade_network_rows(world)
    st.dataframe(rows, width="stretch", hide_index=True) if rows else st.info("No direct trade offers yet.")
