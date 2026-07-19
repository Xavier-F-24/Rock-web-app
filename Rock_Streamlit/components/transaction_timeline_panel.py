import streamlit as st

from Rock_AI.visualization.economy_timeline_adapter import economy_timeline_rows


def render_transaction_timeline(world):
    rows = economy_timeline_rows(world)
    if not rows:
        st.info("No public economy events yet.")
        return
    for row in reversed(rows[-30:]):
        with st.expander(f"Turn {row['turn']} | {row['type']}: {row['summary']}"):
            st.json(row)
