import streamlit as st

from Rock_AI.visualization.action_score_visualizer import action_score_rows


def render_action_candidates(decision):
    rows = action_score_rows(decision)
    if not rows:
        st.info("Run a world round to inspect scored heterogeneous actions.")
        return
    st.dataframe([{k: v for k, v in row.items() if k != "target"} for row in rows], width="stretch", hide_index=True)
    selected = next((row for row in rows if row["selected"]), None)
    if selected:
        with st.expander("Selected action terms", expanded=True):
            st.json(selected["target"])
