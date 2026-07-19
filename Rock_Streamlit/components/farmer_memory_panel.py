import streamlit as st


def render_farmer_memory(agent):
    policy = getattr(agent, "policy", None)
    if policy is None:
        st.caption("This baseline has no neural recurrent memory.")
        return
    state = policy.export_state()
    st.metric("Committed decisions", state["decision_count"])
    with st.expander("Recurrent activation state"):
        st.json(state)
