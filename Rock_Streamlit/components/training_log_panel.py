"""Raw worker log viewer."""

import streamlit as st

def render_training_log(text: str):
    with st.expander("Raw worker console"):
        st.code(text or "No console output yet.", language="text")
