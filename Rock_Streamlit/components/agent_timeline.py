"""Chronological runtime event timeline."""

from __future__ import annotations

import streamlit as st

from .observatory_state_helper import timeline_rows


EVENT_LABELS = {
    "decision": "Decision",
    "breeding": "Breeding",
    "birth": "Birth",
    "mutation": "Mutation",
    "status": "Status",
    "generation": "Generation",
    "pause": "Pause",
    "completion": "Complete",
    "failure": "Failure",
    "info": "Info",
}


def render_timeline(events, *, limit: int = 50) -> None:
    st.subheader("Event Timeline")
    rows = timeline_rows(events)[-limit:]
    if not rows:
        st.caption("No runtime events yet.")
        return
    for row in reversed(rows):
        label = EVENT_LABELS[row["category"]]
        with st.expander(f"{label} | Gen {row['generation']} D{row['decision']}: {row['summary']}"):
            if row["rock_ids"]:
                st.caption("Rocks: " + ", ".join(map(str, row["rock_ids"])))
            st.json(row["details"])
