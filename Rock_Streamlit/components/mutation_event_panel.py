"""Focused mutation and offspring event display."""

from __future__ import annotations

import streamlit as st

from Rock_AI.runtime.runtime_event_helper import RuntimeEventType


def render_mutation_events(events) -> None:
    mutation_events = [event for event in events if event.event_type == RuntimeEventType.MUTATION_OCCURRED]
    child_events = [event for event in events if event.event_type == RuntimeEventType.CHILDREN_CREATED]
    st.subheader("Offspring and Mutations")
    if child_events:
        latest = child_events[-1]
        st.success(f"Latest clutch: {len(latest.rock_ids)} children ({', '.join(map(str, latest.rock_ids))})")
    if mutation_events:
        for event in reversed(mutation_events[-5:]):
            st.warning(f"Gen {event.generation}: {event.summary}")
    elif not child_events:
        st.caption("No offspring events yet.")
