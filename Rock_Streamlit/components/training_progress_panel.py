"""Structured progress charts and event records."""

from __future__ import annotations

import streamlit as st


def generation_rows(events: list[dict]) -> list[dict]:
    rows = [row for row in events if row.get("event_type") in {"generation_completed", "generation_evaluated"}]
    by_generation = {}
    for row in rows: by_generation.setdefault(int(row.get("generation", 0)), {}).update(row)
    return [by_generation[key] for key in sorted(by_generation)]


def render_training_progress(events):
    rows = generation_rows(events)
    if rows:
        st.line_chart({
            "Best fitness": [row.get("best_fitness") for row in rows],
            "Validation": [row.get("validation_quality") for row in rows],
            "Species": [row.get("species_count") for row in rows],
        })
    with st.expander("Structured event log", expanded=True):
        st.dataframe(list(reversed(events[-200:])), width="stretch", hide_index=True)
