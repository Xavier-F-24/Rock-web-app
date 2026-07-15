"""Generation deltas and campaign metric history."""

from __future__ import annotations

from dataclasses import asdict

import streamlit as st

from Rock_AI.visualization.metrics_plot_helper import build_generation_summaries, build_metric_figure, build_metric_series


def render_generation_summaries(events) -> None:
    st.subheader("Generation Summaries")
    summaries = build_generation_summaries(events)
    if summaries:
        st.dataframe([asdict(summary) for summary in summaries], width="stretch", hide_index=True)
    else:
        st.caption("A summary appears after the first generation advances.")
    series = build_metric_series(events)
    if series:
        try:
            st.plotly_chart(build_metric_figure(series), width="stretch")
        except Exception:
            st.dataframe(series, width="stretch", hide_index=True)
