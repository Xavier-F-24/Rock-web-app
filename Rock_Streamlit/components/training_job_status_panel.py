"""Compact status display for a durable training worker."""

from __future__ import annotations

import streamlit as st


def status_metrics(status) -> tuple[dict, ...]:
    return (
        {"label": "Status", "value": status.status.value.replace("_", " ").title()},
        {"label": "Generation", "value": f"{status.current_evolutionary_generation} / {status.requested_ending_generation}"},
        {"label": "Best fitness", "value": "-" if status.current_best_training_fitness is None else f"{status.current_best_training_fitness:.4f}"},
        {"label": "Validation", "value": "-" if status.current_best_validation_fitness is None else f"{status.current_best_validation_fitness:.4f}"},
    )


def render_training_job_status(status, orphan_warning=None):
    columns = st.columns(4)
    for column, metric in zip(columns, status_metrics(status)): column.metric(metric["label"], metric["value"])
    st.caption(f"Job {status.job_id} | PID {status.process_id or '-'} | Heartbeat {status.last_heartbeat_time or '-'}")
    if orphan_warning: st.warning(orphan_warning)
    if status.failure_summary: st.error(status.failure_summary)
    for warning in status.warnings: st.warning(warning)
