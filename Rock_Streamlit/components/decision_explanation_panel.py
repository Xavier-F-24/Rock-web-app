"""Structured, non-chain-of-thought decision explanation display."""

from __future__ import annotations

import streamlit as st

from Rock_AI.visualization.decision_plot_helper import build_contribution_figure, explanation_summary


def render_decision_explanation(explanation) -> None:
    st.subheader("Why this pair?")
    st.write(explanation_summary(explanation))
    if explanation is None:
        return
    columns = st.columns(4)
    columns[0].metric("Score", "-" if explanation.selected_candidate_score is None else f"{explanation.selected_candidate_score:.3f}")
    columns[1].metric("Rank", "-" if explanation.selected_pair_rank is None else f"{explanation.selected_pair_rank}/{explanation.total_legal_candidates}")
    columns[2].metric("Confidence", "-" if explanation.confidence_proxy is None else f"{explanation.confidence_proxy:.1%}")
    columns[3].metric("Lead", "-" if explanation.first_second_score_difference is None else f"{explanation.first_second_score_difference:.3f}")
    if explanation.score_component_contributions:
        try:
            st.plotly_chart(build_contribution_figure(explanation.score_component_contributions), width="stretch")
        except Exception:
            st.json(explanation.score_component_contributions)
    for observation in explanation.notable_genetics_observations:
        st.info(observation)
    for warning in explanation.warnings:
        st.warning(warning)
    if explanation.rejected_alternatives:
        with st.expander("Close alternatives"):
            st.dataframe(list(explanation.rejected_alternatives[:3]), width="stretch", hide_index=True)
