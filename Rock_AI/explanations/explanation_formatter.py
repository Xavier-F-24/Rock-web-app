"""Compact human-facing formatting for structured explanations."""

from __future__ import annotations

from .decision_explanation_helper import DecisionExplanation


def format_decision_explanation(explanation: DecisionExplanation) -> str:
    if explanation.selected_parent_ids is None:
        return f"No pair selected: {explanation.fallback_reason or 'no reason recorded'}."
    names = explanation.selected_parent_names or tuple(map(str, explanation.selected_parent_ids))
    score = explanation.selected_candidate_score
    lines = [
        f"Selected {names[0]} and {names[1]}"
        + (f" with score {score:.3f}." if score is not None else "."),
        f"Rank {explanation.selected_pair_rank or '?'} of {explanation.total_legal_candidates} legal candidates.",
    ]
    lines.extend(explanation.notable_genetics_observations)
    lines.extend(f"Warning: {warning}" for warning in explanation.warnings)
    return "\n".join(lines)
