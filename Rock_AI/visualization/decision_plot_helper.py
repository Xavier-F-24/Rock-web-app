"""Stable table/chart records derived from structured agent explanations."""

from __future__ import annotations


def candidate_table_rows(candidates, *, selected_parent_ids=()) -> list[dict]:
    selected = tuple(map(str, selected_parent_ids or ()))
    selected_key = tuple(sorted(selected)) if len(selected) == 2 else ()
    rows = []
    previous_score = None
    for candidate in sorted(candidates, key=lambda item: item.rank):
        key = tuple(sorted(map(str, candidate.parent_ids)))
        rows.append(
            {
                "rank": candidate.rank,
                "parent_a": candidate.parent_names[0],
                "parent_b": candidate.parent_names[1],
                "parent_a_id": candidate.parent_ids[0],
                "parent_b_id": candidate.parent_ids[1],
                "score": candidate.score,
                "expected_survivors": candidate.predicted_expected_survivors,
                "expected_average_child_value": candidate.predicted_average_child_value,
                "expected_maximum_child_value": candidate.predicted_maximum_child_value,
                "mutation_probability": candidate.mutation_probability,
                "genotype_diversity": candidate.genotype_diversity,
                "phenotype_diversity": candidate.phenotype_diversity,
                "uncertainty": candidate.uncertainty,
                "score_margin": None if previous_score is None else previous_score - candidate.score,
                "selected": key == selected_key,
                "legal": candidate.legality_confirmed,
            }
        )
        previous_score = candidate.score
    return rows


def explanation_summary(explanation) -> str:
    if explanation is None:
        return "Run one decision to see why the agent preferred a pair."
    names = explanation.selected_parent_names
    if not names:
        return explanation.fallback_reason or "The agent did not select a breeding pair."
    drivers = sorted(
        explanation.score_component_contributions.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:2]
    driver_text = " and ".join(name.replace("_", " ") for name, _ in drivers)
    basis = f", driven by {driver_text}" if driver_text else ""
    return (
        f"Selected {names[0]} and {names[1]} at rank "
        f"{explanation.selected_pair_rank} of {explanation.total_legal_candidates}{basis}."
    )


def build_contribution_figure(contributions: dict[str, float]):
    import plotly.graph_objects as go

    ordered = sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)
    return go.Figure(
        go.Bar(
            x=[value for _, value in ordered],
            y=[name.replace("_", " ").title() for name, _ in ordered],
            orientation="h",
            marker_color=["#2f855a" if value >= 0 else "#c2415d" for _, value in ordered],
        )
    ).update_layout(height=260, margin={"l": 20, "r": 20, "t": 10, "b": 20})
