"""Metric series and generation-boundary summaries for observatory plots."""

from __future__ import annotations

from dataclasses import dataclass

from Rock_AI.runtime.runtime_event_helper import RuntimeEventType


@dataclass(frozen=True)
class GenerationSummary:
    generation: int
    pairs_bred: int
    children_produced: int
    mutations: int
    rocks_delta: int
    active_rocks_delta: int
    farm_value_delta: float
    maximum_value_delta: float
    genotype_diversity_delta: float
    phenotype_diversity_delta: float
    rare_traits_delta: int


def build_metric_series(events) -> list[dict]:
    rows = []
    for event in events:
        metrics = event.post_action_metrics
        if not metrics:
            continue
        rows.append(
            {
                "event_index": event.event_index,
                "decision_index": event.decision_index,
                "generation": event.generation,
                "farm_value": float(metrics.get("final_active_rock_value", 0.0)),
                "maximum_value": float(metrics.get("final_maximum_rock_value", 0.0)),
                "genotype_diversity": float(metrics.get("genotype_diversity", 0.0)),
                "phenotype_diversity": float(metrics.get("phenotype_diversity", 0.0)),
            }
        )
    return rows


def build_generation_summaries(events) -> list[GenerationSummary]:
    summaries = []
    all_events = list(events)
    for boundary in all_events:
        if boundary.event_type != RuntimeEventType.GENERATION_ADVANCED:
            continue
        generation = int(boundary.payload.get("from_generation", boundary.generation))
        related = [event for event in all_events if event.generation == generation]
        pre = boundary.pre_action_metrics or {}
        post = boundary.post_action_metrics or {}
        summaries.append(
            GenerationSummary(
                generation=generation,
                pairs_bred=sum(event.event_type == RuntimeEventType.PAIR_SELECTED for event in related),
                children_produced=sum(
                    len(event.rock_ids)
                    for event in related
                    if event.event_type == RuntimeEventType.CHILDREN_CREATED
                ),
                mutations=sum(
                    int(event.payload.get("mutation_count", 0))
                    for event in related
                    if event.event_type == RuntimeEventType.MUTATION_OCCURRED
                ),
                rocks_delta=int(post.get("rock_count", 0) - pre.get("rock_count", 0)),
                active_rocks_delta=int(post.get("active_rock_count", 0) - pre.get("active_rock_count", 0)),
                farm_value_delta=float(post.get("final_active_rock_value", 0) - pre.get("final_active_rock_value", 0)),
                maximum_value_delta=float(post.get("final_maximum_rock_value", 0) - pre.get("final_maximum_rock_value", 0)),
                genotype_diversity_delta=float(post.get("genotype_diversity", 0) - pre.get("genotype_diversity", 0)),
                phenotype_diversity_delta=float(post.get("phenotype_diversity", 0) - pre.get("phenotype_diversity", 0)),
                rare_traits_delta=int(post.get("rare_trait_count", 0) - pre.get("rare_trait_count", 0)),
            )
        )
    return summaries


def build_metric_figure(series: list[dict]):
    import plotly.graph_objects as go

    figure = go.Figure()
    x = [row["decision_index"] for row in series]
    for key, label, color in (
        ("farm_value", "Active farm value", "#2f855a"),
        ("maximum_value", "Maximum rock value", "#b7791f"),
    ):
        figure.add_trace(go.Scatter(x=x, y=[row[key] for row in series], name=label, line={"color": color}))
    figure.update_layout(height=280, margin={"l": 20, "r": 20, "t": 20, "b": 20})
    return figure
