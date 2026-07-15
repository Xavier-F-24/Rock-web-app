from __future__ import annotations

from Rock_AI.explanations.candidate_explanation_helper import CandidateExplanation
from Rock_AI.runtime.runtime_event_helper import RuntimeEvent, RuntimeEventType
from Rock_AI.visualization.decision_plot_helper import candidate_table_rows
from Rock_AI.visualization.metrics_plot_helper import build_generation_summaries, build_metric_series


def _candidate(rank, score, ids):
    return CandidateExplanation(
        parent_ids=ids,
        parent_names=(f"Rock {ids[0]}", f"Rock {ids[1]}"),
        score=score,
        rank=rank,
        legality_confirmed=True,
    )


def test_candidate_rows_preserve_rank_and_selected_pair():
    rows = candidate_table_rows(
        [_candidate(2, 4.0, (3, 4)), _candidate(1, 5.0, (1, 2))],
        selected_parent_ids=(2, 1),
    )
    assert [row["rank"] for row in rows] == [1, 2]
    assert rows[0]["selected"]
    assert rows[1]["score_margin"] == 1.0


def test_generation_summary_and_metric_series_calculate_deltas():
    pre = {
        "rock_count": 4,
        "active_rock_count": 4,
        "final_active_rock_value": 20.0,
        "final_maximum_rock_value": 8.0,
        "genotype_diversity": 0.5,
        "phenotype_diversity": 0.5,
        "rare_trait_count": 1,
    }
    post = {
        "rock_count": 7,
        "active_rock_count": 6,
        "final_active_rock_value": 38.0,
        "final_maximum_rock_value": 12.0,
        "genotype_diversity": 0.75,
        "phenotype_diversity": 0.6,
        "rare_trait_count": 3,
    }
    events = [
        RuntimeEvent("s", 0, 0, 0, RuntimeEventType.PAIR_SELECTED, "pair"),
        RuntimeEvent("s", 1, 0, 0, RuntimeEventType.CHILDREN_CREATED, "children", rock_ids=(5, 6, 7)),
        RuntimeEvent("s", 2, 0, 0, RuntimeEventType.MUTATION_OCCURRED, "mutation", payload={"mutation_count": 2}),
        RuntimeEvent(
            "s", 3, 0, 0, RuntimeEventType.GENERATION_ADVANCED, "advance",
            payload={"from_generation": 0, "to_generation": 1},
            pre_action_metrics=pre,
            post_action_metrics=post,
        ),
    ]
    summary = build_generation_summaries(events)[0]
    assert summary.pairs_bred == 1
    assert summary.children_produced == 3
    assert summary.mutations == 2
    assert summary.farm_value_delta == 18.0
    assert summary.rare_traits_delta == 2
    assert build_metric_series(events)[0]["maximum_value"] == 12.0
