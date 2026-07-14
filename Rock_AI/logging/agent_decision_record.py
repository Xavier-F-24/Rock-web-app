"""One auditable breeding-agent decision."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AgentDecisionRecord:
    episode_id: str
    decision_index: int
    generation: int
    agent_name: str
    observation_summary: dict[str, Any]
    legal_action_count: int
    selected_action: dict[str, Any]
    selected_parent_ids: tuple[int | str, int | str] | None
    ranked_candidate_pairs: list[dict[str, Any]] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    predictor_outputs: dict[str, Any] | None = None
    objective_weights: dict[str, Any] = field(default_factory=dict)
    pre_action_farm_metrics: dict[str, float | int] = field(default_factory=dict)
    immediate_post_action_farm_metrics: dict[str, float | int] = field(default_factory=dict)
    post_action_farm_metrics: dict[str, float | int] = field(default_factory=dict)
    resulting_child_ids: list[int | str] = field(default_factory=list)
    resulting_child_values: list[float] = field(default_factory=list)
    mutation_outcomes: list[dict[str, Any]] = field(default_factory=list)
    status: str = "continue"
    environment_seed: int = 0
    agent_seed: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
