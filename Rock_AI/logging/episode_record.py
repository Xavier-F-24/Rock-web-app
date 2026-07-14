"""Complete record of one breeding-only campaign."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .agent_decision_record import AgentDecisionRecord


@dataclass
class EpisodeRecord:
    episode_id: str
    initial_seed: int
    agent_seed: int
    agent_configuration: dict[str, Any]
    environment_configuration: dict[str, Any]
    breeding_rules: dict[str, Any]
    initial_farm_summary: dict[str, float | int]
    decisions: list[AgentDecisionRecord]
    final_farm_summary: dict[str, float | int]
    termination_reason: str
    total_generations: int
    total_breeding_decisions: int
    runtime_seconds: float
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
