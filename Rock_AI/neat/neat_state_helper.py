"""Typed player-safe temporal context and recurrent agent memory."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from Rock_AI.representations.player_observation_helper import PlayerFeatureVector, PlayerObservation


RECURRENT_STATE_VERSION = 1
TEMPORAL_FEATURE_NAMES = (
    "decision_index", "generation_index", "prior_breedings_this_generation",
    "farm_value_change", "visible_diversity_change", "previous_active_child_count",
    "previous_dead_child_count", "previous_craisened_child_count",
    "previous_mutation_observed", "previous_selected_candidate_score", "previous_stop_action",
)


@dataclass(frozen=True)
class RecurrentDecisionObservation:
    player_observation: PlayerObservation
    temporal_context: PlayerFeatureVector

    def __post_init__(self) -> None:
        if not isinstance(self.player_observation, PlayerObservation):
            raise TypeError("Recurrent policy requires PlayerObservation")
        if tuple(self.temporal_context.feature_names) != TEMPORAL_FEATURE_NAMES:
            raise ValueError("Temporal context schema is incompatible")


@dataclass(frozen=True)
class RecurrentAgentState:
    topology_id: str
    genome_id: int | str
    episode_id: str
    decision_count: int = 0
    node_activations: tuple[tuple[int, float], ...] = ()
    previous_outputs: tuple[float, ...] = ()
    temporal_context: tuple[float, ...] = ()
    state_schema_version: int = RECURRENT_STATE_VERSION

    def activation_map(self) -> dict[int, float]:
        return {int(key): float(value) for key, value in self.node_activations}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecurrentAgentState":
        state = cls(
            topology_id=str(data["topology_id"]), genome_id=data["genome_id"],
            episode_id=str(data["episode_id"]), decision_count=int(data.get("decision_count", 0)),
            node_activations=tuple((int(k), float(v)) for k, v in data.get("node_activations", ())),
            previous_outputs=tuple(map(float, data.get("previous_outputs", ()))),
            temporal_context=tuple(map(float, data.get("temporal_context", ()))),
            state_schema_version=int(data.get("state_schema_version", RECURRENT_STATE_VERSION)),
        )
        if state.state_schema_version != RECURRENT_STATE_VERSION:
            raise ValueError("Unsupported recurrent state schema version")
        return state

    def validate_for(self, topology_id: str, genome_id: int | str) -> None:
        if self.topology_id != topology_id or str(self.genome_id) != str(genome_id):
            raise ValueError("Recurrent state is incompatible with the selected topology")
