"""Serialization-safe observable traces from gameplay model execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelTrace:
    model_type: str
    checkpoint_id: str
    topology_id: str
    observation_schema_version: int
    normalizer_version: int
    observation_hash: str
    feature_names: tuple[str, ...]
    input_values: tuple[float, ...]
    node_activations: dict[str, float]
    connection_signals: tuple[dict[str, Any], ...]
    output_scores: dict[str, float]
    selected_candidate_ids: tuple[int | str, int | str] | None
    candidate_input_hashes: tuple[str, ...]
    trace_semantics: str = (
        "connection_signals are local source_activation * weight values, "
        "not causal explanations"
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

