"""Serialization-safe recurrent topology records and operational limits."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


RECURRENT_TOPOLOGY_VERSION = 1


@dataclass(frozen=True)
class TopologyResourceLimits:
    max_hidden_nodes: int = 128
    max_enabled_connections: int = 4096
    max_total_genes: int = 8192
    max_recurrent_settling_steps: int = 12
    max_genome_evaluation_seconds: float = 30.0
    max_episode_decisions: int = 100
    max_trace_nodes: int = 256
    max_trace_connections: int = 512

    def __post_init__(self) -> None:
        integer_limits = (
            self.max_hidden_nodes, self.max_enabled_connections, self.max_total_genes,
            self.max_recurrent_settling_steps, self.max_episode_decisions,
            self.max_trace_nodes, self.max_trace_connections,
        )
        if min(integer_limits) <= 0 or self.max_genome_evaluation_seconds <= 0:
            raise ValueError("All topology resource limits must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RecurrentNodeGene:
    node_id: int
    node_type: str
    bias: float
    response: float
    activation: str
    aggregation: str


@dataclass(frozen=True)
class RecurrentConnectionGene:
    source_id: int
    target_id: int
    weight: float
    enabled: bool
    innovation_id: int | None = None
    recurrent: bool = False
    self_loop: bool = False


@dataclass(frozen=True)
class RecurrentTopologyArtifact:
    artifact_version: int
    topology_id: str
    genome_id: int | str
    observation_schema_version: int
    normalizer_version: int
    information_access: str
    input_ids: tuple[int, ...]
    output_ids: tuple[int, ...]
    input_feature_names: tuple[str, ...]
    output_names: tuple[str, ...]
    nodes: tuple[RecurrentNodeGene, ...]
    connections: tuple[RecurrentConnectionGene, ...]
    evaluation_config: dict[str, Any]
    resource_limits: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.artifact_version != RECURRENT_TOPOLOGY_VERSION:
            raise ValueError("Unsupported recurrent topology artifact version")
        if self.information_access != "player":
            raise ValueError("Recurrent gameplay topology must be player-certified")
        if len(self.input_ids) != len(self.input_feature_names):
            raise ValueError("Recurrent input IDs and feature names must align")
        if len(self.output_ids) != len(self.output_names):
            raise ValueError("Recurrent output IDs and names must align")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecurrentTopologyArtifact":
        return cls(
            artifact_version=int(data["artifact_version"]),
            topology_id=str(data["topology_id"]),
            genome_id=data["genome_id"],
            observation_schema_version=int(data["observation_schema_version"]),
            normalizer_version=int(data["normalizer_version"]),
            information_access=str(data["information_access"]),
            input_ids=tuple(map(int, data["input_ids"])),
            output_ids=tuple(map(int, data["output_ids"])),
            input_feature_names=tuple(data["input_feature_names"]),
            output_names=tuple(data["output_names"]),
            nodes=tuple(RecurrentNodeGene(**row) for row in data["nodes"]),
            connections=tuple(RecurrentConnectionGene(**row) for row in data["connections"]),
            evaluation_config=dict(data["evaluation_config"]),
            resource_limits=dict(data["resource_limits"]),
            metadata=dict(data.get("metadata", {})),
        )

    @property
    def hidden_node_count(self) -> int:
        return sum(node.node_type == "hidden" for node in self.nodes)

    @property
    def enabled_connection_count(self) -> int:
        return sum(connection.enabled for connection in self.connections)

    @property
    def recurrent_connection_count(self) -> int:
        return sum(connection.enabled and connection.recurrent for connection in self.connections)


def _has_path(adjacency: dict[int, set[int]], start: int, target: int) -> bool:
    pending = [start]
    seen = set()
    while pending:
        node = pending.pop()
        if node == target:
            return True
        if node in seen:
            continue
        seen.add(node)
        pending.extend(adjacency.get(node, ()))
    return False


def classify_recurrent_edges(connections: Iterable[tuple[int, int]]) -> set[tuple[int, int]]:
    edges = tuple(connections)
    recurrent = {(source, target) for source, target in edges if source == target}
    for source, target in edges:
        if source == target:
            continue
        adjacency: dict[int, set[int]] = {}
        for left, right in edges:
            if (left, right) != (source, target):
                adjacency.setdefault(left, set()).add(right)
        if _has_path(adjacency, target, source):
            recurrent.add((source, target))
    return recurrent


def topology_fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
