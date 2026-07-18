"""Safe NEAT champion export and an instrumented feed-forward evaluator."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import neat


NEAT_NETWORK_ARTIFACT_VERSION = 1


@dataclass(frozen=True)
class NeatNodeDefinition:
    node_id: int
    bias: float
    response: float
    activation: str
    aggregation: str


@dataclass(frozen=True)
class NeatConnectionDefinition:
    source_id: int
    target_id: int
    weight: float
    enabled: bool


@dataclass(frozen=True)
class NeatNetworkArtifact:
    artifact_version: int
    topology_id: str
    observation_schema_version: int
    normalizer_version: int
    information_access: str
    input_ids: tuple[int, ...]
    output_ids: tuple[int, ...]
    input_feature_names: tuple[str, ...]
    nodes: tuple[NeatNodeDefinition, ...]
    connections: tuple[NeatConnectionDefinition, ...]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NeatNetworkArtifact":
        if int(data.get("artifact_version", -1)) != NEAT_NETWORK_ARTIFACT_VERSION:
            raise ValueError("Unsupported NEAT network artifact version")
        return cls(
            artifact_version=NEAT_NETWORK_ARTIFACT_VERSION,
            topology_id=str(data["topology_id"]),
            observation_schema_version=int(data["observation_schema_version"]),
            normalizer_version=int(data["normalizer_version"]),
            information_access=str(data["information_access"]),
            input_ids=tuple(map(int, data["input_ids"])),
            output_ids=tuple(map(int, data["output_ids"])),
            input_feature_names=tuple(data["input_feature_names"]),
            nodes=tuple(NeatNodeDefinition(**row) for row in data["nodes"]),
            connections=tuple(NeatConnectionDefinition(**row) for row in data["connections"]),
            metadata=dict(data.get("metadata", {})),
        )


def export_neat_genome(
    genome,
    config,
    input_feature_names: Sequence[str],
    *,
    observation_schema_version: int,
    normalizer_version: int,
    metadata: dict[str, Any] | None = None,
) -> NeatNetworkArtifact:
    input_ids = tuple(config.genome_config.input_keys)
    output_ids = tuple(config.genome_config.output_keys)
    if len(input_ids) != len(input_feature_names):
        raise ValueError("NEAT input count does not match the certified feature schema")
    nodes = tuple(
        NeatNodeDefinition(
            int(node_id), float(node.bias), float(node.response),
            str(node.activation), str(node.aggregation),
        )
        for node_id, node in sorted(genome.nodes.items())
    )
    connections = tuple(
        NeatConnectionDefinition(
            int(key[0]), int(key[1]), float(connection.weight), bool(connection.enabled)
        )
        for key, connection in sorted(genome.connections.items())
    )
    fingerprint = json.dumps(
        {"nodes": [asdict(row) for row in nodes], "connections": [asdict(row) for row in connections]},
        sort_keys=True,
    ).encode("utf-8")
    return NeatNetworkArtifact(
        NEAT_NETWORK_ARTIFACT_VERSION,
        hashlib.sha256(fingerprint).hexdigest()[:20],
        int(observation_schema_version),
        int(normalizer_version),
        "player",
        input_ids,
        output_ids,
        tuple(input_feature_names),
        nodes,
        connections,
        dict(metadata or {}),
    )


def save_neat_artifact(artifact: NeatNetworkArtifact, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def load_neat_artifact(path: str | Path) -> NeatNetworkArtifact:
    return NeatNetworkArtifact.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class InstrumentedNeatNetwork:
    """Evaluate an exported feed-forward topology without loading pickle data."""

    def __init__(self, artifact: NeatNetworkArtifact):
        self.artifact = artifact
        self._activation = neat.activations.ActivationFunctionSet()
        self._aggregation = neat.aggregations.AggregationFunctionSet()
        incoming: dict[int, list[NeatConnectionDefinition]] = {}
        for connection in artifact.connections:
            if connection.enabled:
                incoming.setdefault(connection.target_id, []).append(connection)
        required = set(artifact.output_ids)
        changed = True
        while changed:
            changed = False
            for target in tuple(required):
                for connection in incoming.get(target, ()):
                    if connection.source_id not in artifact.input_ids and connection.source_id not in required:
                        required.add(connection.source_id)
                        changed = True
        node_map = {node.node_id: node for node in artifact.nodes}
        pending = set(required)
        order: list[int] = []
        available = set(artifact.input_ids)
        while pending:
            ready = sorted(
                node_id for node_id in pending
                if all(c.source_id in available for c in incoming.get(node_id, ()))
            )
            if not ready:
                raise ValueError("Exported NEAT topology is not feed-forward")
            for node_id in ready:
                if node_id not in node_map:
                    raise ValueError(f"Missing NEAT node definition {node_id}")
                pending.remove(node_id)
                available.add(node_id)
                order.append(node_id)
        self._nodes = node_map
        self._incoming = incoming
        self._order = tuple(order)

    def activate(self, inputs: Sequence[float]) -> tuple[tuple[float, ...], dict[str, Any]]:
        if len(inputs) != len(self.artifact.input_ids):
            raise ValueError("NEAT input dimension is incompatible with the artifact")
        values = {key: float(value) for key, value in zip(self.artifact.input_ids, inputs)}
        signals: list[dict[str, Any]] = []
        for node_id in self._order:
            node = self._nodes[node_id]
            weighted = []
            for connection in self._incoming.get(node_id, ()):
                signal = values[connection.source_id] * connection.weight
                weighted.append(signal)
                signals.append({
                    "source_id": connection.source_id,
                    "target_id": connection.target_id,
                    "weight": connection.weight,
                    "source_activation": values[connection.source_id],
                    "local_signal": signal,
                })
            aggregate = self._aggregation.get(node.aggregation)(weighted)
            values[node_id] = self._activation.get(node.activation)(
                node.bias + node.response * aggregate
            )
        outputs = tuple(values[node_id] for node_id in self.artifact.output_ids)
        return outputs, {
            "node_activations": {str(key): value for key, value in values.items()},
            "connection_signals": signals,
        }
