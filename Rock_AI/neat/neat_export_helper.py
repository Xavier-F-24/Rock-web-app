"""Safe JSON export for recurrent NEAT genomes."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from .neat_recurrent_network import RecurrentEvaluationConfig
from .neat_topology_helper import (
    RECURRENT_TOPOLOGY_VERSION, RecurrentConnectionGene, RecurrentNodeGene,
    RecurrentTopologyArtifact, TopologyResourceLimits, classify_recurrent_edges,
    topology_fingerprint,
)


def export_recurrent_genome(
    genome, config, input_feature_names: Sequence[str], *,
    observation_schema_version: int, normalizer_version: int,
    output_names: Sequence[str] = ("pair_score", "stop_preference", "confidence"),
    evaluation_config: RecurrentEvaluationConfig | None = None,
    resource_limits: TopologyResourceLimits | None = None,
    metadata: dict[str, Any] | None = None,
) -> RecurrentTopologyArtifact:
    evaluation_config = evaluation_config or RecurrentEvaluationConfig()
    resource_limits = resource_limits or TopologyResourceLimits()
    input_ids = tuple(config.genome_config.input_keys)
    output_ids = tuple(config.genome_config.output_keys)
    if len(input_ids) != len(input_feature_names):
        raise ValueError("Recurrent input count does not match the player feature schema")
    if len(output_ids) != len(output_names):
        raise ValueError("Recurrent output count does not match output names")
    enabled_edges = [key for key, gene in genome.connections.items() if gene.enabled]
    recurrent = classify_recurrent_edges(enabled_edges)
    nodes = tuple(RecurrentNodeGene(
        int(node_id), "output" if node_id in output_ids else "hidden",
        float(node.bias), float(node.response), str(node.activation), str(node.aggregation),
    ) for node_id, node in sorted(genome.nodes.items()))
    connections = tuple(RecurrentConnectionGene(
        int(source), int(target), float(gene.weight), bool(gene.enabled),
        int(getattr(gene, "innovation", 0)) or None,
        (source, target) in recurrent, source == target,
    ) for (source, target), gene in sorted(genome.connections.items()))
    payload = {"nodes": [asdict(row) for row in nodes], "connections": [asdict(row) for row in connections]}
    artifact = RecurrentTopologyArtifact(
        RECURRENT_TOPOLOGY_VERSION, topology_fingerprint(payload), genome.key,
        int(observation_schema_version), int(normalizer_version), "player", input_ids,
        output_ids, tuple(input_feature_names), tuple(output_names), nodes, connections,
        evaluation_config.to_dict(), resource_limits.to_dict(), dict(metadata or {}),
    )
    # Construction validates schema; evaluator construction validates operational limits.
    from .neat_recurrent_network import RecurrentNeatNetwork
    RecurrentNeatNetwork(artifact)
    return artifact


def save_recurrent_artifact(artifact: RecurrentTopologyArtifact, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return destination


def load_recurrent_artifact(path: str | Path) -> RecurrentTopologyArtifact:
    return RecurrentTopologyArtifact.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
