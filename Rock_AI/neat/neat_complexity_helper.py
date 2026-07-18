"""Complexity accounting shared by mutation, training, and export."""

from __future__ import annotations

from .neat_topology_helper import TopologyResourceLimits


def genome_complexity(genome) -> dict[str, int]:
    enabled = sum(bool(connection.enabled) for connection in genome.connections.values())
    output_count = getattr(genome, "_rock_output_count", 3)
    return {
        "hidden_nodes": max(0, len(genome.nodes) - output_count),
        "enabled_connections": enabled,
        "total_genes": len(genome.nodes) + len(genome.connections),
    }


def complexity_within_limits(genome, limits: TopologyResourceLimits) -> bool:
    values = genome_complexity(genome)
    return (
        values["hidden_nodes"] <= limits.max_hidden_nodes
        and values["enabled_connections"] <= limits.max_enabled_connections
        and values["total_genes"] <= limits.max_total_genes
    )
