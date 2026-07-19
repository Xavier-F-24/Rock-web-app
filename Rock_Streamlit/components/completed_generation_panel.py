"""Completion summary and parent-child run comparisons."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st


def comparison_deltas(parent_metrics: dict, child_metrics: dict) -> dict:
    keys = ("best_fitness", "validation_quality", "topology_complexity")
    return {key: float(child_metrics.get(key, 0.0)) - float(parent_metrics.get(key, 0.0)) for key in keys}


def topology_diff(parent: dict, child: dict) -> dict:
    parent_nodes = {row["node_id"]: row for row in parent.get("nodes", ())}; child_nodes = {row["node_id"]: row for row in child.get("nodes", ())}
    parent_edges = {(row["source_id"], row["target_id"]): row for row in parent.get("connections", ())}; child_edges = {(row["source_id"], row["target_id"]): row for row in child.get("connections", ())}
    return {
        "added_nodes": sorted(set(child_nodes) - set(parent_nodes)), "deleted_nodes": sorted(set(parent_nodes) - set(child_nodes)),
        "added_connections": sorted(set(child_edges) - set(parent_edges)), "deleted_connections": sorted(set(parent_edges) - set(child_edges)),
        "changed_activations": sorted(key for key in set(parent_nodes) & set(child_nodes) if parent_nodes[key].get("activation") != child_nodes[key].get("activation")),
        "changed_aggregations": sorted(key for key in set(parent_nodes) & set(child_nodes) if parent_nodes[key].get("aggregation") != child_nodes[key].get("aggregation")),
        "recurrent_edges_added": sorted(key for key in set(child_edges) - set(parent_edges) if child_edges[key].get("recurrent")),
    }


def render_completed_job(status, output_reference: dict | None = None):
    st.success(f"Training completed through generation {status.current_evolutionary_generation}.")
    if output_reference: st.json(output_reference)
