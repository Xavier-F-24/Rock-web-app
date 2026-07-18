"""Pure conversion of safe model traces and NEAT artifacts to graph records."""

from __future__ import annotations

from typing import Any


def model_trace_graph(trace: dict[str, Any] | None, maximum_edges: int = 30) -> dict[str, list[dict[str, Any]]]:
    if not trace:
        return {"nodes": [], "edges": []}
    activations = trace.get("node_activations", {})
    edges = sorted(
        trace.get("connection_signals", ()),
        key=lambda row: abs(float(row.get("local_signal", 0.0))),
        reverse=True,
    )[:maximum_edges]
    node_ids: set[str] = set()
    for edge in edges:
        node_ids.add(str(edge["source_id"]))
        node_ids.add(str(edge["target_id"]))
    if not node_ids:
        node_ids.update(
            key for key, _ in sorted(
                activations.items(), key=lambda item: abs(float(item[1])), reverse=True
            )[:30]
        )
    nodes = [
        {
            "id": node_id,
            "activation": float(activations.get(node_id, 0.0)),
            "label": node_id,
        }
        for node_id in sorted(node_ids)
    ]
    return {"nodes": nodes, "edges": edges}


def training_metrics_rows(lines: list[str]) -> list[dict[str, Any]]:
    import json

    rows = [json.loads(line) for line in lines if line.strip()]
    return sorted(rows, key=lambda row: int(row["generation"]))
