"""Plotly network and activation display for safe model traces."""

from __future__ import annotations

import math

import plotly.graph_objects as go
import streamlit as st

from Rock_AI.visualization.network_visualization_helper import model_trace_graph


def _positions(nodes):
    count = max(1, len(nodes))
    return {
        node["id"]: (
            math.cos(2 * math.pi * index / count),
            math.sin(2 * math.pi * index / count),
        )
        for index, node in enumerate(nodes)
    }


def render_network_trace(trace, *, maximum_edges: int = 30) -> None:
    graph = model_trace_graph(trace, maximum_edges)
    if not graph["nodes"]:
        st.info("Run one agent decision to capture a network activation trace.")
        return
    positions = _positions(graph["nodes"])
    figure = go.Figure()
    for edge in graph["edges"]:
        source, target = str(edge["source_id"]), str(edge["target_id"])
        if source not in positions or target not in positions:
            continue
        signal = float(edge.get("local_signal", 0.0))
        figure.add_trace(go.Scatter(
            x=[positions[source][0], positions[target][0]],
            y=[positions[source][1], positions[target][1]],
            mode="lines",
            line={"color": "#2b6cb0" if signal >= 0 else "#c53030", "width": 1 + min(6, abs(signal))},
            hovertext=f"Local edge signal: {signal:.4f}",
            hoverinfo="text",
            showlegend=False,
        ))
    activations = [node["activation"] for node in graph["nodes"]]
    figure.add_trace(go.Scatter(
        x=[positions[node["id"]][0] for node in graph["nodes"]],
        y=[positions[node["id"]][1] for node in graph["nodes"]],
        mode="markers",
        marker={"size": 13, "color": activations, "colorscale": "RdBu", "cmid": 0, "showscale": True},
        text=[node["label"] for node in graph["nodes"]],
        customdata=activations,
        hovertemplate="%{text}<br>Activation %{customdata:.4f}<extra></extra>",
        showlegend=False,
    ))
    figure.update_layout(height=600, margin=dict(l=10, r=10, t=20, b=10), xaxis_visible=False, yaxis_visible=False)
    st.plotly_chart(figure, width="stretch")
    st.caption("Connection color and width show signed local edge signals, not causal explanations.")
    metadata = trace.get("metadata", {}) if trace else {}
    settling = metadata.get("settling_steps", ())
    if settling:
        step = st.slider("Recurrent settling step", 0, len(settling) - 1, len(settling) - 1, key="ai_obs_settling_step")
        st.caption("Synchronous update: every node in this step read the same previous-state snapshot.")
        st.json({"memory_before": metadata.get("memory_before"), "selected_step": settling[step], "memory_after": metadata.get("memory_after")})


def render_recurrent_topology(artifact: dict, *, maximum_edges: int = 80) -> None:
    """Render a safe exported topology even when no showcase decision exists."""
    nodes = list(artifact.get("nodes", ()))
    nodes.extend({"node_id": node_id, "node_type": "input", "activation": "clamped", "aggregation": "input"} for node_id in artifact.get("input_ids", ()))
    connections = [row for row in artifact.get("connections", ()) if row.get("enabled", True)]
    connections.sort(key=lambda row: abs(float(row.get("weight", 0.0))), reverse=True)
    connections = connections[:maximum_edges]
    positions = _positions([{"id": str(row["node_id"])} for row in nodes])
    figure = go.Figure()
    for edge in connections:
        source, target = str(edge["source_id"]), str(edge["target_id"])
        if source not in positions or target not in positions:
            continue
        weight = float(edge["weight"])
        figure.add_trace(go.Scatter(
            x=[positions[source][0], positions[target][0]], y=[positions[source][1], positions[target][1]],
            mode="lines", showlegend=False,
            line={"color": "#2b6cb0" if weight >= 0 else "#c53030", "width": 1 + min(5, abs(weight)), "dash": "dot" if edge.get("recurrent") else "solid"},
            hovertext=f"weight {weight:.4f}; recurrent={bool(edge.get('recurrent'))}", hoverinfo="text",
        ))
    figure.add_trace(go.Scatter(
        x=[positions[str(row["node_id"])][0] for row in nodes], y=[positions[str(row["node_id"])][1] for row in nodes],
        mode="markers+text", text=[row.get("node_type", "node") for row in nodes], textposition="top center",
        marker={"size": 15, "color": ["#d69e2e" if row.get("node_type") == "output" else "#2b6cb0" if row.get("node_type") == "input" else "#718096" for row in nodes]},
        hovertext=[f"node {row['node_id']}<br>{row['activation']} / {row['aggregation']}" for row in nodes], hoverinfo="text", showlegend=False,
    ))
    figure.update_layout(height=600, margin=dict(l=10, r=10, t=20, b=10), xaxis_visible=False, yaxis_visible=False)
    st.plotly_chart(figure, width="stretch")
    st.caption("Dotted connections are recurrent. Self-loops are listed in the topology details below.")
