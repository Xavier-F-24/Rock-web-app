"""Authoritative synchronous evaluator for exported recurrent NEAT topologies."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import neat

from .neat_state_helper import RecurrentAgentState
from .neat_topology_helper import RecurrentTopologyArtifact, TopologyResourceLimits


@dataclass(frozen=True)
class RecurrentEvaluationConfig:
    settling_steps: int = 3
    state_decay: float = 0.0
    activation_clip: float = 30.0

    def __post_init__(self) -> None:
        if self.settling_steps <= 0:
            raise ValueError("settling_steps must be positive")
        if not 0.0 <= self.state_decay <= 1.0:
            raise ValueError("state_decay must be in [0, 1]")
        if self.activation_clip <= 0:
            raise ValueError("activation_clip must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RecurrentActivationResult:
    outputs: tuple[float, ...]
    state: RecurrentAgentState
    trace: dict[str, Any]


class RecurrentNumericalError(RuntimeError):
    pass


class RecurrentNeatNetwork:
    """Evaluate every non-input node from the same prior-state snapshot."""

    def __init__(self, artifact: RecurrentTopologyArtifact):
        self.artifact = artifact
        self.config = RecurrentEvaluationConfig(**artifact.evaluation_config)
        self.limits = TopologyResourceLimits(**artifact.resource_limits)
        if self.config.settling_steps > self.limits.max_recurrent_settling_steps:
            raise ValueError("Topology settling steps exceed the resource limit")
        if artifact.hidden_node_count > self.limits.max_hidden_nodes:
            raise ValueError("Topology exceeds the hidden-node limit")
        if artifact.enabled_connection_count > self.limits.max_enabled_connections:
            raise ValueError("Topology exceeds the enabled-connection limit")
        if len(artifact.nodes) + len(artifact.connections) > self.limits.max_total_genes:
            raise ValueError("Topology exceeds the total-gene limit")
        self._activations = neat.activations.ActivationFunctionSet()
        self._aggregations = neat.aggregations.AggregationFunctionSet()
        self._nodes = {node.node_id: node for node in artifact.nodes}
        self._state_node_ids = tuple(sorted(self._nodes))
        self._incoming = {node_id: [] for node_id in self._state_node_ids}
        for connection in artifact.connections:
            if connection.enabled and connection.target_id in self._incoming:
                self._incoming[connection.target_id].append(connection)

    def initial_state(self, episode_id: str = "episode") -> RecurrentAgentState:
        return RecurrentAgentState(
            topology_id=self.artifact.topology_id,
            genome_id=self.artifact.genome_id,
            episode_id=episode_id,
            node_activations=tuple((node_id, 0.0) for node_id in self._state_node_ids),
            previous_outputs=tuple(0.0 for _ in self.artifact.output_ids),
        )

    def activate(
        self,
        inputs: Sequence[float],
        state: RecurrentAgentState | None = None,
        *,
        commit: bool = True,
        episode_id: str = "episode",
        temporal_context: Sequence[float] = (),
    ) -> RecurrentActivationResult:
        if len(inputs) != len(self.artifact.input_ids):
            raise ValueError("Recurrent input dimension is incompatible with the artifact")
        if not all(math.isfinite(float(value)) for value in inputs):
            raise RecurrentNumericalError("Recurrent inputs contain NaN or infinity")
        initial = state or self.initial_state(episode_id)
        initial.validate_for(self.artifact.topology_id, self.artifact.genome_id)
        prior = {node_id: 0.0 for node_id in self._state_node_ids}
        prior.update(initial.activation_map())
        input_values = {key: float(value) for key, value in zip(self.artifact.input_ids, inputs)}
        settling_trace: list[dict[str, Any]] = []
        final_signals: list[dict[str, Any]] = []
        for step in range(self.config.settling_steps):
            current: dict[int, float] = {}
            signals: list[dict[str, Any]] = []
            for node_id in self._state_node_ids:
                node = self._nodes[node_id]
                weighted: list[float] = []
                for connection in self._incoming.get(node_id, ()):
                    source_value = input_values.get(connection.source_id, prior.get(connection.source_id, 0.0))
                    signal = source_value * connection.weight
                    weighted.append(signal)
                    signals.append({
                        "source_id": connection.source_id,
                        "target_id": connection.target_id,
                        "weight": connection.weight,
                        "source_activation": source_value,
                        "local_signal": signal,
                        "recurrent": connection.recurrent,
                        "self_loop": connection.self_loop,
                    })
                aggregate = self._aggregations.get(node.aggregation)(weighted)
                activated = self._activations.get(node.activation)(node.bias + node.response * aggregate)
                value = (1.0 - self.config.state_decay) * activated + self.config.state_decay * prior[node_id]
                value = max(-self.config.activation_clip, min(self.config.activation_clip, float(value)))
                if not math.isfinite(value):
                    raise RecurrentNumericalError(f"Node {node_id} produced a non-finite activation")
                current[node_id] = value
            prior = current
            final_signals = signals
            settling_trace.append({"step": step, "node_activations": {str(k): v for k, v in current.items()}})
        outputs = tuple(prior.get(node_id, 0.0) for node_id in self.artifact.output_ids)
        next_state = RecurrentAgentState(
            topology_id=self.artifact.topology_id,
            genome_id=self.artifact.genome_id,
            episode_id=initial.episode_id,
            decision_count=initial.decision_count + (1 if commit else 0),
            node_activations=tuple(sorted(prior.items())),
            previous_outputs=outputs,
            temporal_context=tuple(map(float, temporal_context)),
        )
        return RecurrentActivationResult(outputs, next_state, {
            "state_before": initial.to_dict(),
            "settling_steps": settling_trace,
            "state_after": next_state.to_dict(),
            "node_activations": {str(key): value for key, value in prior.items()},
            "connection_signals": final_signals,
            "committed": bool(commit),
        })
