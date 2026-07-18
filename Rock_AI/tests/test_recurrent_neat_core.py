from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from Rock_AI.neat.neat_export_helper import load_recurrent_artifact, save_recurrent_artifact
from Rock_AI.neat.neat_recurrent_network import RecurrentEvaluationConfig, RecurrentNeatNetwork
from Rock_AI.neat.neat_state_helper import RecurrentDecisionObservation, TEMPORAL_FEATURE_NAMES
from Rock_AI.neat.neat_topology_helper import (
    RECURRENT_TOPOLOGY_VERSION, RecurrentConnectionGene, RecurrentNodeGene,
    RecurrentTopologyArtifact, TopologyResourceLimits,
)
from Rock_AI.policies.recurrent_neat_pair_ranking_policy import RecurrentNeatPairRankingPolicy
from Rock_AI.representations.information_provenance_helper import FeatureDefinition, InformationProvenance
from Rock_AI.representations.player_observation_helper import (
    PlayerCandidateObservation, PlayerFeatureVector, PlayerObservation,
)


def _artifact(input_count=1, settling_steps=1, *, limits=None):
    limits = limits or TopologyResourceLimits()
    inputs = tuple(-(index + 1) for index in range(input_count))
    nodes = (
        RecurrentNodeGene(0, "output", 0.0, 1.0, "tanh", "sum"),
        RecurrentNodeGene(1, "output", 0.0, 1.0, "tanh", "sum"),
        RecurrentNodeGene(2, "output", 0.0, 1.0, "tanh", "sum"),
    )
    connections = (
        RecurrentConnectionGene(inputs[0], 0, 1.0, True),
        RecurrentConnectionGene(0, 0, 0.8, True, recurrent=True, self_loop=True),
    )
    return RecurrentTopologyArtifact(
        RECURRENT_TOPOLOGY_VERSION, "test-topology", 7, 2, 1, "player",
        inputs, (0, 1, 2), tuple(f"x{index}" for index in range(input_count)),
        ("pair_score", "stop_preference", "confidence"), nodes, connections,
        RecurrentEvaluationConfig(settling_steps=settling_steps).to_dict(), limits.to_dict(), {},
    )


def _vector(name, value):
    definition = FeatureDefinition(name, InformationProvenance.PLAYER_OBSERVATION)
    return PlayerFeatureVector((float(value),), (True,), (definition,))


def _observation(order=("a", "b")):
    candidates = []
    values = {"a": (1, 2, 0.2), "b": (3, 4, 0.8)}
    for key in order:
        left, right, value = values[key]
        candidates.append(PlayerCandidateObservation(
            (left, right), _vector("parent", value), _vector("parent", value),
            _vector("rule", 0.0), _vector("farm", 0.0), _vector("objective", 0.0),
            _vector("metadata", 0.0), key,
        ))
    player = PlayerObservation(2, 1, 0, 3, tuple(candidates), "observation")
    definitions = tuple(FeatureDefinition(name, InformationProvenance.PLAYER_OBSERVATION) for name in TEMPORAL_FEATURE_NAMES)
    temporal = PlayerFeatureVector((0.0,) * len(definitions), (False,) * len(definitions), definitions)
    return RecurrentDecisionObservation(player, temporal)


def test_self_loop_changes_future_decision_state():
    network = RecurrentNeatNetwork(_artifact())
    first = network.activate((1.0,), commit=True)
    second = network.activate((1.0,), first.state, commit=True)
    assert second.outputs[0] != pytest.approx(first.outputs[0])
    reset = network.activate((1.0,), network.initial_state(), commit=True)
    assert reset.outputs == pytest.approx(first.outputs)


def test_synchronous_cycle_is_deterministic():
    artifact = _artifact(settling_steps=3)
    network = RecurrentNeatNetwork(artifact)
    left = network.activate((0.5,), network.initial_state("same"), commit=True)
    right = network.activate((0.5,), network.initial_state("same"), commit=True)
    assert left.outputs == pytest.approx(right.outputs)
    assert left.trace["settling_steps"] == right.trace["settling_steps"]


def test_candidate_enumeration_order_does_not_change_selected_memory():
    observation = _observation()
    input_count = len(observation.temporal_context.model_values()) + 14
    artifact = _artifact(input_count=input_count)
    # Score uses the first symmetric parent feature.
    connections = (replace(artifact.connections[0], source_id=artifact.input_ids[0]), artifact.connections[1])
    artifact = replace(artifact, connections=connections, input_feature_names=tuple(f"f{i}" for i in range(input_count)))
    first = RecurrentNeatPairRankingPolicy(artifact, checkpoint_id="memory")
    second = RecurrentNeatPairRankingPolicy(artifact, checkpoint_id="memory")
    decision_a = first.rank_observation(observation)
    decision_b = second.rank_observation(_observation(("b", "a")))
    assert decision_a.selected_best_pair == decision_b.selected_best_pair
    assert first.state.node_activations == second.state.node_activations
    assert first.state.decision_count == second.state.decision_count == 1


def test_safe_artifact_roundtrip():
    artifact = _artifact()
    path = Path("Rock_AI/tests/_recurrent_roundtrip.json")
    try:
        save_recurrent_artifact(artifact, path)
        loaded = load_recurrent_artifact(path)
        assert loaded == artifact
        assert RecurrentNeatNetwork(loaded).activate((0.25,)).outputs == pytest.approx(
            RecurrentNeatNetwork(artifact).activate((0.25,)).outputs
        )
    finally:
        path.unlink(missing_ok=True)


def test_resource_limits_are_enforced():
    artifact = _artifact(limits=TopologyResourceLimits(max_enabled_connections=1))
    with pytest.raises(ValueError, match="enabled-connection"):
        RecurrentNeatNetwork(artifact)


def test_recurrent_policy_rejects_plain_player_observation():
    observation = _observation()
    artifact = _artifact(input_count=len(observation.temporal_context.model_values()) + 14)
    policy = RecurrentNeatPairRankingPolicy(artifact, checkpoint_id="memory")
    with pytest.raises(TypeError, match="RecurrentDecisionObservation"):
        policy.rank_observation(observation.player_observation)
