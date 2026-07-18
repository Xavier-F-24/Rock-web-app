from __future__ import annotations

from pathlib import Path

from Rock_AI.agents.recurrent_neat_breeding_agent import RecurrentNeatBreedingAgent
from Rock_AI.environments.breeding_campaign_environment import BreedingCampaignEnvironment
from Rock_AI.neat.neat_export_helper import save_recurrent_artifact
from Rock_AI.neat.neat_recurrent_network import RecurrentEvaluationConfig
from Rock_AI.neat.neat_state_helper import TEMPORAL_FEATURE_NAMES
from Rock_AI.neat.neat_topology_helper import (
    RECURRENT_TOPOLOGY_VERSION, RecurrentConnectionGene, RecurrentNodeGene,
    RecurrentTopologyArtifact, TopologyResourceLimits,
)
from Rock_AI.policies.recurrent_neat_pair_ranking_policy import RecurrentNeatPairRankingPolicy
from Rock_AI.representations.player_candidate_helper import neat_symmetric_candidate_vector
from Rock_AI.representations.player_observation_adapter import PlayerObservationAdapter
from Rock_AI.runtime import AgentRuntimeManager, StartSessionCommand, StepSessionCommand


def _runtime_artifact(path: Path):
    environment = BreedingCampaignEnvironment(seed=445)
    campaign = environment.reset(445)
    recurrent = PlayerObservationAdapter().build_recurrent(campaign)
    width = len(neat_symmetric_candidate_vector(recurrent.player_observation.candidates[0])) + len(recurrent.temporal_context.model_values())
    input_ids = tuple(-(index + 1) for index in range(width))
    nodes = tuple(RecurrentNodeGene(index, "output", 0.0, 1.0, "tanh", "sum") for index in range(3))
    connections = (
        RecurrentConnectionGene(input_ids[0], 0, 1.0, True),
        RecurrentConnectionGene(0, 0, 0.5, True, recurrent=True, self_loop=True),
    )
    artifact = RecurrentTopologyArtifact(
        RECURRENT_TOPOLOGY_VERSION, "runtime-recurrent", 88, 2, 1, "player",
        input_ids, (0, 1, 2), tuple(f"feature.{index}" for index in range(width)),
        ("pair_score", "stop_preference", "confidence"), nodes, connections,
        RecurrentEvaluationConfig(2).to_dict(), TopologyResourceLimits().to_dict(), {},
    )
    save_recurrent_artifact(artifact, path)


def test_runtime_save_load_preserves_recurrent_memory():
    artifact_path = Path("Rock_AI/tests/_runtime_recurrent_network.json")
    session_path = Path("Rock_AI/tests/_runtime_recurrent_session.json")
    try:
        _runtime_artifact(artifact_path)
        policy = RecurrentNeatPairRankingPolicy.load(artifact_path)
        manager = AgentRuntimeManager()
        session = manager.create_session(
            agent=RecurrentNeatBreedingAgent(policy), seed=445,
            session_id="recurrent-runtime",
        )
        manager.apply(session.session_id, StartSessionCommand())
        manager.apply(session.session_id, StepSessionCommand())
        state_before = policy.export_state()
        manager.save_session(session.session_id, session_path)
        restored = AgentRuntimeManager().load_session(session_path)
        assert isinstance(restored.agent, RecurrentNeatBreedingAgent)
        assert restored.agent.policy.export_state() == state_before
        assert restored.checkpoint_metadata["recurrent_neat_network_artifact_path"] == str(artifact_path)
    finally:
        artifact_path.unlink(missing_ok=True)
        session_path.unlink(missing_ok=True)
