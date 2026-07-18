from __future__ import annotations

import json

from Rock_AI.agents.neural_breeding_agent import NeuralBreedingAgent
from Rock_AI.agents.neat_breeding_agent import NeatBreedingAgent
from Rock_AI.models.neat_network_helper import (
    NeatConnectionDefinition,
    NeatNetworkArtifact,
    NeatNodeDefinition,
    save_neat_artifact,
)
from Rock_AI.policies.neat_pair_ranking_policy import NeatPairRankingPolicy
from Rock_AI.agents.random_breeding_agent import RandomBreedingAgent
from Rock_AI.models.pair_ranker_model import PairRankerModel, PairRankerModelConfig
from Rock_AI.policies.neural_pair_ranking_policy import NeuralPairRankingPolicy
from Rock_AI.representations.encoding_schema_helper import get_default_encoding_schema
from Rock_AI.representations.player_candidate_helper import candidate_arrays
from Rock_AI.representations.player_observation_adapter import PlayerObservationAdapter
from Rock_AI.datasets.pair_ranking_record_helper import FarmerObjectiveProfile
from Rock_GameState.rock_game_state_helper import GameMaster
from Rock_AI.runtime import (
    AgentRuntimeManager,
    RunToCompletionCommand,
    StartSessionCommand,
    StepSessionCommand,
)
from Rock_AI.training.train_pair_ranker import save_pair_ranker_checkpoint


def _decision_signature(session):
    return [
        (row.selected_action, row.resulting_child_ids, row.resulting_child_values)
        for row in session.decision_history
    ]


def test_active_session_save_load_preserves_subsequent_behavior(tmp_path):
    original_manager = AgentRuntimeManager()
    original = original_manager.create_session(
        agent=RandomBreedingAgent(), seed=130, session_id="persistent"
    )
    original_manager.apply(original.session_id, StartSessionCommand())
    original_manager.apply(original.session_id, StepSessionCommand())
    destination = original_manager.save_session(original.session_id, tmp_path / "session.json")

    restored_manager = AgentRuntimeManager()
    restored = restored_manager.load_session(destination)
    assert restored.status == original.status
    assert restored.current_decision_index == original.current_decision_index
    assert len(restored.event_history) == len(original.event_history)

    original_manager.apply(original.session_id, RunToCompletionCommand())
    restored_manager.apply(restored.session_id, RunToCompletionCommand())
    assert _decision_signature(restored) == _decision_signature(original)
    assert restored.environment.state.mutation_count == original.environment.state.mutation_count
    assert len(restored.current_farm_state.rocks) == len(original.current_farm_state.rocks)


def _ranker_checkpoint(path):
    schema = get_default_encoding_schema()
    adapter = PlayerObservationAdapter()
    observation = adapter.build(GameMaster(seed=131), None, FarmerObjectiveProfile())
    candidate = observation.candidates[0]
    arrays = candidate_arrays(candidate)
    config = PairRankerModelConfig(
        len(arrays.parent_a_features), len(arrays.rule_features), len(arrays.farm_features),
        len(arrays.objective_features), len(arrays.metadata_features), 0,
        8, 4, (12,), (10,), 0.0,
    )
    model = PairRankerModel(config)
    save_pair_ranker_checkpoint(
        path,
        model_state_dict=model.state_dict(),
        model_architecture_config=config.to_dict(),
        feature_names={
            "parent": list(candidate.parent_a.feature_names) + [f"{name}.observed_mask" for name in candidate.parent_a.feature_names],
            "rules": list(candidate.public_rules.feature_names) + [f"{name}.observed_mask" for name in candidate.public_rules.feature_names],
            "farm": list(candidate.visible_farm.feature_names) + [f"{name}.observed_mask" for name in candidate.visible_farm.feature_names],
            "objective": list(candidate.objective.feature_names) + [f"{name}.observed_mask" for name in candidate.objective.feature_names],
            "metadata": list(candidate.visible_pair_metadata.feature_names) + [f"{name}.observed_mask" for name in candidate.visible_pair_metadata.feature_names],
            "predictor": [],
        },
        normalization_statistics={"mean": 0.0, "standard_deviation": 1.0},
        encoding_schema_version=schema.version,
        dataset_schema_version=1,
        information_access="player",
        observation_schema_version=observation.schema_version,
        player_feature_normalizer=adapter.normalizer.to_dict(),
    )
    return path


def test_neural_session_serializes_checkpoint_metadata_not_live_model(tmp_path):
    checkpoint = _ranker_checkpoint(tmp_path / "ranker.pt")
    agent = NeuralBreedingAgent(NeuralPairRankingPolicy.load(checkpoint))
    manager = AgentRuntimeManager()
    session = manager.create_session(agent=agent, seed=131, session_id="neural-save")
    destination = manager.save_session(session.session_id, tmp_path / "neural.json")
    text = destination.read_text(encoding="utf-8")
    payload = json.loads(text)
    assert payload["session"]["checkpoint_metadata"]["ranker_checkpoint_path"] == str(checkpoint)
    assert "model_state_dict" not in text
    restored = AgentRuntimeManager().load_session(destination)
    assert isinstance(restored.agent, NeuralBreedingAgent)
    assert restored.checkpoint_metadata["ranker_checkpoint_path"] == str(checkpoint)


def test_neat_session_round_trip_uses_safe_json_artifact(tmp_path):
    artifact_path = tmp_path / "network.json"
    save_neat_artifact(NeatNetworkArtifact(
        artifact_version=1,
        topology_id="runtime-test",
        observation_schema_version=2,
        normalizer_version=1,
        information_access="player",
        input_ids=(-1,),
        output_ids=(0,),
        input_feature_names=("visible.test",),
        nodes=(NeatNodeDefinition(0, 0.0, 1.0, "identity", "sum"),),
        connections=(NeatConnectionDefinition(-1, 0, 1.0, True),),
        metadata={},
    ), artifact_path)
    manager = AgentRuntimeManager()
    session = manager.create_session(
        agent=NeatBreedingAgent(NeatPairRankingPolicy.load(artifact_path)),
        seed=132,
        session_id="neat-save",
    )
    destination = manager.save_session(session.session_id, tmp_path / "neat.json")
    restored = AgentRuntimeManager().load_session(destination)
    assert isinstance(restored.agent, NeatBreedingAgent)
    assert restored.checkpoint_metadata["topology_id"] == "runtime-test"
