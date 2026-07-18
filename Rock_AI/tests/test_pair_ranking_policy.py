from __future__ import annotations

import torch
import pytest

import Rock_Genetics.rock_genetic_helper as genetics
from Rock_AI.datasets.pair_ranking_dataset_generator import PairRankingDatasetGenerator, SyntheticFarm
from Rock_AI.datasets.pair_ranking_record_helper import FarmerObjectiveProfile
from Rock_AI.models.pair_ranker_model import PairRankerModel, PairRankerModelConfig
from Rock_AI.policies.neural_pair_ranking_policy import NeuralPairRankingPolicy
from Rock_AI.representations.encoding_schema_helper import get_default_encoding_schema
from Rock_AI.representations.player_candidate_helper import candidate_arrays
from Rock_AI.representations.player_observation_adapter import PlayerObservationAdapter
from Rock_AI.training.train_pair_ranker import save_pair_ranker_checkpoint
from Rock_AI.training.training_config_helper import PairRankingDataConfig


def _observation(farm):
    return PlayerObservationAdapter().build(farm, None, FarmerObjectiveProfile())


def _checkpoint(tmp_path, farm):
    schema = get_default_encoding_schema()
    observation = _observation(farm)
    sample = candidate_arrays(observation.candidates[0])
    config = PairRankerModelConfig(
        len(sample.parent_a_features), len(sample.rule_features), len(sample.farm_features),
        len(sample.objective_features), len(sample.metadata_features), 0,
        8, 4, (12,), (10,), 0.0,
    )
    model = PairRankerModel(config)
    path = tmp_path / "ranker.pt"
    save_pair_ranker_checkpoint(
        path,
        model_state_dict=model.state_dict(),
        model_architecture_config=config.to_dict(),
        feature_names={
            "parent": list(observation.candidates[0].parent_a.feature_names) + [f"{name}.observed_mask" for name in observation.candidates[0].parent_a.feature_names],
            "rules": list(observation.candidates[0].public_rules.feature_names) + [f"{name}.observed_mask" for name in observation.candidates[0].public_rules.feature_names],
            "farm": list(observation.candidates[0].visible_farm.feature_names) + [f"{name}.observed_mask" for name in observation.candidates[0].visible_farm.feature_names],
            "objective": list(observation.candidates[0].objective.feature_names) + [f"{name}.observed_mask" for name in observation.candidates[0].objective.feature_names],
            "metadata": list(observation.candidates[0].visible_pair_metadata.feature_names) + [f"{name}.observed_mask" for name in observation.candidates[0].visible_pair_metadata.feature_names],
            "predictor": [],
        },
        normalization_statistics={"mean": 0.0, "standard_deviation": 1.0},
        encoding_schema_version=schema.version,
        information_access="player",
        observation_schema_version=observation.schema_version,
        player_feature_normalizer=PlayerObservationAdapter().normalizer.to_dict(),
    )
    return path


def _rocks(count=4):
    generator = PairRankingDatasetGenerator(PairRankingDataConfig(number_of_farms=1, trials_per_pair=1, minimum_rocks=4, maximum_rocks=4))
    farm = generator.create_procedural_farm(0)
    return list(farm.rocks.values())[:count]


def test_policy_returns_only_legal_original_ids_deterministically(tmp_path):
    rocks = _rocks()
    farm = SyntheticFarm(rocks, "policy-farm")
    policy = NeuralPairRankingPolicy.load(_checkpoint(tmp_path, farm))
    first = policy.rank_observation(_observation(farm))
    second = policy.rank_observation(_observation(farm))
    assert first == second
    assert first.selected_best_pair is not None
    ids = set(farm.rocks)
    assert all(set(pair.parent_ids) <= ids for pair in first.ranked_pairs)
    validator = __import__("Rock_Breeding.rock_breeding_helper", fromlist=["BreedingMaster"]).BreedingMaster()
    assert all(validator.validate_breeding_pair(farm.get_rock(a), farm.get_rock(b), game=farm)["valid"] for a, b in (row.parent_ids for row in first.ranked_pairs))
    with pytest.raises(TypeError):
        policy.rank_legal_pairs(rocks, None)


def test_policy_handles_one_and_zero_legal_pairs(tmp_path):
    rocks = _rocks(2)
    one_farm = SyntheticFarm(rocks, "one")
    policy = NeuralPairRankingPolicy.load(_checkpoint(tmp_path, one_farm))
    one = policy.rank_observation(_observation(one_farm))
    assert len(one.ranked_pairs) == 1
    rocks[1].sex = genetics.Sex.MALE
    zero = policy.rank_observation(_observation(SyntheticFarm(rocks, "zero")))
    assert zero.selected_best_pair is None
    assert zero.no_action_reason


def test_schema_mismatch_fails_clearly(tmp_path):
    farm = SyntheticFarm(_rocks(), "mismatch")
    path = _checkpoint(tmp_path, farm)
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["observation_schema_version"] = 999
    torch.save(checkpoint, path)
    try:
        NeuralPairRankingPolicy.load(path)
    except ValueError as error:
        assert "schema" in str(error).lower()
    else:
        raise AssertionError("Expected schema mismatch")
