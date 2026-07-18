from __future__ import annotations

import numpy as np

from Rock_AI.datasets.pair_ranking_record_helper import FarmerObjectiveProfile, PairRankingCandidate, PairRankingGroup
from Rock_AI.datasets.pair_ranking_storage_helper import save_pair_ranking_dataset
from Rock_AI.training.train_pair_ranker import evaluate_pair_ranker_checkpoint, train_pair_ranker
from Rock_AI.training.training_config_helper import PairRankerTrainingConfig


def _group(group_index: int) -> PairRankingGroup:
    candidates = []
    for candidate_index in range(3):
        signal = float(candidate_index + group_index * 0.1)
        candidates.append(PairRankingCandidate(
            parent_ids=(group_index * 10 + candidate_index, group_index * 10 + candidate_index + 1000),
            parent_a_features=np.full(6, signal, np.float32),
            parent_b_features=np.full(6, signal / 2, np.float32),
            rule_features=np.zeros(2, np.float32), farm_features=np.zeros(2, np.float32),
            objective_features=np.ones(10, np.float32), metadata_features=np.asarray([signal], np.float32),
            predictor_features=np.zeros(0, np.float32), utility_components=np.zeros(9, np.float32),
            utility_score=signal * 4, uncertainty=0.0, rank=3 - candidate_index,
            best_pair=candidate_index == 2,
        ))
    return PairRankingGroup(f"g{group_index}", f"l{group_index}", tuple(candidates), group_index, 1, FarmerObjectiveProfile(), {}, tuple(range(6)))


def test_smoke_training_checkpoint_and_evaluation(tmp_path):
    groups = [_group(index) for index in range(12)]
    splits = {"train": groups[:8], "validation": groups[8:10], "test": groups[10:]}
    manifest = {
        "encoding_schema_version": 2,
        "information_access": "player",
        "observation_schema_version": 2,
        "player_feature_normalizer": {"version": 1},
        "game_rules_version": "test",
        "feature_names": {"parent": [f"p{i}" for i in range(6)], "rules": ["r0", "r1"], "farm": ["f0", "f1"], "objective": [f"o{i}" for i in range(10)], "metadata": ["m0"], "predictor": []},
        "utility_component_names": [f"u{i}" for i in range(9)],
        "dimensions": {"parent": 6, "rules": 2, "farm": 2, "objective": 10, "metadata": 1, "predictor": 0},
    }
    dataset = tmp_path / "dataset"
    save_pair_ranking_dataset(dataset, splits, manifest)
    result = train_pair_ranker(PairRankerTrainingConfig(
        dataset_path=str(dataset), output_directory=str(tmp_path / "run"), number_of_epochs=8,
        batch_size=4, learning_rate=0.01, dropout=0.0, early_stopping_patience=0,
        parent_embedding_dimension=8, auxiliary_embedding_dimension=4,
        encoder_hidden_dimensions=(12,), trunk_hidden_dimensions=(12,), device="cpu",
    ))
    assert result["best_checkpoint"].exists()
    assert result["history"][-1]["training_loss"] < result["history"][0]["training_loss"]
    evaluation = evaluate_pair_ranker_checkpoint(dataset, result["best_checkpoint"], "test")
    assert evaluation["metrics"]["mean_utility_regret"] >= 0
