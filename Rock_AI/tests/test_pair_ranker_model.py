from __future__ import annotations

from types import SimpleNamespace

import torch

from Rock_AI.datasets.pair_ranking_record_helper import FarmerObjectiveProfile
from Rock_AI.models.pair_ranker_model import (
    PairRankerModel,
    PairRankerModelConfig,
    PairRankingLossConfig,
    group_aware_pair_ranking_loss,
)
from Rock_AI.models.pair_scoring_helper import score_pair_evaluation
from Rock_AI.models.breeding_predictor_model import BreedingPredictorModel, BreedingPredictorModelConfig
from Rock_AI.models.model_output_helper import TargetLayout


def _model():
    return PairRankerModel(PairRankerModelConfig(7, 3, 4, 10, 7, 0, 8, 4, (12,), (10,), 0.0)).eval()


def test_pair_ranker_is_parent_symmetric_and_has_variable_batch_shapes():
    model = _model()
    parent_a = torch.randn(2, 5, 7)
    parent_b = torch.randn(2, 5, 7)
    auxiliaries = (torch.randn(2, 5, 3), torch.randn(2, 5, 4), torch.randn(2, 5, 10), torch.randn(2, 5, 7))
    first = model(parent_a, parent_b, *auxiliaries)
    second = model(parent_b, parent_a, *auxiliaries)
    assert first.shape == (2, 5)
    assert torch.allclose(first, second, atol=1e-6)


def test_group_loss_ignores_padding_and_never_compares_farms():
    scores = torch.tensor([[2.0, 1.0, 100.0], [0.0, 3.0, 0.0]], requires_grad=True)
    utilities = torch.tensor([[2.0, 1.0, -100.0], [0.0, 3.0, 0.0]])
    mask = torch.tensor([[True, True, False], [True, True, False]])
    loss = group_aware_pair_ranking_loss(scores, utilities, mask, PairRankingLossConfig())
    changed_padding = scores.detach().clone()
    changed_padding[:, 2] = -9999
    second = group_aware_pair_ranking_loss(changed_padding, utilities, mask, PairRankingLossConfig())
    assert torch.allclose(loss["total_loss"], second["total_loss"])
    assert loss["pairwise_ranking_loss"] < 1.0
    loss["total_loss"].backward()


def test_objective_weights_change_controlled_utility_ordering():
    estimate = lambda error: SimpleNamespace(standard_error=error)
    expectation = SimpleNamespace(
        expected_child_value=estimate(0.0),
        expected_maximum_child_value=estimate(0.0),
        expected_survivor_count=estimate(0.0),
        per_gene_outcome_distributions={},
    )
    value_pair = SimpleNamespace(
        expectation=expectation,
        explanation_fields={"raw_components": {
            "expected_value": 10.0, "maximum_value": 8.0, "survivor_value": 1.0,
            "genotype_diversity": 0.1, "phenotype_diversity": 0.1,
            "rare_trait": 0.0, "mutation_opportunity": 0.0,
        }},
    )
    diversity_pair = SimpleNamespace(
        expectation=expectation,
        explanation_fields={"raw_components": {
            "expected_value": 2.0, "maximum_value": 2.0, "survivor_value": 1.0,
            "genotype_diversity": 1.0, "phenotype_diversity": 1.0,
            "rare_trait": 0.0, "mutation_opportunity": 0.0,
        }},
    )
    value_objective = FarmerObjectiveProfile(2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    diversity_objective = FarmerObjectiveProfile(0.0, 0.0, 0.0, 10.0, 10.0, 0.0, 0.0)
    assert score_pair_evaluation(value_pair, value_objective).score > score_pair_evaluation(diversity_pair, value_objective).score
    assert score_pair_evaluation(value_pair, diversity_objective).score < score_pair_evaluation(diversity_pair, diversity_objective).score


def test_predictor_parent_encoder_transfer_and_freeze():
    ranker = _model()
    predictor_config = BreedingPredictorModelConfig(
        parent_feature_dimension=7,
        rule_feature_dimension=3,
        context_feature_dimension=0,
        parent_embedding_dimension=8,
        rule_embedding_dimension=4,
        context_embedding_dimension=4,
        encoder_hidden_dimensions=(12,),
        trunk_hidden_dimensions=(10,),
        dropout=0.0,
    )
    predictor = BreedingPredictorModel(
        predictor_config,
        TargetLayout.from_target_names(
            ("expected_raw_clutch_size", "probability_at_least_one_mutation")
        ),
    )
    checkpoint = {
        "encoding_schema_version": 1,
        "model_architecture_config": predictor_config.to_dict(),
        "model_state_dict": predictor.state_dict(),
    }
    ranker.load_parent_encoder_from_predictor(checkpoint, freeze=True)
    assert all(not parameter.requires_grad for parameter in ranker.parent_encoder.parameters())
    for expected, actual in zip(predictor.parent_encoder.parameters(), ranker.parent_encoder.parameters()):
        assert torch.equal(expected, actual)
