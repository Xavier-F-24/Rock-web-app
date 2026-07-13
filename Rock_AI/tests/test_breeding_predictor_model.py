from __future__ import annotations

import torch

from Rock_AI.models.breeding_predictor_model import (
    BreedingPredictorModel,
    BreedingPredictorModelConfig,
)
from Rock_AI.models.model_output_helper import TargetLayout


def _layout():
    return TargetLayout.from_target_names(
        (
            "expected_raw_clutch_size",
            "probability_at_least_one_mutation",
            "gene.eyes.allele_pair.0|0",
            "gene.eyes.allele_pair.0|1",
            "gene.eyes.allele_pair.1|0",
            "gene.eyes.allele_pair.1|1",
            "phenotype.eyes=n/a",
            "phenotype.eyes=eye",
            "phenotype.eyes=double eye",
        )
    )


def _model():
    config = BreedingPredictorModelConfig(
        parent_feature_dimension=7,
        rule_feature_dimension=3,
        context_feature_dimension=5,
        parent_embedding_dimension=8,
        rule_embedding_dimension=4,
        context_embedding_dimension=4,
        encoder_hidden_dimensions=(12,),
        trunk_hidden_dimensions=(10,),
        dropout=0.0,
        context_swap_pairs=((0, 1), (3, 4)),
    )
    return BreedingPredictorModel(config, _layout()).eval()


def test_forward_shapes_and_valid_probabilities():
    model = _model()
    output = model(
        torch.randn(4, 7),
        torch.randn(4, 7),
        torch.randn(4, 3),
        torch.randn(4, 5),
    )

    assert output.scalar_normalized.shape == (4, 1)
    assert output.binary_probabilities.shape == (4, 1)
    assert torch.all((output.binary_probabilities >= 0) & (output.binary_probabilities <= 1))
    assert torch.allclose(output.genotype_probabilities[0].sum(dim=1), torch.ones(4))
    assert torch.allclose(output.phenotype_probabilities[0].sum(dim=1), torch.ones(4))


def test_parent_order_and_context_order_are_explicitly_symmetric():
    model = _model()
    parent_a = torch.randn(3, 7)
    parent_b = torch.randn(3, 7)
    rules = torch.randn(3, 3)
    context = torch.randn(3, 5)
    swapped_context = context.clone()
    swapped_context[:, [0, 1]] = swapped_context[:, [1, 0]]
    swapped_context[:, [3, 4]] = swapped_context[:, [4, 3]]

    first = model(parent_a, parent_b, rules, context)
    second = model(parent_b, parent_a, rules, swapped_context)

    assert torch.allclose(first.scalar_normalized, second.scalar_normalized, atol=1e-6)
    assert torch.allclose(first.binary_probabilities, second.binary_probabilities, atol=1e-6)
    assert torch.allclose(first.genotype_probabilities[0], second.genotype_probabilities[0], atol=1e-6)
