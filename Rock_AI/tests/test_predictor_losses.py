from __future__ import annotations

import numpy as np
import torch

from Rock_AI.models.breeding_predictor_model import (
    BreedingPredictorModel,
    BreedingPredictorModelConfig,
)
from Rock_AI.models.loss_helper import PredictorLossConfig, predictor_multitask_loss
from Rock_AI.models.model_output_helper import TargetLayout
from Rock_AI.training.predictor_data_helper import TargetNormalizer


def _layout():
    return TargetLayout.from_target_names(
        (
            "expected_raw_clutch_size",
            "expected_survivor_count",
            "probability_at_least_one_mutation",
            "gene.eyes.allele_pair.0|0",
            "gene.eyes.allele_pair.1|1",
            "phenotype.eyes=n/a",
            "phenotype.eyes=eye",
        )
    )


def test_target_normalization_round_trip_uses_only_scalar_columns():
    layout = _layout()
    targets = np.asarray([[1, 2, 0.1, 1, 0, 1, 0], [3, 4, 0.9, 0, 1, 0, 1]], dtype=np.float32)
    mask = np.ones_like(targets, dtype=bool)
    normalizer = TargetNormalizer.fit(targets, mask, layout)
    scalar = targets[:, list(layout.scalar_indices)]

    assert np.allclose(normalizer.denormalize_array(normalizer.normalize_array(scalar)), scalar)
    assert normalizer.scalar_indices == layout.scalar_indices


def test_masks_exclude_unavailable_targets_from_loss():
    layout = _layout()
    model = BreedingPredictorModel(
        BreedingPredictorModelConfig(4, 2, 0, 4, 2, 2, (), (6,), 0.0),
        layout,
    )
    output = model(torch.ones(2, 4), torch.ones(2, 4), torch.ones(2, 2))
    targets = torch.tensor(
        [[1.0, 2.0, 0.1, 1.0, 0.0, 1.0, 0.0], [3.0, 4.0, 0.9, 0.0, 1.0, 0.0, 1.0]]
    )
    mask = torch.ones_like(targets, dtype=torch.bool)
    mask[0, 0] = False
    mask[0, 2] = False
    mask[0, 3] = False
    normalizer = TargetNormalizer.fit(targets.numpy(), mask.numpy(), layout)
    normalized = normalizer.normalize_scalar_tensor(targets)
    first = predictor_multitask_loss(
        output, targets, mask, layout, PredictorLossConfig(), normalized
    )
    changed = targets.clone()
    changed[0, 0] = 100000.0
    changed[0, 2] = 1.0
    changed[0, 3] = 100000.0
    changed_normalized = normalizer.normalize_scalar_tensor(changed)
    second = predictor_multitask_loss(
        output, changed, mask, layout, PredictorLossConfig(), changed_normalized
    )

    assert torch.allclose(first["total_loss"], second["total_loss"])
    assert set(first) == {
        "total_loss",
        "scalar_loss",
        "probability_loss",
        "genotype_distribution_loss",
        "phenotype_distribution_loss",
    }
