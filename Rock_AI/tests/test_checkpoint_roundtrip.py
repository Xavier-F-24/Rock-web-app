from __future__ import annotations

import torch

from Rock_AI.models.breeding_predictor_model import (
    BreedingPredictorModel,
    BreedingPredictorModelConfig,
)
from Rock_AI.models.model_output_helper import TargetLayout
from Rock_AI.training.checkpoint_helper import (
    load_predictor_checkpoint,
    save_predictor_checkpoint,
)


def test_checkpoint_round_trip_preserves_predictions_and_metadata(tmp_path):
    layout = TargetLayout.from_target_names(
        ("expected_raw_clutch_size", "probability_at_least_one_mutation")
    )
    config = BreedingPredictorModelConfig(
        parent_feature_dimension=3,
        rule_feature_dimension=2,
        context_feature_dimension=0,
        parent_embedding_dimension=4,
        rule_embedding_dimension=2,
        encoder_hidden_dimensions=(),
        trunk_hidden_dimensions=(5,),
        dropout=0.0,
    )
    model = BreedingPredictorModel(config, layout).eval()
    optimizer = torch.optim.Adam(model.parameters())
    inputs = (torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 2))
    before = model(*inputs).scalar_normalized.detach().clone()
    path = save_predictor_checkpoint(
        tmp_path / "model.pt",
        model=model,
        optimizer=optimizer,
        epoch=3,
        best_validation_metric=0.5,
        model_config=config.to_dict(),
        target_names=layout.target_names,
        feature_names={"parent": ["a", "b", "c"], "rules": ["r1", "r2"], "context": []},
        loss_config={},
        normalization_statistics={"scalar_indices": [0], "means": [1.0], "standard_deviations": [2.0]},
        encoding_schema_version=1,
        dataset_schema_version=1,
        game_rules_version="test-rules",
        training_seed=99,
        training_config={},
    )
    checkpoint = load_predictor_checkpoint(path)
    restored = BreedingPredictorModel(config, layout).eval()
    restored.load_state_dict(checkpoint["model_state_dict"])
    after = restored(*inputs).scalar_normalized.detach()

    assert torch.allclose(before, after)
    assert checkpoint["epoch"] == 3
    assert checkpoint["game_rules_version"] == "test-rules"
    assert checkpoint["optimizer_state_dict"] is not None
