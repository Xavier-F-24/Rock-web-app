"""Complete, portable breeding-predictor checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from Rock_AI.models.model_output_helper import TargetLayout


def save_predictor_checkpoint(
    path: str | Path,
    *,
    model,
    optimizer,
    epoch: int,
    best_validation_metric: float,
    model_config: dict[str, Any],
    target_names: list[str] | tuple[str, ...],
    feature_names: dict[str, list[str] | tuple[str, ...]],
    loss_config: dict[str, Any],
    normalization_statistics: dict[str, Any],
    encoding_schema_version: int,
    dataset_schema_version: int,
    game_rules_version: str,
    training_seed: int,
    training_config: dict[str, Any],
    information_access: str = "player",
    observation_schema_version: int | None = None,
    player_feature_normalizer: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    layout = TargetLayout.from_target_names(target_names)
    payload = {
        "checkpoint_version": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": int(epoch),
        "best_validation_metric": float(best_validation_metric),
        "model_architecture_config": model_config,
        "input_dimensions": {
            "parent": model_config["parent_feature_dimension"],
            "rules": model_config["rule_feature_dimension"],
            "context": model_config["context_feature_dimension"],
        },
        "output_dimensions": {
            "total_targets": len(target_names),
            "scalar": len(layout.scalar_indices),
            "binary_probability": len(layout.binary_probability_indices),
            "genotype_distribution": layout.genotype_output_dimension,
            "phenotype_distribution": layout.phenotype_output_dimension,
            "genotype_groups": len(layout.genotype_groups),
            "phenotype_groups": len(layout.phenotype_groups),
        },
        "target_names": list(target_names),
        "feature_names": {key: list(value) for key, value in feature_names.items()},
        "loss_configuration": loss_config,
        "normalization_statistics": normalization_statistics,
        "encoding_schema_version": int(encoding_schema_version),
        "dataset_schema_version": int(dataset_schema_version),
        "game_rules_version": game_rules_version,
        "training_seed": int(training_seed),
        "training_configuration": training_config,
        "metrics": metrics or {},
        "information_access": information_access,
        "observation_schema_version": int(
            observation_schema_version
            if observation_schema_version is not None
            else encoding_schema_version
        ),
        "player_feature_normalizer": player_feature_normalizer,
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_predictor_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=True)
    required = {
        "model_state_dict",
        "model_architecture_config",
        "target_names",
        "feature_names",
        "normalization_statistics",
        "encoding_schema_version",
        "information_access",
        "observation_schema_version",
        "player_feature_normalizer",
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing required fields: {sorted(missing)}")
    return checkpoint
