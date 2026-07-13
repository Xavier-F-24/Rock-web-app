"""Deterministic CPU/CUDA training loop for the first breeding predictor."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from Rock_AI.evaluation.predictor_evaluator import PredictorEvaluator
from Rock_AI.models.breeding_predictor_model import (
    BreedingPredictorModel,
    BreedingPredictorModelConfig,
)
from Rock_AI.models.loss_helper import PredictorLossConfig, predictor_multitask_loss
from Rock_AI.models.model_output_helper import TargetLayout
from Rock_AI.training.checkpoint_helper import (
    load_predictor_checkpoint,
    save_predictor_checkpoint,
)
from Rock_AI.training.predictor_data_helper import (
    NpzPredictorDataset,
    TargetNormalizer,
    context_swap_pairs,
    make_data_loader,
)
from Rock_AI.training.training_config_helper import PredictorTrainingConfig
from Rock_AI.training.training_metrics_helper import (
    TrainingHistory,
    format_epoch_summary,
)


def set_training_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return device


def _loss_config(config: PredictorTrainingConfig) -> PredictorLossConfig:
    return PredictorLossConfig(
        scalar_loss_weight=config.scalar_loss_weight,
        probability_loss_weight=config.probability_loss_weight,
        phenotype_distribution_weight=config.phenotype_distribution_weight,
        genotype_distribution_weight=config.genotype_distribution_weight,
    )


def _average_components(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        name: float(np.mean([row[name] for row in rows]))
        for name in rows[0]
    } if rows else {"total_loss": 0.0}


def train_breeding_predictor(config: PredictorTrainingConfig) -> dict[str, Any]:
    set_training_seed(config.seed, config.deterministic)
    device = select_device(config.device)
    train_dataset = NpzPredictorDataset(config.dataset_directory, "train")
    validation_dataset = NpzPredictorDataset(config.dataset_directory, "validation")
    layout = TargetLayout.from_target_names(train_dataset.target_names)
    loss_config = _loss_config(config)
    normalizer = TargetNormalizer.fit(
        train_dataset.arrays["targets"],
        train_dataset.arrays["target_mask"],
        layout,
    )
    model_config = BreedingPredictorModelConfig(
        parent_feature_dimension=train_dataset.arrays["parent_a_features"].shape[1],
        rule_feature_dimension=train_dataset.arrays["rule_features"].shape[1],
        context_feature_dimension=train_dataset.arrays["context_features"].shape[1],
        parent_embedding_dimension=config.parent_embedding_dimension,
        rule_embedding_dimension=config.rule_embedding_dimension,
        context_embedding_dimension=config.context_embedding_dimension,
        encoder_hidden_dimensions=config.encoder_hidden_dimensions,
        trunk_hidden_dimensions=config.trunk_hidden_dimensions,
        dropout=config.dropout,
        context_swap_pairs=context_swap_pairs(train_dataset.context_feature_names),
    )
    model = BreedingPredictorModel(model_config, layout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    start_epoch = 1
    best_validation = float("inf")
    if config.resume_checkpoint:
        checkpoint = load_predictor_checkpoint(config.resume_checkpoint, map_location=device)
        if checkpoint["target_names"] != list(layout.target_names):
            raise ValueError("Resume checkpoint target schema does not match dataset")
        model.load_state_dict(checkpoint["model_state_dict"])
        if checkpoint.get("optimizer_state_dict"):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        normalizer = TargetNormalizer.from_dict(checkpoint["normalization_statistics"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation = float(checkpoint["best_validation_metric"])

    train_loader = make_data_loader(
        train_dataset,
        config.batch_size,
        shuffle=True,
        seed=config.seed,
        number_of_workers=config.number_of_workers,
    )
    validation_loader = make_data_loader(
        validation_dataset,
        config.batch_size,
        shuffle=False,
        seed=config.seed,
        number_of_workers=config.number_of_workers,
    )
    evaluator = PredictorEvaluator(model, layout, normalizer, loss_config, device)
    output_directory = config.output_path
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "training_config.json").write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_directory / "normalization.json").write_text(
        json.dumps(normalizer.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    history = TrainingHistory()
    epochs_without_improvement = 0
    feature_names = {
        "parent": train_dataset.parent_feature_names,
        "rules": train_dataset.rule_feature_names,
        "context": train_dataset.context_feature_names,
    }
    manifest = train_dataset.manifest

    def save(path: Path, epoch: int, metrics: dict[str, Any]) -> None:
        save_predictor_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_validation_metric=best_validation,
            model_config=model_config.to_dict(),
            target_names=layout.target_names,
            feature_names=feature_names,
            loss_config=loss_config.to_dict(),
            normalization_statistics=normalizer.to_dict(),
            encoding_schema_version=int(manifest["encoding_schema_version"]),
            dataset_schema_version=int(manifest["dataset_version"]),
            game_rules_version=str(manifest.get("game_rules_version", "unknown")),
            training_seed=config.seed,
            training_config=config.to_dict(),
            metrics=metrics,
        )

    for epoch in range(start_epoch, config.number_of_epochs + 1):
        model.train()
        train_rows: list[dict[str, float]] = []
        for batch in train_loader:
            moved = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(
                moved["parent_a_features"],
                moved["parent_b_features"],
                moved["rule_features"],
                moved["context_features"],
            )
            components = predictor_multitask_loss(
                output,
                moved["targets"],
                moved["target_mask"],
                layout,
                loss_config,
                normalizer.normalize_scalar_tensor(moved["targets"]),
            )
            components["total_loss"].backward()
            if config.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            train_rows.append({name: float(value.detach().item()) for name, value in components.items()})
        train_average = _average_components(train_rows)

        if epoch % config.validation_frequency == 0:
            validation_metrics, validation_losses = evaluator.evaluate_loader(validation_loader)
            validation_losses["aggregate_validation_score"] = float(
                validation_metrics["aggregate_validation_score"]
            )
        else:
            validation_metrics = {}
            validation_losses = {
                "total_loss": float("nan"),
                "scalar_loss": float("nan"),
                "probability_loss": float("nan"),
                "genotype_distribution_loss": float("nan"),
                "phenotype_distribution_loss": float("nan"),
            }
        history.add(epoch, train_average, validation_losses)
        print(format_epoch_summary(epoch, train_average, validation_losses))

        current_validation = validation_losses["total_loss"]
        improved = np.isfinite(current_validation) and current_validation < best_validation
        if improved:
            best_validation = current_validation
            epochs_without_improvement = 0
            save(output_directory / "best.pt", epoch, validation_metrics)
        else:
            epochs_without_improvement += 1
        if epoch % config.checkpoint_frequency == 0 or epoch == config.number_of_epochs:
            save(output_directory / "latest.pt", epoch, validation_metrics)
        history.save(output_directory / "training_history.json")
        if config.early_stopping_patience and epochs_without_improvement >= config.early_stopping_patience:
            print(f"Early stopping after {epoch} epochs.")
            break

    return {
        "model": model,
        "normalizer": normalizer,
        "layout": layout,
        "history": history,
        "best_checkpoint": output_directory / "best.pt",
        "latest_checkpoint": output_directory / "latest.pt",
        "best_validation_loss": best_validation,
        "device": str(device),
    }
