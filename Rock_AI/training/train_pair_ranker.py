"""Group-batched training loop for the supervised breeding-pair ranker."""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from Rock_AI.datasets.pair_ranking_storage_helper import load_pair_ranking_split
from Rock_AI.evaluation.pair_ranker_metrics import calculate_pair_ranker_metrics
from Rock_AI.models.pair_ranker_model import (
    PairRankerModel,
    PairRankerModelConfig,
    PairRankingLossConfig,
    group_aware_pair_ranking_loss,
)
from Rock_AI.training.training_config_helper import PairRankerTrainingConfig


FEATURE_KEYS = (
    "parent_a_features",
    "parent_b_features",
    "rule_features",
    "farm_features",
    "objective_features",
    "metadata_features",
    "predictor_features",
)


class PairRankingNpzDataset(Dataset):
    def __init__(self, directory: str | Path, split: str):
        self.arrays, self.group_metadata, self.manifest = load_pair_ranking_split(directory, split)
        self.offsets = self.arrays["group_offsets"]

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        result = {key: self.arrays[key][start:end] for key in FEATURE_KEYS}
        result["utility_scores"] = self.arrays["utility_scores"][start:end]
        return result


def collate_ranking_groups(rows: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
    maximum = max(len(row["utility_scores"]) for row in rows)
    batch: dict[str, torch.Tensor] = {}
    for key in FEATURE_KEYS:
        width = rows[0][key].shape[1]
        values = np.zeros((len(rows), maximum, width), dtype=np.float32)
        for index, row in enumerate(rows):
            values[index, : len(row[key])] = row[key]
        batch[key] = torch.from_numpy(values)
    utilities = np.zeros((len(rows), maximum), dtype=np.float32)
    mask = np.zeros((len(rows), maximum), dtype=np.bool_)
    for index, row in enumerate(rows):
        count = len(row["utility_scores"])
        utilities[index, :count] = row["utility_scores"]
        mask[index, :count] = True
    batch["utility_scores"] = torch.from_numpy(utilities)
    batch["candidate_mask"] = torch.from_numpy(mask)
    return batch


class UtilityNormalizer:
    def __init__(self, mean: float, standard_deviation: float):
        self.mean = float(mean)
        self.standard_deviation = max(float(standard_deviation), 1e-6)

    @classmethod
    def fit(cls, dataset: PairRankingNpzDataset) -> "UtilityNormalizer":
        values = dataset.arrays["utility_scores"]
        return cls(float(values.mean()), float(values.std()))

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.standard_deviation

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.standard_deviation + self.mean

    def to_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "standard_deviation": self.standard_deviation}


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = torch.device(name)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return selected


def _loader(dataset, batch_size, shuffle, seed, workers):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=workers,
        collate_fn=collate_ranking_groups,
    )


def _forward(model: PairRankerModel, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return model(*(batch[key] for key in FEATURE_KEYS))


def _evaluate(model, loader, normalizer, loss_config, device):
    model.eval()
    losses = []
    predicted = []
    targets = []
    masks = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            scores = _forward(model, batch)
            normalized = normalizer.normalize(batch["utility_scores"])
            components = group_aware_pair_ranking_loss(scores, normalized, batch["candidate_mask"], loss_config)
            losses.append({key: float(value.item()) for key, value in components.items()})
            predicted.append(normalizer.denormalize(scores).cpu().numpy())
            targets.append(batch["utility_scores"].cpu().numpy())
            masks.append(batch["candidate_mask"].cpu().numpy())
    metrics = calculate_pair_ranker_metrics(np.concatenate(predicted), np.concatenate(targets), np.concatenate(masks))
    average = {key: float(np.mean([row[key] for row in losses])) for key in losses[0]}
    return metrics, average


def save_pair_ranker_checkpoint(path: str | Path, **payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"checkpoint_version": 1, **payload}, path)


def load_pair_ranker_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    required = {"model_state_dict", "model_architecture_config", "feature_names", "normalization_statistics"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Pair-ranker checkpoint is missing: {sorted(missing)}")
    return checkpoint


def train_pair_ranker(config: PairRankerTrainingConfig) -> dict[str, Any]:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = _device(config.device)
    train_data = PairRankingNpzDataset(config.dataset_directory, "train")
    validation_data = PairRankingNpzDataset(config.dataset_directory, "validation")
    dimensions = train_data.manifest["dimensions"]
    model_config = PairRankerModelConfig(
        parent_feature_dimension=dimensions["parent"],
        rule_feature_dimension=dimensions["rules"],
        farm_feature_dimension=dimensions["farm"],
        objective_feature_dimension=dimensions["objective"],
        metadata_feature_dimension=dimensions["metadata"],
        predictor_feature_dimension=dimensions["predictor"],
        parent_embedding_dimension=config.parent_embedding_dimension,
        auxiliary_embedding_dimension=config.auxiliary_embedding_dimension,
        encoder_hidden_dimensions=config.encoder_hidden_dimensions,
        trunk_hidden_dimensions=config.trunk_hidden_dimensions,
        dropout=config.dropout,
    )
    model = PairRankerModel(model_config).to(device)
    if config.transferred_predictor_checkpoint:
        checkpoint = torch.load(config.transferred_predictor_checkpoint, map_location=device, weights_only=False)
        if int(checkpoint.get("encoding_schema_version", -1)) != int(train_data.manifest["encoding_schema_version"]):
            raise ValueError("Transferred predictor encoding schema is incompatible")
        model.load_parent_encoder_from_predictor(checkpoint, config.freeze_transferred_parent_encoder)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    normalizer = UtilityNormalizer.fit(train_data)
    loss_config = PairRankingLossConfig(
        config.utility_regression_weight,
        config.pairwise_ranking_weight,
        config.best_pair_weight,
        config.tie_tolerance,
    )
    start_epoch = 1
    best_regret = float("inf")
    if config.resume_checkpoint:
        checkpoint = load_pair_ranker_checkpoint(config.resume_checkpoint, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_regret = float(checkpoint["best_validation_regret"])
        normalizer = UtilityNormalizer(**checkpoint["normalization_statistics"])
    train_loader = _loader(train_data, config.batch_size, True, config.seed, config.number_of_workers)
    validation_loader = _loader(validation_data, config.batch_size, False, config.seed, config.number_of_workers)
    output = config.output_path
    output.mkdir(parents=True, exist_ok=True)
    (output / "training_config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    history = []
    stale_epochs = 0

    def save(name: str, epoch: int, metrics: dict) -> None:
        save_pair_ranker_checkpoint(
            output / name,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            epoch=epoch,
            best_validation_regret=best_regret,
            model_architecture_config=model_config.to_dict(),
            feature_names=train_data.manifest["feature_names"],
            input_dimensions=dimensions,
            normalization_statistics=normalizer.to_dict(),
            loss_configuration=asdict(loss_config),
            encoding_schema_version=train_data.manifest["encoding_schema_version"],
            dataset_schema_version=train_data.manifest["dataset_version"],
            game_rules_version=train_data.manifest.get("game_rules_version", "unknown"),
            training_configuration=config.to_dict(),
            training_seed=config.seed,
            metrics=metrics,
        )

    for epoch in range(start_epoch, config.number_of_epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            scores = _forward(model, batch)
            components = group_aware_pair_ranking_loss(
                scores,
                normalizer.normalize(batch["utility_scores"]),
                batch["candidate_mask"],
                loss_config,
            )
            components["total_loss"].backward()
            if config.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            train_losses.append(float(components["total_loss"].item()))
        metrics, validation_losses = _evaluate(model, validation_loader, normalizer, loss_config, device)
        row = {"epoch": epoch, "training_loss": float(np.mean(train_losses)), "validation_losses": validation_losses, "metrics": metrics}
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={row['training_loss']:.5f} "
            f"validation_loss={validation_losses['total_loss']:.5f} regret={metrics['mean_utility_regret']:.5f}"
        )
        regret = metrics["mean_utility_regret"]
        if regret < best_regret:
            best_regret = regret
            stale_epochs = 0
            save("best.pt", epoch, metrics)
        else:
            stale_epochs += 1
        save("latest.pt", epoch, metrics)
        (output / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if config.early_stopping_patience and stale_epochs >= config.early_stopping_patience:
            break
    return {"model": model, "history": history, "best_checkpoint": output / "best.pt", "latest_checkpoint": output / "latest.pt"}


def evaluate_pair_ranker_checkpoint(dataset_path: str | Path, checkpoint_path: str | Path, split: str = "test", device: str = "cpu") -> dict:
    selected = _device(device)
    data = PairRankingNpzDataset(dataset_path, split)
    checkpoint = load_pair_ranker_checkpoint(checkpoint_path, selected)
    if int(checkpoint["encoding_schema_version"]) != int(data.manifest["encoding_schema_version"]):
        raise ValueError("Checkpoint encoding schema is incompatible with dataset")
    values = dict(checkpoint["model_architecture_config"])
    values["encoder_hidden_dimensions"] = tuple(values["encoder_hidden_dimensions"])
    values["trunk_hidden_dimensions"] = tuple(values["trunk_hidden_dimensions"])
    model = PairRankerModel(PairRankerModelConfig(**values)).to(selected)
    model.load_state_dict(checkpoint["model_state_dict"])
    norm = checkpoint["normalization_statistics"]
    normalizer = UtilityNormalizer(norm["mean"], norm["standard_deviation"])
    loss_config = PairRankingLossConfig(**checkpoint["loss_configuration"])
    metrics, losses = _evaluate(model, _loader(data, 16, False, 0, 0), normalizer, loss_config, selected)
    return {"metrics": metrics, "losses": losses}
