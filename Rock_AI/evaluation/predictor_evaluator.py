"""Dataset evaluation plus checkpoint-backed inference from real rocks."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

import Rock_Genetics.rock_genetic_helper as genetics
from Rock_AI.datasets.breeding_record_helper import EncodedBreedingRules
from Rock_AI.evaluation.predictor_metrics import calculate_predictor_metrics
from Rock_AI.models.breeding_predictor_model import (
    BreedingPredictorModel,
    BreedingPredictorModelConfig,
)
from Rock_AI.models.loss_helper import PredictorLossConfig, predictor_multitask_loss
from Rock_AI.models.model_output_helper import BreedingPredictorOutput, TargetLayout
from Rock_AI.representations.encoding_schema_helper import get_default_encoding_schema
from Rock_AI.representations.rock_encoder_helper import encode_rock
from Rock_AI.training.checkpoint_helper import load_predictor_checkpoint
from Rock_AI.training.predictor_data_helper import TargetNormalizer


def output_to_prediction_tensor(
    output: BreedingPredictorOutput,
    layout: TargetLayout,
    normalizer: TargetNormalizer,
) -> torch.Tensor:
    prediction = output.scalar_normalized.new_zeros(
        (output.scalar_normalized.shape[0], len(layout.target_names))
    )
    prediction[:, list(layout.scalar_indices)] = normalizer.denormalize_scalar_tensor(
        output.scalar_normalized
    )
    prediction[:, list(layout.binary_probability_indices)] = output.binary_probabilities
    for values, group in zip(output.genotype_probabilities, layout.genotype_groups):
        prediction[:, list(group.target_indices)] = values
    for values, group in zip(output.phenotype_probabilities, layout.phenotype_groups):
        prediction[:, list(group.target_indices)] = values
    return prediction


class PredictorEvaluator:
    def __init__(
        self,
        model: BreedingPredictorModel,
        layout: TargetLayout,
        normalizer: TargetNormalizer,
        loss_config: PredictorLossConfig,
        device: torch.device,
    ):
        self.model = model
        self.layout = layout
        self.normalizer = normalizer
        self.loss_config = loss_config
        self.device = device

    def evaluate_loader(self, loader) -> tuple[dict[str, Any], dict[str, float]]:
        self.model.eval()
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        losses: list[dict[str, float]] = []
        with torch.no_grad():
            for batch in loader:
                moved = {key: value.to(self.device) for key, value in batch.items()}
                output = self.model(
                    moved["parent_a_features"],
                    moved["parent_b_features"],
                    moved["rule_features"],
                    moved["context_features"],
                )
                normalized_targets = self.normalizer.normalize_scalar_tensor(moved["targets"])
                components = predictor_multitask_loss(
                    output,
                    moved["targets"],
                    moved["target_mask"],
                    self.layout,
                    self.loss_config,
                    normalized_targets,
                )
                losses.append({name: float(value.item()) for name, value in components.items()})
                predictions.append(
                    output_to_prediction_tensor(output, self.layout, self.normalizer).cpu().numpy()
                )
                targets.append(moved["targets"].cpu().numpy())
                masks.append(moved["target_mask"].cpu().numpy())
        average_losses = {
            name: float(np.mean([values[name] for values in losses]))
            for name in losses[0]
        } if losses else {"total_loss": 0.0}
        metrics = calculate_predictor_metrics(
            np.concatenate(predictions),
            np.concatenate(targets),
            np.concatenate(masks),
            self.layout,
            self.normalizer,
        ) if predictions else {}
        return metrics, average_losses


class BreedingPredictor:
    def __init__(
        self,
        model: BreedingPredictorModel,
        checkpoint: dict[str, Any],
        device: torch.device,
    ):
        self.model = model.eval()
        self.checkpoint = checkpoint
        self.device = device
        self.layout = TargetLayout.from_target_names(checkpoint["target_names"])
        self.normalizer = TargetNormalizer.from_dict(checkpoint["normalization_statistics"])
        self.feature_names = checkpoint["feature_names"]

    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
    ) -> "BreedingPredictor":
        selected_device = torch.device(device)
        checkpoint = load_predictor_checkpoint(checkpoint_path, map_location=selected_device)
        current_schema = get_default_encoding_schema()
        if int(checkpoint["encoding_schema_version"]) != current_schema.version:
            raise ValueError(
                f"Checkpoint encoding schema {checkpoint['encoding_schema_version']} is incompatible "
                f"with current schema {current_schema.version}"
            )
        config_values = dict(checkpoint["model_architecture_config"])
        for name in (
            "encoder_hidden_dimensions",
            "trunk_hidden_dimensions",
            "context_swap_pairs",
        ):
            config_values[name] = tuple(
                tuple(item) if isinstance(item, list) else item for item in config_values[name]
            )
        model_config = BreedingPredictorModelConfig(**config_values)
        layout = TargetLayout.from_target_names(checkpoint["target_names"])
        model = BreedingPredictorModel(model_config, layout).to(selected_device)
        model.load_state_dict(checkpoint["model_state_dict"])
        return cls(model, checkpoint, selected_device)

    def _context_from_rocks(
        self,
        parent_a: genetics.Rock,
        parent_b: genetics.Rock,
    ) -> np.ndarray:
        schema = get_default_encoding_schema()
        values = {
            "parent_a_generation_normalized": parent_a.generation / schema.generation_scale,
            "parent_b_generation_normalized": parent_b.generation / schema.generation_scale,
            "generation_difference_normalized": abs(parent_a.generation - parent_b.generation) / schema.generation_scale,
            "parent_a_value_normalized": parent_a.value / schema.value_scale,
            "parent_b_value_normalized": parent_b.value / schema.value_scale,
        }
        unknown = set(self.feature_names["context"]) - set(values)
        if unknown:
            raise ValueError(f"Cannot construct checkpoint context features: {sorted(unknown)}")
        return np.asarray([values[name] for name in self.feature_names["context"]], dtype=np.float32)

    def predict(
        self,
        parent_a: genetics.Rock,
        parent_b: genetics.Rock,
        rules: EncodedBreedingRules | Mapping[str, Any],
        context: np.ndarray | list[float] | None = None,
    ) -> dict[str, Any]:
        schema = get_default_encoding_schema()
        encoded_a = encode_rock(parent_a, schema)
        encoded_b = encode_rock(parent_b, schema)
        expected_parent_names = tuple(self.feature_names["parent"])
        if encoded_a.continuous_feature_names + encoded_a.categorical_feature_names + encoded_a.genotype_feature_names + encoded_a.phenotype_feature_names != expected_parent_names:
            raise ValueError("Checkpoint parent feature order is incompatible with current encoder")
        encoded_rules = EncodedBreedingRules.from_config(rules)
        if tuple(encoded_rules.feature_names) != tuple(self.feature_names["rules"]):
            raise ValueError("Checkpoint rule feature order is incompatible with current rules")
        context_array = (
            self._context_from_rocks(parent_a, parent_b)
            if context is None
            else np.asarray(context, dtype=np.float32)
        )
        expected_context_width = self.model.config.context_feature_dimension
        if context_array.shape != (expected_context_width,):
            raise ValueError(
                f"context must have shape ({expected_context_width},), not {context_array.shape}"
            )
        with torch.no_grad():
            output = self.model(
                torch.from_numpy(encoded_a.as_feature_vector().astype(np.float32)).unsqueeze(0).to(self.device),
                torch.from_numpy(encoded_b.as_feature_vector().astype(np.float32)).unsqueeze(0).to(self.device),
                torch.tensor(encoded_rules.feature_values, dtype=torch.float32, device=self.device).unsqueeze(0),
                torch.from_numpy(context_array).unsqueeze(0).to(self.device),
            )
            full = output_to_prediction_tensor(output, self.layout, self.normalizer)[0].cpu().numpy()
        result: dict[str, Any] = {
            "parent_ids": [parent_a.id, parent_b.id],
            "scalar_predictions": {
                self.layout.target_names[index]: float(full[index]) for index in self.layout.scalar_indices
            },
            "binary_probability_predictions": {
                self.layout.target_names[index]: float(full[index])
                for index in self.layout.binary_probability_indices
            },
            "genotype_distributions": {},
            "phenotype_distributions": {},
        }
        for group in self.layout.genotype_groups:
            result["genotype_distributions"][group.name] = {
                name: float(full[index])
                for name, index in zip(group.target_names, group.target_indices)
            }
        for group in self.layout.phenotype_groups:
            result["phenotype_distributions"][group.name] = {
                name: float(full[index])
                for name, index in zip(group.target_names, group.target_indices)
            }
        return result
