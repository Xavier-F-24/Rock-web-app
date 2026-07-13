"""Target-appropriate scalar, binary, and distribution metrics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from Rock_AI.models.model_output_helper import TargetLayout
from Rock_AI.training.predictor_data_helper import TargetNormalizer


def _masked_values(predictions, targets, mask):
    valid = mask.astype(bool)
    return predictions[valid], targets[valid]


def _calibration_error(predictions: np.ndarray, targets: np.ndarray, bins: int = 10) -> float:
    if not len(predictions):
        return 0.0
    error = 0.0
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        selected = (predictions >= boundaries[index]) & (
            predictions <= boundaries[index + 1] if index == bins - 1 else predictions < boundaries[index + 1]
        )
        if selected.any():
            error += selected.mean() * abs(predictions[selected].mean() - targets[selected].mean())
    return float(error)


def calculate_predictor_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    target_mask: np.ndarray,
    layout: TargetLayout,
    normalizer: TargetNormalizer,
) -> dict[str, Any]:
    scalar_indices = list(layout.scalar_indices)
    scalar_prediction, scalar_target = _masked_values(
        predictions[:, scalar_indices],
        targets[:, scalar_indices],
        target_mask[:, scalar_indices],
    )
    scalar_errors = scalar_prediction - scalar_target
    scalar_mae = float(np.mean(np.abs(scalar_errors))) if len(scalar_errors) else 0.0
    scalar_rmse = float(np.sqrt(np.mean(scalar_errors ** 2))) if len(scalar_errors) else 0.0
    target_centered = scalar_target - scalar_target.mean() if len(scalar_target) else scalar_target
    denominator = float(np.sum(target_centered ** 2))
    r_squared = 1.0 - float(np.sum(scalar_errors ** 2)) / denominator if denominator > 1e-12 else 0.0
    normalized_prediction = normalizer.normalize_array(predictions[:, scalar_indices])
    normalized_target = normalizer.normalize_array(targets[:, scalar_indices])
    normalized_mask = target_mask[:, scalar_indices].astype(bool)
    normalized_rmse = float(
        np.sqrt(np.mean((normalized_prediction[normalized_mask] - normalized_target[normalized_mask]) ** 2))
    ) if normalized_mask.any() else 0.0
    scalar_per_target: dict[str, dict[str, float]] = {}
    for index in layout.scalar_indices:
        prediction_values, target_values = _masked_values(
            predictions[:, index], targets[:, index], target_mask[:, index]
        )
        errors = prediction_values - target_values
        centered = target_values - target_values.mean() if len(target_values) else target_values
        total_variance = float(np.sum(centered ** 2))
        scalar_per_target[layout.target_names[index]] = {
            "mae": float(np.mean(np.abs(errors))) if len(errors) else 0.0,
            "rmse": float(np.sqrt(np.mean(errors ** 2))) if len(errors) else 0.0,
            "r_squared": (
                1.0 - float(np.sum(errors ** 2)) / total_variance
                if total_variance > 1e-12
                else 0.0
            ),
        }

    binary_indices = list(layout.binary_probability_indices)
    binary_prediction, binary_target = _masked_values(
        predictions[:, binary_indices],
        targets[:, binary_indices],
        target_mask[:, binary_indices],
    )
    clipped = np.clip(binary_prediction.astype(np.float64), 1e-7, 1.0 - 1e-7)
    binary_target = binary_target.astype(np.float64)
    binary_cross_entropy = float(
        np.mean(-(binary_target * np.log(clipped) + (1.0 - binary_target) * np.log(1.0 - clipped)))
    ) if len(clipped) else 0.0
    brier = float(np.mean((binary_prediction - binary_target) ** 2)) if len(clipped) else 0.0

    def distribution_metrics(groups) -> dict[str, float]:
        cross_entropies: list[float] = []
        divergences: list[float] = []
        variations: list[float] = []
        accuracies: list[float] = []
        mass_errors: list[float] = []
        for group in groups:
            indices = list(group.target_indices)
            prediction = np.clip(predictions[:, indices], 1e-8, 1.0)
            target = targets[:, indices]
            mask = target_mask[:, indices].astype(bool)
            masked_target = target * mask
            mass = masked_target.sum(axis=1, keepdims=True)
            valid = mass[:, 0] > 0
            if not valid.any():
                continue
            normalized_target = masked_target[valid] / np.maximum(mass[valid], 1e-12)
            valid_prediction = prediction[valid]
            cross_entropies.extend(-(normalized_target * np.log(valid_prediction)).sum(axis=1))
            divergences.extend(
                (normalized_target * (np.log(np.clip(normalized_target, 1e-8, 1.0)) - np.log(valid_prediction))).sum(axis=1)
            )
            variations.extend(0.5 * np.abs(valid_prediction - normalized_target).sum(axis=1))
            accuracies.extend(np.argmax(valid_prediction, axis=1) == np.argmax(normalized_target, axis=1))
            mass_errors.extend(np.abs(valid_prediction.sum(axis=1) - 1.0))
        return {
            "cross_entropy": float(np.mean(cross_entropies)) if cross_entropies else 0.0,
            "kl_divergence": float(np.mean(divergences)) if divergences else 0.0,
            "total_variation_distance": float(np.mean(variations)) if variations else 0.0,
            "top_category_accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
            "probability_mass_sum_error": float(np.mean(mass_errors)) if mass_errors else 0.0,
        }

    genotype = distribution_metrics(layout.genotype_groups)
    phenotype = distribution_metrics(layout.phenotype_groups)
    aggregate = normalized_rmse + brier + genotype["total_variation_distance"] + phenotype["total_variation_distance"]
    return {
        "scalar": {
            "mae": scalar_mae,
            "rmse": scalar_rmse,
            "r_squared": r_squared,
            "normalized_rmse": normalized_rmse,
            "per_target": scalar_per_target,
        },
        "binary_probability": {
            "binary_cross_entropy": binary_cross_entropy,
            "brier_score": brier,
            "expected_calibration_error": _calibration_error(binary_prediction, binary_target),
        },
        "genotype_distribution": genotype,
        "phenotype_distribution": phenotype,
        "aggregate_validation_score": aggregate,
    }
