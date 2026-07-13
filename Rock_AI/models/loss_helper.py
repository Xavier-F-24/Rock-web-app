"""Balanced masked multi-task loss for predictor heads."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch.nn import functional as F

from Rock_AI.models.model_output_helper import BreedingPredictorOutput, TargetLayout


@dataclass(frozen=True)
class PredictorLossConfig:
    scalar_loss_weight: float = 1.0
    probability_loss_weight: float = 1.0
    phenotype_distribution_weight: float = 1.0
    genotype_distribution_weight: float = 1.0
    scalar_loss: str = "huber"

    def to_dict(self) -> dict:
        return asdict(self)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)
    denominator = weights.sum().clamp_min(1.0)
    return (values * weights).sum() / denominator


def _distribution_loss(
    probabilities: tuple[torch.Tensor, ...],
    groups,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for prediction, group in zip(probabilities, groups):
        indices = list(group.target_indices)
        group_target = targets[:, indices]
        group_mask = target_mask[:, indices].to(group_target.dtype)
        masked_target = group_target * group_mask
        mass = masked_target.sum(dim=-1, keepdim=True)
        valid_rows = mass.squeeze(-1) > 0
        if not valid_rows.any():
            continue
        normalized = masked_target / mass.clamp_min(1e-12)
        masked_prediction = prediction * group_mask
        normalized_prediction = masked_prediction / masked_prediction.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)
        cross_entropy = -(
            normalized * torch.log(normalized_prediction.clamp_min(1e-8))
        ).sum(dim=-1)
        losses.append(cross_entropy[valid_rows].mean())
    if not losses:
        return targets.new_zeros(())
    return torch.stack(losses).mean()


def predictor_multitask_loss(
    output: BreedingPredictorOutput,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    layout: TargetLayout,
    config: PredictorLossConfig,
    normalized_scalar_targets: torch.Tensor,
) -> dict[str, torch.Tensor]:
    scalar_mask = target_mask[:, list(layout.scalar_indices)]
    if config.scalar_loss == "mse":
        scalar_values = F.mse_loss(
            output.scalar_normalized, normalized_scalar_targets, reduction="none"
        )
    else:
        scalar_values = F.smooth_l1_loss(
            output.scalar_normalized, normalized_scalar_targets, reduction="none"
        )
    scalar_loss = _masked_mean(scalar_values, scalar_mask)

    binary_indices = list(layout.binary_probability_indices)
    binary_targets = targets[:, binary_indices]
    binary_mask = target_mask[:, binary_indices]
    binary_values = F.binary_cross_entropy_with_logits(
        output.binary_logits, binary_targets, reduction="none"
    )
    probability_loss = _masked_mean(binary_values, binary_mask)
    genotype_loss = _distribution_loss(
        output.genotype_probabilities,
        layout.genotype_groups,
        targets,
        target_mask,
    )
    phenotype_loss = _distribution_loss(
        output.phenotype_probabilities,
        layout.phenotype_groups,
        targets,
        target_mask,
    )
    total = (
        config.scalar_loss_weight * scalar_loss
        + config.probability_loss_weight * probability_loss
        + config.genotype_distribution_weight * genotype_loss
        + config.phenotype_distribution_weight * phenotype_loss
    )
    return {
        "total_loss": total,
        "scalar_loss": scalar_loss,
        "probability_loss": probability_loss,
        "genotype_distribution_loss": genotype_loss,
        "phenotype_distribution_loss": phenotype_loss,
    }
