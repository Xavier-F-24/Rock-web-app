"""Mean and shallow symmetric baselines for honest model comparison."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from Rock_AI.models.loss_helper import PredictorLossConfig, predictor_multitask_loss
from Rock_AI.models.model_output_helper import BreedingPredictorOutput, TargetLayout
from Rock_AI.training.predictor_data_helper import TargetNormalizer


class TrainingTargetMeanBaseline:
    def __init__(self, mean_targets: np.ndarray):
        self.mean_targets = np.asarray(mean_targets, dtype=np.float32)

    @classmethod
    def fit(cls, targets: np.ndarray, target_mask: np.ndarray) -> "TrainingTargetMeanBaseline":
        weights = target_mask.astype(np.float64)
        denominator = weights.sum(axis=0)
        means = (targets * weights).sum(axis=0) / np.maximum(denominator, 1.0)
        return cls(means)

    def predict(self, count: int) -> np.ndarray:
        return np.repeat(self.mean_targets[None, :], count, axis=0)


class ShallowLinearBaseline(nn.Module):
    def __init__(
        self,
        parent_dimension: int,
        rule_dimension: int,
        context_dimension: int,
        layout: TargetLayout,
    ):
        super().__init__()
        self.layout = layout
        width = parent_dimension * 3 + rule_dimension + context_dimension
        self.input_normalizer = nn.LayerNorm(width, elementwise_affine=False)
        self.scalar_head = nn.Linear(width, len(layout.scalar_indices))
        self.binary_head = nn.Linear(width, len(layout.binary_probability_indices))
        self.genotype_heads = nn.ModuleList(
            nn.Linear(width, len(group.target_indices)) for group in layout.genotype_groups
        )
        self.phenotype_heads = nn.ModuleList(
            nn.Linear(width, len(group.target_indices)) for group in layout.phenotype_groups
        )

    def forward(self, parent_a, parent_b, rules, context) -> BreedingPredictorOutput:
        combined = torch.cat(
            (parent_a + parent_b, torch.abs(parent_a - parent_b), parent_a * parent_b, rules, context),
            dim=-1,
        )
        combined = self.input_normalizer(combined)
        binary_logits = self.binary_head(combined)
        genotype_logits = tuple(head(combined) for head in self.genotype_heads)
        phenotype_logits = tuple(head(combined) for head in self.phenotype_heads)
        return BreedingPredictorOutput(
            scalar_normalized=self.scalar_head(combined),
            binary_logits=binary_logits,
            binary_probabilities=torch.sigmoid(binary_logits),
            genotype_logits=genotype_logits,
            genotype_probabilities=tuple(torch.softmax(values, dim=-1) for values in genotype_logits),
            phenotype_logits=phenotype_logits,
            phenotype_probabilities=tuple(torch.softmax(values, dim=-1) for values in phenotype_logits),
        )


def fit_shallow_linear_baseline(
    model: ShallowLinearBaseline,
    loader,
    normalizer: TargetNormalizer,
    layout: TargetLayout,
    *,
    epochs: int = 50,
    learning_rate: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> ShallowLinearBaseline:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_config = PredictorLossConfig()
    for _ in range(epochs):
        model.train()
        for batch in loader:
            moved = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(
                moved["parent_a_features"],
                moved["parent_b_features"],
                moved["rule_features"],
                moved["context_features"],
            )
            losses = predictor_multitask_loss(
                output,
                moved["targets"],
                moved["target_mask"],
                layout,
                loss_config,
                normalizer.normalize_scalar_tensor(moved["targets"]),
            )
            losses["total_loss"].backward()
            optimizer.step()
    return model.eval()
