"""Symmetric multi-head MLP for breeding outcome prediction."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from Rock_AI.models.model_output_helper import BreedingPredictorOutput, TargetLayout


def _mlp(input_dimension: int, hidden_dimensions: tuple[int, ...], output_dimension: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_dimension
    for hidden in hidden_dimensions:
        layers.extend((nn.Linear(current, hidden), nn.ReLU(), nn.Dropout(dropout)))
        current = hidden
    layers.append(nn.Linear(current, output_dimension))
    return nn.Sequential(*layers)


@dataclass(frozen=True)
class BreedingPredictorModelConfig:
    parent_feature_dimension: int
    rule_feature_dimension: int
    context_feature_dimension: int
    parent_embedding_dimension: int = 64
    rule_embedding_dimension: int = 24
    context_embedding_dimension: int = 16
    encoder_hidden_dimensions: tuple[int, ...] = (128,)
    trunk_hidden_dimensions: tuple[int, ...] = (128, 96)
    dropout: float = 0.1
    context_swap_pairs: tuple[tuple[int, int], ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


class BreedingPredictorModel(nn.Module):
    def __init__(self, config: BreedingPredictorModelConfig, target_layout: TargetLayout):
        super().__init__()
        self.config = config
        self.target_layout = target_layout
        self.parent_encoder = _mlp(
            config.parent_feature_dimension,
            config.encoder_hidden_dimensions,
            config.parent_embedding_dimension,
            config.dropout,
        )
        self.rule_encoder = _mlp(
            config.rule_feature_dimension,
            (),
            config.rule_embedding_dimension,
            config.dropout,
        )
        if config.context_feature_dimension:
            self.context_encoder: nn.Module | None = _mlp(
                config.context_feature_dimension,
                (),
                config.context_embedding_dimension,
                config.dropout,
            )
            context_width = config.context_embedding_dimension
        else:
            self.context_encoder = None
            context_width = 0
        combined_width = (
            config.parent_embedding_dimension * 3
            + config.rule_embedding_dimension
            + context_width
        )
        trunk_layers: list[nn.Module] = []
        current = combined_width
        for hidden in config.trunk_hidden_dimensions:
            trunk_layers.extend((nn.Linear(current, hidden), nn.ReLU(), nn.Dropout(config.dropout)))
            current = hidden
        self.trunk = nn.Sequential(*trunk_layers) if trunk_layers else nn.Identity()
        self.scalar_head = nn.Linear(current, len(target_layout.scalar_indices))
        self.binary_head = nn.Linear(current, len(target_layout.binary_probability_indices))
        self.genotype_heads = nn.ModuleList(
            nn.Linear(current, len(group.target_indices)) for group in target_layout.genotype_groups
        )
        self.phenotype_heads = nn.ModuleList(
            nn.Linear(current, len(group.target_indices)) for group in target_layout.phenotype_groups
        )

    def _symmetric_context(self, context: torch.Tensor) -> torch.Tensor:
        if not self.config.context_swap_pairs:
            return context
        result = context.clone()
        for left, right in self.config.context_swap_pairs:
            left_value = context[..., left]
            right_value = context[..., right]
            result[..., left] = left_value + right_value
            result[..., right] = torch.abs(left_value - right_value)
        return result

    def forward(
        self,
        parent_a_features: torch.Tensor,
        parent_b_features: torch.Tensor,
        rule_features: torch.Tensor,
        context_features: torch.Tensor | None = None,
    ) -> BreedingPredictorOutput:
        embedding_a = self.parent_encoder(parent_a_features)
        embedding_b = self.parent_encoder(parent_b_features)
        pieces = [
            embedding_a + embedding_b,
            torch.abs(embedding_a - embedding_b),
            embedding_a * embedding_b,
            self.rule_encoder(rule_features),
        ]
        if self.context_encoder is not None:
            if context_features is None:
                context_features = parent_a_features.new_zeros(
                    (parent_a_features.shape[0], self.config.context_feature_dimension)
                )
            pieces.append(self.context_encoder(self._symmetric_context(context_features)))
        shared = self.trunk(torch.cat(pieces, dim=-1))
        binary_logits = self.binary_head(shared)
        genotype_logits = tuple(head(shared) for head in self.genotype_heads)
        phenotype_logits = tuple(head(shared) for head in self.phenotype_heads)
        return BreedingPredictorOutput(
            scalar_normalized=self.scalar_head(shared),
            binary_logits=binary_logits,
            binary_probabilities=torch.sigmoid(binary_logits),
            genotype_logits=genotype_logits,
            genotype_probabilities=tuple(torch.softmax(logits, dim=-1) for logits in genotype_logits),
            phenotype_logits=phenotype_logits,
            phenotype_probabilities=tuple(torch.softmax(logits, dim=-1) for logits in phenotype_logits),
        )
