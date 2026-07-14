"""Symmetric feed-forward scorer for variable breeding-pair candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


def _mlp(input_width: int, hidden: tuple[int, ...], output_width: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_width
    for width in hidden:
        layers.extend((nn.Linear(current, width), nn.ReLU(), nn.Dropout(dropout)))
        current = width
    layers.append(nn.Linear(current, output_width))
    return nn.Sequential(*layers)


@dataclass(frozen=True)
class PairRankerModelConfig:
    parent_feature_dimension: int
    rule_feature_dimension: int
    farm_feature_dimension: int
    objective_feature_dimension: int
    metadata_feature_dimension: int
    predictor_feature_dimension: int = 0
    parent_embedding_dimension: int = 64
    auxiliary_embedding_dimension: int = 24
    encoder_hidden_dimensions: tuple[int, ...] = (128,)
    trunk_hidden_dimensions: tuple[int, ...] = (128, 64)
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)


class PairRankerModel(nn.Module):
    def __init__(self, config: PairRankerModelConfig):
        super().__init__()
        self.config = config
        self.parent_encoder = _mlp(
            config.parent_feature_dimension,
            config.encoder_hidden_dimensions,
            config.parent_embedding_dimension,
            config.dropout,
        )
        self.rule_encoder = _mlp(config.rule_feature_dimension, (), config.auxiliary_embedding_dimension, config.dropout)
        self.farm_encoder = _mlp(config.farm_feature_dimension, (), config.auxiliary_embedding_dimension, config.dropout)
        self.objective_encoder = _mlp(config.objective_feature_dimension, (), config.auxiliary_embedding_dimension, config.dropout)
        self.metadata_encoder = _mlp(config.metadata_feature_dimension, (), config.auxiliary_embedding_dimension, config.dropout)
        self.predictor_encoder = (
            _mlp(config.predictor_feature_dimension, (), config.auxiliary_embedding_dimension, config.dropout)
            if config.predictor_feature_dimension else None
        )
        auxiliary_count = 4 + int(self.predictor_encoder is not None)
        combined = config.parent_embedding_dimension * 3 + config.auxiliary_embedding_dimension * auxiliary_count
        self.scoring_trunk = _mlp(combined, config.trunk_hidden_dimensions, 1, config.dropout)

    def forward(
        self,
        parent_a_features: torch.Tensor,
        parent_b_features: torch.Tensor,
        rule_features: torch.Tensor,
        farm_features: torch.Tensor,
        objective_features: torch.Tensor,
        metadata_features: torch.Tensor,
        predictor_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shape = parent_a_features.shape[:-1]
        embedding_a = self.parent_encoder(parent_a_features)
        embedding_b = self.parent_encoder(parent_b_features)
        pieces = [
            embedding_a + embedding_b,
            torch.abs(embedding_a - embedding_b),
            embedding_a * embedding_b,
            self.rule_encoder(rule_features),
            self.farm_encoder(farm_features),
            self.objective_encoder(objective_features),
            self.metadata_encoder(metadata_features),
        ]
        if self.predictor_encoder is not None:
            if predictor_features is None:
                predictor_features = parent_a_features.new_zeros((*shape, self.config.predictor_feature_dimension))
            pieces.append(self.predictor_encoder(predictor_features))
        return self.scoring_trunk(torch.cat(pieces, dim=-1)).squeeze(-1)

    def load_parent_encoder_from_predictor(self, checkpoint: dict, freeze: bool = False) -> None:
        expected_schema = checkpoint.get("encoding_schema_version")
        architecture = checkpoint.get("model_architecture_config", {})
        if int(architecture.get("parent_feature_dimension", -1)) != self.config.parent_feature_dimension:
            raise ValueError("Predictor parent feature dimension is incompatible with ranker")
        try:
            self.parent_encoder.load_state_dict(
                {
                    key.removeprefix("parent_encoder."): value
                    for key, value in checkpoint["model_state_dict"].items()
                    if key.startswith("parent_encoder.")
                },
                strict=True,
            )
        except RuntimeError as error:
            raise ValueError("Predictor parent encoder architecture is incompatible with ranker") from error
        if freeze:
            for parameter in self.parent_encoder.parameters():
                parameter.requires_grad = False


@dataclass(frozen=True)
class PairRankingLossConfig:
    utility_regression_weight: float = 1.0
    pairwise_ranking_weight: float = 1.0
    best_pair_weight: float = 0.5
    tie_tolerance: float = 1e-5


def group_aware_pair_ranking_loss(
    scores: torch.Tensor,
    utilities: torch.Tensor,
    candidate_mask: torch.Tensor,
    config: PairRankingLossConfig | None = None,
) -> dict[str, torch.Tensor]:
    config = config or PairRankingLossConfig()
    valid_scores = scores[candidate_mask]
    valid_utilities = utilities[candidate_mask]
    regression = F.smooth_l1_loss(valid_scores, valid_utilities) if valid_scores.numel() else scores.sum() * 0
    ranking_terms = []
    best_terms = []
    for group_index in range(scores.shape[0]):
        mask = candidate_mask[group_index]
        group_scores = scores[group_index][mask]
        group_utilities = utilities[group_index][mask]
        if group_scores.numel() == 0:
            continue
        best = torch.isclose(group_utilities, group_utilities.max(), atol=config.tie_tolerance, rtol=0)
        best_terms.append(-torch.logsumexp(torch.log_softmax(group_scores, dim=0)[best], dim=0))
        differences = group_utilities[:, None] - group_utilities[None, :]
        preferred = differences > config.tie_tolerance
        if preferred.any():
            score_differences = group_scores[:, None] - group_scores[None, :]
            ranking_terms.append(F.softplus(-score_differences[preferred]).mean())
    ranking = torch.stack(ranking_terms).mean() if ranking_terms else scores.sum() * 0
    best_pair = torch.stack(best_terms).mean() if best_terms else scores.sum() * 0
    total = (
        regression * config.utility_regression_weight
        + ranking * config.pairwise_ranking_weight
        + best_pair * config.best_pair_weight
    )
    return {
        "total_loss": total,
        "utility_regression_loss": regression,
        "pairwise_ranking_loss": ranking,
        "best_pair_loss": best_pair,
    }
