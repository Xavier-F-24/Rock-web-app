"""Validated configuration for predictor dataset generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainingDataConfig:
    number_of_parent_pairs: int = 100
    trials_per_pair: int = 100
    seed: int = 1234
    mutation_chance_range: tuple[float, float] = (0.0, 0.12)
    death_chance_range: tuple[float, float] = (0.0, 0.15)
    craisen_chance_range: tuple[float, float] = (0.0, 0.5)
    clutch_mean_range: tuple[float, float] = (1.0, 2.5)
    clutch_std_range: tuple[float, float] = (1.0, 2.5)
    value_thresholds: tuple[float, ...] = (5.0, 10.0, 20.0)
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    output_directory: str = "training_data/breeding_predictor_v1"
    file_format: str = "npz"
    maximum_rocks: int = 128
    include_context_features: bool = True
    game_rules_version: str = "rock-game-breeding-v1"

    def __post_init__(self) -> None:
        if self.number_of_parent_pairs <= 0:
            raise ValueError("number_of_parent_pairs must be positive")
        if self.trials_per_pair <= 0:
            raise ValueError("trials_per_pair must be positive")
        if self.maximum_rocks <= 0:
            raise ValueError("maximum_rocks must be positive")
        for name in (
            "mutation_chance_range",
            "death_chance_range",
            "craisen_chance_range",
        ):
            low, high = getattr(self, name)
            if not 0.0 <= low <= high <= 1.0:
                raise ValueError(f"{name} must be ordered within [0, 1]")
        for name in ("clutch_mean_range", "clutch_std_range"):
            low, high = getattr(self, name)
            if low < 0.0 or high < low:
                raise ValueError(f"{name} must be non-negative and ordered")
        fractions = self.train_fraction + self.validation_fraction + self.test_fraction
        if abs(fractions - 1.0) > 1e-9:
            raise ValueError("train, validation, and test fractions must sum to 1")
        if min(self.train_fraction, self.validation_fraction, self.test_fraction) < 0.0:
            raise ValueError("split fractions cannot be negative")
        if self.file_format not in {"npz", "npz+jsonl"}:
            raise ValueError("file_format must be 'npz' or 'npz+jsonl'")
        if any(value < 0 for value in self.value_thresholds):
            raise ValueError("value thresholds cannot be negative")

    @property
    def output_path(self) -> Path:
        return Path(self.output_directory)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["value_thresholds"] = list(self.value_thresholds)
        return result


@dataclass(frozen=True)
class PredictorTrainingConfig:
    dataset_path: str
    output_directory: str
    seed: int = 1234
    batch_size: int = 32
    learning_rate: float = 1e-3
    number_of_epochs: int = 20
    encoder_hidden_dimensions: tuple[int, ...] = (128,)
    trunk_hidden_dimensions: tuple[int, ...] = (128, 96)
    parent_embedding_dimension: int = 64
    rule_embedding_dimension: int = 24
    context_embedding_dimension: int = 16
    dropout: float = 0.1
    weight_decay: float = 1e-5
    validation_frequency: int = 1
    early_stopping_patience: int = 5
    checkpoint_frequency: int = 1
    device: str = "auto"
    scalar_loss_weight: float = 1.0
    probability_loss_weight: float = 1.0
    phenotype_distribution_weight: float = 1.0
    genotype_distribution_weight: float = 1.0
    number_of_workers: int = 0
    deterministic: bool = True
    gradient_clip_norm: float | None = 5.0
    resume_checkpoint: str | None = None

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.number_of_epochs <= 0:
            raise ValueError("batch_size and number_of_epochs must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.validation_frequency <= 0 or self.checkpoint_frequency <= 0:
            raise ValueError("validation and checkpoint frequencies must be positive")
        if self.early_stopping_patience < 0 or self.number_of_workers < 0:
            raise ValueError("patience and worker count cannot be negative")

    @property
    def dataset_directory(self) -> Path:
        return Path(self.dataset_path)

    @property
    def output_path(self) -> Path:
        return Path(self.output_directory)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PairRankingDataConfig:
    number_of_farms: int = 100
    trials_per_pair: int = 100
    seed: int = 1234
    output_directory: str = "training_data/pair_ranker_v1"
    predictor_checkpoint: str | None = None
    minimum_rocks: int = 4
    maximum_rocks: int = 8
    mutation_chance_range: tuple[float, float] = (0.0, 0.12)
    death_chance_range: tuple[float, float] = (0.0, 0.15)
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    retain_single_candidate_farms: bool = False
    game_rules_version: str = "rock-game-breeding-v1"
    observation_access: str = "player"

    def __post_init__(self) -> None:
        if self.number_of_farms <= 0 or self.trials_per_pair <= 0:
            raise ValueError("number_of_farms and trials_per_pair must be positive")
        if not 2 <= self.minimum_rocks <= self.maximum_rocks:
            raise ValueError("rock bounds must satisfy 2 <= minimum <= maximum")
        if abs(self.train_fraction + self.validation_fraction + self.test_fraction - 1.0) > 1e-9:
            raise ValueError("split fractions must sum to 1")
        for low, high in (self.mutation_chance_range, self.death_chance_range):
            if not 0.0 <= low <= high <= 1.0:
                raise ValueError("probability ranges must be ordered within [0, 1]")
        if self.observation_access != "player":
            raise ValueError("Gameplay pair-ranking datasets must use player observations")

    @property
    def output_path(self) -> Path:
        return Path(self.output_directory)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PairRankerTrainingConfig:
    dataset_path: str
    output_directory: str
    seed: int = 1234
    batch_size: int = 4
    learning_rate: float = 1e-3
    number_of_epochs: int = 20
    parent_embedding_dimension: int = 64
    auxiliary_embedding_dimension: int = 24
    encoder_hidden_dimensions: tuple[int, ...] = (128,)
    trunk_hidden_dimensions: tuple[int, ...] = (128, 64)
    dropout: float = 0.1
    weight_decay: float = 1e-5
    utility_regression_weight: float = 1.0
    pairwise_ranking_weight: float = 1.0
    best_pair_weight: float = 0.5
    tie_tolerance: float = 1e-5
    early_stopping_patience: int = 5
    gradient_clip_norm: float | None = 5.0
    device: str = "auto"
    number_of_workers: int = 0
    resume_checkpoint: str | None = None
    transferred_predictor_checkpoint: str | None = None
    freeze_transferred_parent_encoder: bool = False

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.number_of_epochs <= 0 or self.learning_rate <= 0:
            raise ValueError("batch size, epochs, and learning rate must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if min(self.utility_regression_weight, self.pairwise_ranking_weight, self.best_pair_weight) < 0:
            raise ValueError("loss weights cannot be negative")

    @property
    def dataset_directory(self) -> Path:
        return Path(self.dataset_path)

    @property
    def output_path(self) -> Path:
        return Path(self.output_directory)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
