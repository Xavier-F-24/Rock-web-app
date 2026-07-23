"""Immutable, validated manifest input for local NEAT workers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class TrainingOperation(str, Enum):
    NEW_RUN = "new_run"
    CONTINUE = "continue"
    CONTINUE_AS_BRANCH = "continue_as_branch"
    BRANCH_CHAMPION = "branch_champion"


class TrainingBackendKind(str, Enum):
    BREEDING_PAIR = "breeding_pair"
    FULL_FARMER = "full_farmer"


class BranchInitializationStrategy(str, Enum):
    CHAMPION_PLUS_MUTATIONS = "champion_plus_mutations"
    CHAMPION_AND_DIVERSE_SEEDS = "champion_and_diverse_seeds"
    CHAMPION_CLONES_WITH_PERTURBATION = "champion_clones_with_perturbation"
    CUSTOM_MIX = "custom_mix"


class TrainingSafetyTier(str, Enum):
    SMOKE = "smoke"
    STANDARD = "standard"
    ADVANCED = "advanced"


@dataclass(frozen=True)
class TrainingJobConfig:
    operation: TrainingOperation
    source_run: str
    output_run: str
    additional_generations: int
    seed: int
    source_checkpoint: str | None = None
    source_generation: int | None = None
    source_champion: str | None = None
    dataset_path: str | None = None
    population_size: int = 20
    training_scenarios: int = 10
    validation_scenarios: int = 5
    campaign_generations: int = 3
    checkpoint_frequency: int = 1
    showcase_frequency: int = 1
    trace_frequency: int = 1
    worker_count: int = 1
    deterministic: bool = True
    safety_tier: TrainingSafetyTier = TrainingSafetyTier.SMOKE
    initialization_strategy: BranchInitializationStrategy = BranchInitializationStrategy.CHAMPION_AND_DIVERSE_SEEDS
    exact_elite_count: int = 1
    champion_descendant_fraction: float = 0.60
    fresh_genome_fraction: float = 0.25
    historical_diversity_fraction: float = 0.15
    structural_mutation_scale: float = 1.0
    weight_mutation_scale: float = 1.0
    preserve_recurrent_structure: bool = True
    permit_simplification_mutations: bool = True
    supervised_weight: float = 0.60
    campaign_weight: float = 0.40
    complexity_penalty: float = 0.00001
    advanced_acknowledged: bool = False
    trainer_kind: TrainingBackendKind = TrainingBackendKind.BREEDING_PAIR
    worlds_per_genome: int = 3
    max_rounds_per_world: int = 6
    maximum_decisions_per_farm: int = 100
    maximum_no_progress_rounds: int = 8
    maximum_consecutive_passes: int = 6
    maximum_failed_transactions: int = 12
    maximum_episode_wall_clock_seconds: float = 300.0
    heartbeat_interval_seconds: float = 5.0
    curriculum_start: str = "imports"
    curriculum_max: str = "opponent_generalization"
    minimum_generations_per_stage: int = 3
    curriculum_stability_window: int = 3
    curriculum_invalid_rate_threshold: float = 0.05
    curriculum_validation_threshold: float = -0.25

    def __post_init__(self) -> None:
        if self.additional_generations <= 0 or self.population_size <= 0:
            raise ValueError("Generation and population limits must be positive")
        if min(self.training_scenarios, self.validation_scenarios, self.worker_count) < 0 or self.worker_count == 0:
            raise ValueError("Scenario counts cannot be negative and worker_count must be positive")
        if abs(self.supervised_weight + self.campaign_weight - 1.0) > 1e-9:
            raise ValueError("Fitness weights must sum to one")
        if min(self.worlds_per_genome, self.max_rounds_per_world, self.minimum_generations_per_stage, self.curriculum_stability_window) <= 0:
            raise ValueError("Full-farmer training counts must be positive")
        if min(self.maximum_decisions_per_farm, self.maximum_no_progress_rounds, self.maximum_consecutive_passes, self.maximum_failed_transactions) <= 0:
            raise ValueError("Full-farmer liveness limits must be positive")
        if self.maximum_episode_wall_clock_seconds <= 0 or self.heartbeat_interval_seconds <= 0:
            raise ValueError("Full-farmer timing limits must be positive")
        if not 0.0 <= self.curriculum_invalid_rate_threshold <= 1.0:
            raise ValueError("Curriculum invalid-rate threshold must be between zero and one")
        fractions = self.champion_descendant_fraction + self.fresh_genome_fraction + self.historical_diversity_fraction
        if abs(fractions - 1.0) > 1e-9:
            raise ValueError("Branch population fractions must sum to one")
        if self.safety_tier is TrainingSafetyTier.SMOKE:
            limits = (self.population_size <= 20, self.additional_generations <= 5, self.training_scenarios <= 10, self.validation_scenarios <= 5, self.worker_count == 1)
            if not all(limits):
                raise ValueError("Smoke Training limits were exceeded")
        if self.safety_tier is TrainingSafetyTier.ADVANCED and not self.advanced_acknowledged:
            raise ValueError("Advanced Training requires explicit acknowledgement")
        if self.operation is not TrainingOperation.NEW_RUN and not self.source_run:
            raise ValueError("Continuation and branch operations require a source run")
        if self.operation is TrainingOperation.CONTINUE and not self.source_checkpoint:
            raise ValueError("Full continuation requires a trusted local checkpoint")
        if self.operation is TrainingOperation.BRANCH_CHAMPION and not self.source_champion:
            raise ValueError("Champion branching requires a safe source champion")

    def validate_paths(self, repository_root: Path) -> None:
        root = repository_root.resolve()
        for label, value in (("source_run", self.source_run), ("output_run", self.output_run), ("dataset_path", self.dataset_path), ("source_checkpoint", self.source_checkpoint), ("source_champion", self.source_champion)):
            if not value:
                continue
            resolved = (root / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
            if root != resolved and root not in resolved.parents:
                raise ValueError(f"{label} must remain inside the repository")
        source_resolved = (root / self.source_run).resolve() if self.source_run and not Path(self.source_run).is_absolute() else Path(self.source_run).resolve() if self.source_run else None
        output_resolved = (root / self.output_run).resolve() if not Path(self.output_run).is_absolute() else Path(self.output_run).resolve()
        if self.operation is TrainingOperation.BRANCH_CHAMPION and source_resolved == output_resolved:
            raise ValueError("Champion branches cannot overwrite their parent run")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self); payload["operation"] = self.operation.value; payload["safety_tier"] = self.safety_tier.value; payload["initialization_strategy"] = self.initialization_strategy.value; payload["trainer_kind"] = self.trainer_kind.value
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingJobConfig":
        payload = dict(data); payload["operation"] = TrainingOperation(payload["operation"]); payload["safety_tier"] = TrainingSafetyTier(payload.get("safety_tier", "smoke")); payload["initialization_strategy"] = BranchInitializationStrategy(payload.get("initialization_strategy", "champion_and_diverse_seeds")); payload["trainer_kind"] = TrainingBackendKind(payload.get("trainer_kind", "breeding_pair"))
        return cls(**payload)
