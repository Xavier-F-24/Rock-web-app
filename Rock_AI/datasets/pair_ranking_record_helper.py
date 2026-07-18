"""Typed records for variable-length breeding-pair ranking groups."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


OBJECTIVE_FEATURE_NAMES = (
    "immediate_expected_value_weight",
    "maximum_offspring_value_weight",
    "survivor_count_weight",
    "genotype_diversity_weight",
    "phenotype_diversity_weight",
    "rare_trait_weight",
    "mutation_opportunity_weight",
    "gene_preservation_weight",
    "risk_aversion_weight",
    "uncertainty_penalty_weight",
    "mortality_penalty_weight",
    "craisen_penalty_weight",
    "zero_survivor_penalty_weight",
    "relatedness_penalty_weight",
    "aleatoric_risk_penalty_weight",
    "epistemic_uncertainty_penalty_weight",
)


@dataclass(frozen=True)
class FarmerObjectiveProfile:
    immediate_expected_value_weight: float = 1.0
    maximum_offspring_value_weight: float = 0.25
    survivor_count_weight: float = 0.5
    genotype_diversity_weight: float = 2.0
    phenotype_diversity_weight: float = 2.0
    rare_trait_weight: float = 3.0
    mutation_opportunity_weight: float = 0.5
    gene_preservation_weight: float = 0.0
    risk_aversion_weight: float = 0.0
    uncertainty_penalty_weight: float = 0.0
    mortality_penalty_weight: float = 1.0
    craisen_penalty_weight: float = 2.0
    zero_survivor_penalty_weight: float = 2.0
    relatedness_penalty_weight: float = 0.25
    aleatoric_risk_penalty_weight: float = 0.25
    epistemic_uncertainty_penalty_weight: float = 0.10
    preserved_gene: str | None = None
    preserved_allele: int | None = None

    @property
    def feature_values(self) -> tuple[float, ...]:
        values = asdict(self)
        return tuple(float(values[name]) for name in OBJECTIVE_FEATURE_NAMES)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PairRankingCandidate:
    parent_ids: tuple[int | str, int | str]
    parent_a_features: np.ndarray
    parent_b_features: np.ndarray
    rule_features: np.ndarray
    farm_features: np.ndarray
    objective_features: np.ndarray
    metadata_features: np.ndarray
    predictor_features: np.ndarray
    utility_components: np.ndarray
    utility_score: float
    uncertainty: float
    rank: int = 0
    best_pair: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PairRankingGroup:
    group_id: str
    lineage_group_id: str
    candidates: tuple[PairRankingCandidate, ...]
    evaluation_seed: int
    monte_carlo_trial_count: int
    objective_profile: FarmerObjectiveProfile
    breeding_rules: dict[str, Any]
    rock_ids: tuple[int | str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("A stored ranking group must contain at least one candidate")


PAIR_METADATA_FEATURE_NAMES = (
    "parent_value_sum_normalized",
    "parent_value_difference_normalized",
    "parent_generation_sum_normalized",
    "parent_generation_difference_normalized",
    "allele_difference_fraction",
    "phenotype_difference_fraction",
    "relatedness_coefficient",
)

UTILITY_COMPONENT_NAMES = (
    "expected_value",
    "maximum_value",
    "survivor_value",
    "genotype_diversity",
    "phenotype_diversity",
    "rare_trait",
    "mutation_opportunity",
    "gene_preservation",
    "uncertainty_penalty",
    "mortality_penalty",
    "craisen_penalty",
    "zero_survivor_penalty",
    "relatedness_penalty",
    "aleatoric_risk_penalty",
    "epistemic_uncertainty_penalty",
)
