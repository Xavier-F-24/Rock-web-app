"""Typed outputs for hybrid analytical and Monte Carlo evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ScalarEstimate:
    mean: float
    standard_deviation: float
    standard_error: float
    sample_count: int
    method: str = "monte_carlo"

    @property
    def confidence_95(self) -> tuple[float, float]:
        margin = 1.96 * self.standard_error
        return self.mean - margin, self.mean + margin


@dataclass(frozen=True)
class GeneOutcomeDistribution:
    gene_name: str
    allele_pair_probabilities: dict[str, float]
    phenotype_probabilities: dict[str, float]
    homozygous_probability: float
    heterozygous_probability: float
    non_mutation_allele_pair_probabilities: dict[str, float]
    non_mutation_phenotype_probabilities: dict[str, float]
    mutation_adjusted_allele_pair_probabilities: dict[str, float]
    mutation_adjusted_phenotype_probabilities: dict[str, float]
    mutation_chance: float
    calculation_method: str = "analytical"

    @staticmethod
    def pair_key(allele_a: int, allele_b: int) -> str:
        return f"{int(allele_a)}|{int(allele_b)}"

    @staticmethod
    def parse_pair_key(key: str) -> tuple[int, int]:
        left, right = key.split("|", maxsplit=1)
        return int(left), int(right)


@dataclass(frozen=True)
class BreedingExpectationRecord:
    parent_ids: tuple[int | str, int | str]
    breeding_rule_encoding: dict[str, Any]
    per_gene_outcome_distributions: dict[str, GeneOutcomeDistribution]
    expected_child_value: ScalarEstimate
    expected_average_surviving_child_value: ScalarEstimate
    expected_maximum_child_value: ScalarEstimate
    expected_raw_clutch_size: ScalarEstimate
    expected_survivor_count: ScalarEstimate
    mutation_probability: float
    expected_mutations_per_child: float
    phenotype_probability_vector: dict[str, float]
    genotype_diversity_estimate: ScalarEstimate
    phenotype_diversity_estimate: ScalarEstimate
    confidence_metadata: dict[str, Any]
    field_methods: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
