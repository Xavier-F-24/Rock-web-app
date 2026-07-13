"""Fixed-width predictor examples and target-column schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import Rock_Genetics.rock_genetic_helper as genetics
from Rock_AI.datasets.breeding_expectation_record import BreedingExpectationRecord
from Rock_AI.representations.encoding_schema_helper import EncodingSchema, get_default_encoding_schema


SCALAR_TARGET_NAMES = (
    "expected_raw_clutch_size",
    "expected_survivor_count",
    "expected_average_surviving_child_value",
    "expected_maximum_surviving_child_value",
    "expected_mutation_count",
    "probability_at_least_one_mutation",
    "genotype_diversity_estimate",
    "phenotype_diversity_estimate",
)


@dataclass(frozen=True)
class PredictorTargetSchema:
    value_thresholds: tuple[float, ...]
    scalar_target_names: tuple[str, ...]
    threshold_target_names: tuple[str, ...]
    allele_distribution_target_names: tuple[str, ...]
    phenotype_target_names: tuple[str, ...]

    @property
    def target_names(self) -> tuple[str, ...]:
        return (
            self.scalar_target_names
            + self.threshold_target_names
            + self.allele_distribution_target_names
            + self.phenotype_target_names
        )

    @classmethod
    def build(
        cls,
        value_thresholds: tuple[float, ...],
        schema: EncodingSchema | None = None,
    ) -> "PredictorTargetSchema":
        schema = schema or get_default_encoding_schema()
        thresholds = tuple(sorted({float(value) for value in value_thresholds}))
        allele_names: list[str] = []
        for gene_name in schema.gene_names:
            alleles = tuple(sorted(genetics.GENE_SPECS[gene_name].options))
            allele_names.extend(
                f"gene.{gene_name}.allele_pair.{allele_a}|{allele_b}"
                for allele_a in alleles
                for allele_b in alleles
            )
        phenotype_names = tuple(
            f"phenotype.{gene_name}={phenotype}"
            for gene_name in schema.gene_names
            for phenotype in schema.phenotype_values[gene_name]
        )
        return cls(
            value_thresholds=thresholds,
            scalar_target_names=SCALAR_TARGET_NAMES,
            threshold_target_names=tuple(
                f"probability_surviving_value_at_least_{value:g}" for value in thresholds
            ),
            allele_distribution_target_names=tuple(allele_names),
            phenotype_target_names=phenotype_names,
        )


@dataclass(frozen=True)
class PredictorExample:
    parent_a_features: np.ndarray
    parent_b_features: np.ndarray
    rule_features: np.ndarray
    context_features: np.ndarray
    schema_version: int
    expected_raw_clutch_size: float
    expected_survivor_count: float
    expected_average_surviving_child_value: float
    expected_maximum_surviving_child_value: float
    surviving_value_threshold_probabilities: dict[str, float]
    expected_mutation_count: float
    probability_at_least_one_mutation: float
    genotype_diversity_estimate: float
    phenotype_diversity_estimate: float
    per_gene_child_allele_pair_distributions: dict[str, dict[str, float]]
    phenotype_probability_vector: dict[str, float]
    metadata: dict[str, Any]

    @classmethod
    def from_expectation(
        cls,
        parent_a_features: np.ndarray,
        parent_b_features: np.ndarray,
        rule_features: np.ndarray,
        context_features: np.ndarray,
        expectation: BreedingExpectationRecord,
        metadata: dict[str, Any],
        schema_version: int,
    ) -> "PredictorExample":
        return cls(
            parent_a_features=np.asarray(parent_a_features, dtype=np.float32),
            parent_b_features=np.asarray(parent_b_features, dtype=np.float32),
            rule_features=np.asarray(rule_features, dtype=np.float32),
            context_features=np.asarray(context_features, dtype=np.float32),
            schema_version=int(schema_version),
            expected_raw_clutch_size=expectation.expected_raw_clutch_size.mean,
            expected_survivor_count=expectation.expected_survivor_count.mean,
            expected_average_surviving_child_value=expectation.expected_average_surviving_child_value.mean,
            expected_maximum_surviving_child_value=expectation.expected_maximum_surviving_child_value.mean,
            surviving_value_threshold_probabilities=dict(
                expectation.surviving_value_threshold_probabilities
            ),
            expected_mutation_count=expectation.expected_mutations_per_child,
            probability_at_least_one_mutation=expectation.mutation_probability,
            genotype_diversity_estimate=expectation.genotype_diversity_estimate.mean,
            phenotype_diversity_estimate=expectation.phenotype_diversity_estimate.mean,
            per_gene_child_allele_pair_distributions={
                gene_name: dict(distribution.allele_pair_probabilities)
                for gene_name, distribution in expectation.per_gene_outcome_distributions.items()
            },
            phenotype_probability_vector=dict(expectation.phenotype_probability_vector),
            metadata=metadata,
        )

    def target_vector(self, target_schema: PredictorTargetSchema) -> np.ndarray:
        values = [
            self.expected_raw_clutch_size,
            self.expected_survivor_count,
            self.expected_average_surviving_child_value,
            self.expected_maximum_surviving_child_value,
            self.expected_mutation_count,
            self.probability_at_least_one_mutation,
            self.genotype_diversity_estimate,
            self.phenotype_diversity_estimate,
        ]
        values.extend(
            self.surviving_value_threshold_probabilities.get(str(threshold), 0.0)
            for threshold in target_schema.value_thresholds
        )
        for name in target_schema.allele_distribution_target_names:
            _, gene_name, _, pair_key = name.split(".", maxsplit=3)
            values.append(
                self.per_gene_child_allele_pair_distributions.get(gene_name, {}).get(
                    pair_key, 0.0
                )
            )
        for name in target_schema.phenotype_target_names:
            payload = name.removeprefix("phenotype.")
            values.append(self.phenotype_probability_vector.get(payload, 0.0))
        return np.asarray(values, dtype=np.float32)
