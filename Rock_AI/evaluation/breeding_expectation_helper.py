"""Hybrid exact genetics and real-engine Monte Carlo expectations."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import Any, Iterable, Mapping

import Rock_Genetics.rock_genetic_helper as genetics
from Rock_AI.datasets.breeding_expectation_record import (
    BreedingExpectationRecord,
    ScalarEstimate,
)
from Rock_AI.datasets.breeding_record_helper import BreedingRecord, EncodedBreedingRules
from Rock_AI.environments.breeding_training_environment import BreedingTrainingEnvironment
from Rock_AI.evaluation.genetics_evaluator import GeneticsEvaluator


def _estimate(values: Iterable[float], method: str = "monte_carlo") -> ScalarEstimate:
    samples = [float(value) for value in values]
    if not samples:
        return ScalarEstimate(0.0, 0.0, 0.0, 0, method)
    standard_deviation = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return ScalarEstimate(
        mean=statistics.fmean(samples),
        standard_deviation=standard_deviation,
        standard_error=standard_deviation / math.sqrt(len(samples)),
        sample_count=len(samples),
        method=method,
    )


def _genotype_signature(genotype: dict[str, list[int]]) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (gene_name, int(alleles[0]), int(alleles[1]))
        for gene_name, alleles in sorted(genotype.items())
    )


def _phenotype_signature(phenotypes: dict[str, str | None]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (gene_name, "<missing>" if phenotype is None else str(phenotype))
        for gene_name, phenotype in sorted(phenotypes.items())
    )


class BreedingExpectationEvaluator:
    def __init__(self, genetics_evaluator: GeneticsEvaluator | None = None):
        self.genetics_evaluator = genetics_evaluator or GeneticsEvaluator()

    @staticmethod
    def _trial_child_mean(record: BreedingRecord) -> float:
        return statistics.fmean(record.child_values) if record.child_values else 0.0

    @staticmethod
    def _trial_survivor_mean(record: BreedingRecord) -> float | None:
        values = [
            value
            for value, status in zip(record.child_values, record.child_statuses)
            if status == genetics.RockStatus.ACTIVE.value
        ]
        return statistics.fmean(values) if values else None

    @staticmethod
    def _trial_diversity(items: tuple[dict[str, Any], ...], signature) -> float:
        if not items:
            return 0.0
        return len({signature(item) for item in items}) / len(items)

    @staticmethod
    def _phenotype_vector(records: list[BreedingRecord]) -> dict[str, float]:
        counts: Counter[str] = Counter()
        gene_totals: Counter[str] = Counter()
        for record in records:
            for child in record.child_phenotypes:
                for gene_name, phenotype in child.items():
                    key = f"{gene_name}={phenotype if phenotype is not None else '<missing>'}"
                    counts[key] += 1
                    gene_totals[gene_name] += 1
        return {
            key: count / gene_totals[key.split("=", maxsplit=1)[0]]
            for key, count in sorted(counts.items())
        }

    def evaluate(
        self,
        parent_a: genetics.Rock,
        parent_b: genetics.Rock,
        *,
        rules: EncodedBreedingRules | Mapping[str, Any] | None = None,
        trial_count: int = 1000,
        seed: int = 0,
    ) -> BreedingExpectationRecord:
        if trial_count <= 0:
            raise ValueError("trial_count must be positive")
        encoded_rules = EncodedBreedingRules.from_config(rules)
        distributions = self.genetics_evaluator.evaluate_all_genes(
            parent_a,
            parent_b,
            mutation_chance=encoded_rules.mutation_chance,
        )
        environment = BreedingTrainingEnvironment(seed=seed, rules=encoded_rules)
        records = environment.repeat_pair(
            parent_a,
            parent_b,
            trial_count,
            start_seed=seed,
            rules=encoded_rules,
        )

        expected_child_value = _estimate(self._trial_child_mean(record) for record in records)
        survivor_means = [self._trial_survivor_mean(record) for record in records]
        expected_surviving_value = _estimate(
            value for value in survivor_means if value is not None
        )
        expected_maximum = _estimate(
            max(record.child_values) if record.child_values else 0.0 for record in records
        )
        expected_clutch = _estimate(record.clutch_size for record in records)
        expected_survivors = _estimate(record.survivor_count for record in records)
        genotype_diversity = _estimate(
            self._trial_diversity(record.child_genotypes, _genotype_signature)
            for record in records
        )
        phenotype_diversity = _estimate(
            self._trial_diversity(record.child_phenotypes, _phenotype_signature)
            for record in records
        )

        mutation_attempts = 2 * (
            len(self.genetics_evaluator.schema.gene_names)
            + len(self.genetics_evaluator.schema.death_gene_names)
        )
        mutation_probability = 1.0 - (1.0 - encoded_rules.mutation_chance) ** mutation_attempts
        expected_mutations = mutation_attempts * encoded_rules.mutation_chance
        estimates = {
            "expected_child_value": expected_child_value,
            "expected_average_surviving_child_value": expected_surviving_value,
            "expected_maximum_child_value": expected_maximum,
            "expected_raw_clutch_size": expected_clutch,
            "expected_survivor_count": expected_survivors,
            "genotype_diversity_estimate": genotype_diversity,
            "phenotype_diversity_estimate": phenotype_diversity,
        }
        return BreedingExpectationRecord(
            parent_ids=(parent_a.id, parent_b.id),
            breeding_rule_encoding=encoded_rules.to_dict(),
            per_gene_outcome_distributions=distributions,
            expected_child_value=expected_child_value,
            expected_average_surviving_child_value=expected_surviving_value,
            expected_maximum_child_value=expected_maximum,
            expected_raw_clutch_size=expected_clutch,
            expected_survivor_count=expected_survivors,
            mutation_probability=mutation_probability,
            expected_mutations_per_child=expected_mutations,
            phenotype_probability_vector=self._phenotype_vector(records),
            genotype_diversity_estimate=genotype_diversity,
            phenotype_diversity_estimate=phenotype_diversity,
            confidence_metadata={
                "seed": int(seed),
                "trial_count": int(trial_count),
                "confidence_level": 0.95,
                "confidence_intervals": {
                    name: list(estimate.confidence_95) for name, estimate in estimates.items()
                },
                "total_returned_children": sum(len(record.child_ids) for record in records),
                "diversity_definition": "mean within-trial unique-outcome fraction",
                "mutation_scope": "analytical probability across ordinary and death-gene mutation attempts per conceived child",
            },
            field_methods={
                "per_gene_outcome_distributions": "analytical",
                "expected_child_value": "monte_carlo_real_breeding_engine",
                "expected_average_surviving_child_value": "monte_carlo_real_breeding_engine",
                "expected_maximum_child_value": "monte_carlo_real_breeding_engine",
                "expected_raw_clutch_size": "monte_carlo_real_breeding_engine",
                "expected_survivor_count": "monte_carlo_real_breeding_engine",
                "mutation_probability": "analytical",
                "expected_mutations_per_child": "analytical",
                "phenotype_probability_vector": "monte_carlo_real_breeding_engine",
                "genotype_diversity_estimate": "monte_carlo_real_breeding_engine",
                "phenotype_diversity_estimate": "monte_carlo_real_breeding_engine",
            },
        )
