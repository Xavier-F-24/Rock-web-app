"""Transparent objective-dependent labels for pair ranking."""

from __future__ import annotations

import math
from dataclasses import dataclass

import Rock_Genetics.rock_genetic_helper as genetics
from Rock_AI.datasets.pair_ranking_record_helper import (
    FarmerObjectiveProfile,
    UTILITY_COMPONENT_NAMES,
)
from Rock_AI.evaluation.pair_evaluator import PairEvaluation


@dataclass(frozen=True)
class ScoredPairUtility:
    score: float
    raw_components: dict[str, float]
    contributions: dict[str, float]
    uncertainty: float


def _gene_preservation_probability(
    evaluation: PairEvaluation,
    objective: FarmerObjectiveProfile,
) -> float:
    if objective.preserved_gene is None or objective.preserved_allele is None:
        return 0.0
    distribution = evaluation.expectation.per_gene_outcome_distributions.get(
        objective.preserved_gene
    )
    if distribution is None:
        return 0.0
    probability = 0.0
    for key, outcome_probability in distribution.allele_pair_probabilities.items():
        left, right = distribution.parse_pair_key(key)
        if objective.preserved_allele in (left, right):
            probability += outcome_probability
    return float(probability)


def score_pair_evaluation(
    evaluation: PairEvaluation,
    objective: FarmerObjectiveProfile,
) -> ScoredPairUtility:
    expectation = evaluation.expectation
    raw = dict(evaluation.explanation_fields["raw_components"])
    raw["gene_preservation"] = _gene_preservation_probability(evaluation, objective)
    standard_errors = (
        expectation.expected_child_value.standard_error,
        expectation.expected_maximum_child_value.standard_error,
        expectation.expected_survivor_count.standard_error,
    )
    uncertainty = math.sqrt(sum(value * value for value in standard_errors))
    raw["uncertainty_penalty"] = uncertainty
    weights = {
        "expected_value": objective.immediate_expected_value_weight,
        "maximum_value": objective.maximum_offspring_value_weight,
        "survivor_value": objective.survivor_count_weight,
        "genotype_diversity": objective.genotype_diversity_weight,
        "phenotype_diversity": objective.phenotype_diversity_weight,
        "rare_trait": objective.rare_trait_weight,
        "mutation_opportunity": objective.mutation_opportunity_weight,
        "gene_preservation": objective.gene_preservation_weight,
        "uncertainty_penalty": -(
            objective.risk_aversion_weight + objective.uncertainty_penalty_weight
        ),
    }
    contributions = {name: raw[name] * weights[name] for name in UTILITY_COMPONENT_NAMES}
    return ScoredPairUtility(sum(contributions.values()), raw, contributions, uncertainty)


def pair_diversity_features(
    parent_a: genetics.Rock,
    parent_b: genetics.Rock,
) -> tuple[float, float]:
    gene_names = tuple(sorted(genetics.GENE_SPECS))
    allele_differences = 0
    phenotype_differences = 0
    for name in gene_names:
        pair_a = parent_a.genotype.genes[name]
        pair_b = parent_b.genotype.genes[name]
        alleles_a = {int(pair_a.allele_a.value), int(pair_a.allele_b.value)}
        alleles_b = {int(pair_b.allele_a.value), int(pair_b.allele_b.value)}
        allele_differences += int(alleles_a != alleles_b)
        phenotype_differences += int(pair_a.phenotype != pair_b.phenotype)
    denominator = max(1, len(gene_names))
    return allele_differences / denominator, phenotype_differences / denominator
