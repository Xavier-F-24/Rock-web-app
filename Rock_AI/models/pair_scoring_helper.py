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
    raw["mortality_penalty"] = getattr(
        getattr(expectation, "expected_dead_count", None), "mean", 0.0
    )
    raw["craisen_penalty"] = getattr(
        getattr(expectation, "expected_craisened_count", None), "mean", 0.0
    )
    raw["zero_survivor_penalty"] = float(
        getattr(expectation, "probability_zero_active_survivors", 0.0)
    )
    raw["relatedness_penalty"] = float(
        evaluation.explanation_fields["raw_components"].get(
            "relatedness_penalty", 0.0
        )
    )
    raw["aleatoric_risk_penalty"] = (
        float(getattr(expectation, "surviving_clutch_value_variance", 0.0)) ** 0.5
    )
    raw["epistemic_uncertainty_penalty"] = uncertainty
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
        "mortality_penalty": -objective.mortality_penalty_weight,
        "craisen_penalty": -objective.craisen_penalty_weight,
        "zero_survivor_penalty": -objective.zero_survivor_penalty_weight,
        "relatedness_penalty": -objective.relatedness_penalty_weight,
        "aleatoric_risk_penalty": -(
            objective.risk_aversion_weight
            + objective.aleatoric_risk_penalty_weight
        ),
        "epistemic_uncertainty_penalty": -(
            objective.uncertainty_penalty_weight
            + objective.epistemic_uncertainty_penalty_weight
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
