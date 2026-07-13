"""Transparent utility scoring and deterministic legal-pair ranking."""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import Rock_Breeding.rock_breeding_helper as breeding
import Rock_Genetics.rock_genetic_helper as genetics
from Rock_AI.datasets.breeding_expectation_record import BreedingExpectationRecord
from Rock_AI.datasets.breeding_record_helper import EncodedBreedingRules
from Rock_AI.evaluation.breeding_expectation_helper import BreedingExpectationEvaluator


@dataclass(frozen=True)
class PairUtilityWeights:
    expected_value_weight: float = 1.0
    maximum_value_weight: float = 0.25
    survivor_count_weight: float = 0.5
    genotype_diversity_weight: float = 2.0
    phenotype_diversity_weight: float = 2.0
    rare_trait_weight: float = 3.0
    mutation_opportunity_weight: float = 0.5


@dataclass(frozen=True)
class PairEvaluation:
    parent_ids: tuple[int | str, int | str]
    immediate_expected_value_score: float
    survivor_score: float
    diversity_score: float
    genotype_diversity_score: float
    phenotype_diversity_score: float
    rarity_score: float
    mutation_opportunity_score: float
    combined_utility_score: float
    score_components: dict[str, float]
    explanation_fields: dict[str, Any]
    expectation: BreedingExpectationRecord

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        return result


class _FarmLookup:
    def __init__(self, rocks: list[genetics.Rock]):
        self.rocks = {int(rock.id): rock for rock in rocks}

    def get_rock(self, rock_id: int) -> genetics.Rock | None:
        return self.rocks.get(int(rock_id))


def _extract_rocks(farm: object) -> list[genetics.Rock]:
    source = getattr(farm, "rocks", None)
    if source is None:
        source = getattr(farm, "rock_list", None)
    if source is None and isinstance(farm, (dict, list, tuple)):
        source = farm
    if source is None:
        raise TypeError("farm must expose rocks or rock_list, or be a rock collection")
    values = source.values() if isinstance(source, Mapping) else source
    return sorted(
        values,
        key=lambda rock: (0, int(rock.id)) if isinstance(rock.id, int) else (1, str(rock.id)),
    )


def _id_sort_key(rock_id: int | str) -> tuple[int, int | str]:
    return (0, rock_id) if isinstance(rock_id, int) else (1, str(rock_id))


class PairEvaluator:
    def __init__(self, expectation_evaluator: BreedingExpectationEvaluator | None = None):
        self.expectation_evaluator = expectation_evaluator or BreedingExpectationEvaluator()

    @staticmethod
    def _initial_allele_probabilities(gene_name: str) -> dict[int, float]:
        spec = genetics.GENE_SPECS[gene_name]
        counts = {allele: 0 for allele in spec.options}
        for roll in range(1, 21):
            allele = genetics.GenomeFactory.get_allele_from_roll(roll, spec).value
            counts[allele] += 1
        return {allele: count / 20.0 for allele, count in counts.items()}

    def _rarity_score(self, expectation: BreedingExpectationRecord) -> float:
        if not expectation.per_gene_outcome_distributions:
            return 0.0
        gene_scores: list[float] = []
        rarity_threshold = 0.10
        for gene_name, distribution in expectation.per_gene_outcome_distributions.items():
            initial = self._initial_allele_probabilities(gene_name)
            score = 0.0
            for key, probability in distribution.allele_pair_probabilities.items():
                allele_a, allele_b = distribution.parse_pair_key(key)
                allele_scores = [
                    max(0.0, rarity_threshold - initial[allele]) / rarity_threshold
                    for allele in (allele_a, allele_b)
                ]
                score += probability * sum(allele_scores) / 2.0
            gene_scores.append(score)
        return sum(gene_scores) / len(gene_scores)

    def evaluate_pair(
        self,
        parent_a: genetics.Rock,
        parent_b: genetics.Rock,
        rules: EncodedBreedingRules | Mapping[str, Any] | None = None,
        trial_count: int = 1000,
        seed: int = 0,
        weights: PairUtilityWeights | None = None,
        *,
        game: object | None = None,
    ) -> PairEvaluation:
        weights = weights or PairUtilityWeights()
        validation = breeding.BreedingMaster().validate_breeding_pair(
            parent_a,
            parent_b,
            game=game,
            warn_relatedness=False,
        )
        if not validation["valid"]:
            raise ValueError("Invalid breeding pair: " + "; ".join(validation["errors"]))
        expectation = self.expectation_evaluator.evaluate(
            parent_a,
            parent_b,
            rules=rules,
            trial_count=trial_count,
            seed=seed,
        )
        immediate = expectation.expected_child_value.mean
        survivor_score = (
            expectation.expected_survivor_count.mean
            * expectation.expected_average_surviving_child_value.mean
        )
        genotype_diversity = expectation.genotype_diversity_estimate.mean
        phenotype_diversity = expectation.phenotype_diversity_estimate.mean
        diversity = (genotype_diversity + phenotype_diversity) / 2.0
        rarity = self._rarity_score(expectation)
        mutation_opportunity = (
            expectation.mutation_probability + expectation.expected_mutations_per_child
        )
        raw_components = {
            "expected_value": immediate,
            "maximum_value": expectation.expected_maximum_child_value.mean,
            "survivor_value": survivor_score,
            "genotype_diversity": genotype_diversity,
            "phenotype_diversity": phenotype_diversity,
            "rare_trait": rarity,
            "mutation_opportunity": mutation_opportunity,
        }
        contributions = {
            "expected_value": raw_components["expected_value"] * weights.expected_value_weight,
            "maximum_value": raw_components["maximum_value"] * weights.maximum_value_weight,
            "survivor_value": raw_components["survivor_value"] * weights.survivor_count_weight,
            "genotype_diversity": raw_components["genotype_diversity"] * weights.genotype_diversity_weight,
            "phenotype_diversity": raw_components["phenotype_diversity"] * weights.phenotype_diversity_weight,
            "rare_trait": raw_components["rare_trait"] * weights.rare_trait_weight,
            "mutation_opportunity": raw_components["mutation_opportunity"] * weights.mutation_opportunity_weight,
        }
        combined = sum(contributions.values())
        return PairEvaluation(
            parent_ids=(parent_a.id, parent_b.id),
            immediate_expected_value_score=immediate,
            survivor_score=survivor_score,
            diversity_score=diversity,
            genotype_diversity_score=genotype_diversity,
            phenotype_diversity_score=phenotype_diversity,
            rarity_score=rarity,
            mutation_opportunity_score=mutation_opportunity,
            combined_utility_score=combined,
            score_components=contributions,
            explanation_fields={
                "raw_components": raw_components,
                "weights": asdict(weights),
                "formula": "combined_utility_score = sum(raw_component * matching_weight)",
                "rarity_definition": "mean expected presence of alleles with starter-roll probability below 10%",
                "survivor_definition": "expected survivor count multiplied by expected average surviving child value",
                "seed": int(seed),
                "trial_count": int(trial_count),
            },
            expectation=expectation,
        )

    def rank_pairs(
        self,
        farm: object,
        rules: EncodedBreedingRules | Mapping[str, Any] | None = None,
        trial_count: int = 1000,
        seed: int = 0,
        weights: PairUtilityWeights | None = None,
        *,
        game: object | None = None,
    ) -> list[PairEvaluation]:
        rocks = _extract_rocks(farm)
        lookup = game or (farm if hasattr(farm, "get_rock") else _FarmLookup(rocks))
        validator = breeding.BreedingMaster()
        legal_pairs = [
            pair
            for pair in itertools.combinations(rocks, 2)
            if validator.validate_breeding_pair(
                pair[0], pair[1], game=lookup, warn_relatedness=False
            )["valid"]
        ]
        evaluations = [
            self.evaluate_pair(
                parent_a,
                parent_b,
                rules=rules,
                trial_count=trial_count,
                seed=int(seed) + index,
                weights=weights,
                game=lookup,
            )
            for index, (parent_a, parent_b) in enumerate(legal_pairs)
        ]
        return sorted(
            evaluations,
            key=lambda result: (
                -result.combined_utility_score,
                _id_sort_key(result.parent_ids[0]),
                _id_sort_key(result.parent_ids[1]),
            ),
        )
