"""Exact ordinary-gene inheritance using the game's expression engine."""

from __future__ import annotations

import copy
from collections import defaultdict

import Rock_Genetics.rock_genetic_helper as genetics
from Rock_AI.datasets.breeding_expectation_record import GeneOutcomeDistribution
from Rock_AI.representations.encoding_schema_helper import EncodingSchema, get_default_encoding_schema


class GeneticsEvaluator:
    def __init__(self, schema: EncodingSchema | None = None):
        self.schema = schema or get_default_encoding_schema()
        self.expression_engine = genetics.ExpressionEngine()

    @staticmethod
    def _transmission_probabilities(
        pair: genetics.GenePair,
        possible_alleles: tuple[int, ...],
        mutation_chance: float,
    ) -> dict[int, float]:
        probabilities: dict[int, float] = defaultdict(float)
        for source in pair.alleles:
            source_value = int(source.value)
            source_probability = 0.5
            alternatives = tuple(value for value in possible_alleles if value != source_value)
            if not alternatives:
                probabilities[source_value] += source_probability
                continue
            retained_probability = source_probability * (1.0 - mutation_chance)
            if retained_probability:
                probabilities[source_value] += retained_probability
            each_mutation = source_probability * mutation_chance / len(alternatives)
            if each_mutation:
                for alternative in alternatives:
                    probabilities[alternative] += each_mutation
        return dict(sorted((key, value) for key, value in probabilities.items() if value > 0.0))

    @staticmethod
    def _cross(
        parent_a: dict[int, float],
        parent_b: dict[int, float],
    ) -> dict[str, float]:
        outcomes: dict[str, float] = defaultdict(float)
        for allele_a, probability_a in parent_a.items():
            for allele_b, probability_b in parent_b.items():
                probability = probability_a * probability_b
                if not probability:
                    continue
                key = GeneOutcomeDistribution.pair_key(allele_a, allele_b)
                outcomes[key] += probability
        return dict(sorted(outcomes.items()))

    def _phenotype_for_pair(
        self,
        template: genetics.Rock,
        gene_name: str,
        allele_a: int,
        allele_b: int,
        sex: genetics.Sex,
    ) -> str:
        rock = copy.deepcopy(template)
        rock.sex = sex
        spec = genetics.GENE_SPECS[gene_name]
        rock.genotype.genes[gene_name] = genetics.GenePair(
            allele_a=genetics.Allele(allele_a),
            allele_b=genetics.Allele(allele_b),
            name_of_gene=gene_name,
            dominance_type=spec.expression_rule,
        )
        self.expression_engine.instantiate_phenotype(rock)
        phenotype = rock.genotype.genes[gene_name].phenotype
        return "<missing>" if phenotype is None else str(phenotype)

    def _phenotype_probabilities(
        self,
        template: genetics.Rock,
        gene_name: str,
        pair_probabilities: dict[str, float],
    ) -> dict[str, float]:
        probabilities: dict[str, float] = defaultdict(float)
        sexes = tuple(genetics.Sex)
        sex_probability = 1.0 / len(sexes)
        for key, pair_probability in pair_probabilities.items():
            allele_a, allele_b = GeneOutcomeDistribution.parse_pair_key(key)
            for sex in sexes:
                phenotype = self._phenotype_for_pair(template, gene_name, allele_a, allele_b, sex)
                probabilities[phenotype] += pair_probability * sex_probability
        return dict(sorted(probabilities.items()))

    def evaluate_gene(
        self,
        parent_a: genetics.Rock,
        parent_b: genetics.Rock,
        gene_name: str,
        *,
        mutation_chance: float = 0.0,
    ) -> GeneOutcomeDistribution:
        if gene_name not in genetics.GENE_SPECS:
            raise KeyError(f"Unknown gene: {gene_name}")
        if not 0.0 <= mutation_chance <= 1.0:
            raise ValueError("mutation_chance must be between 0 and 1")
        try:
            pair_a = parent_a.genotype.genes[gene_name]
            pair_b = parent_b.genotype.genes[gene_name]
        except KeyError as exc:
            raise ValueError(f"Both parents must contain gene {gene_name!r}") from exc

        possible_alleles = tuple(sorted(genetics.GENE_SPECS[gene_name].options))
        no_mutation_a = self._transmission_probabilities(pair_a, possible_alleles, 0.0)
        no_mutation_b = self._transmission_probabilities(pair_b, possible_alleles, 0.0)
        adjusted_a = self._transmission_probabilities(pair_a, possible_alleles, mutation_chance)
        adjusted_b = self._transmission_probabilities(pair_b, possible_alleles, mutation_chance)
        no_mutation_pairs = self._cross(no_mutation_a, no_mutation_b)
        adjusted_pairs = self._cross(adjusted_a, adjusted_b)
        no_mutation_phenotypes = self._phenotype_probabilities(
            parent_a, gene_name, no_mutation_pairs
        )
        adjusted_phenotypes = self._phenotype_probabilities(
            parent_a, gene_name, adjusted_pairs
        )
        selected_pairs = adjusted_pairs if mutation_chance else no_mutation_pairs
        selected_phenotypes = adjusted_phenotypes if mutation_chance else no_mutation_phenotypes
        homozygous = sum(
            probability
            for key, probability in selected_pairs.items()
            if len(set(GeneOutcomeDistribution.parse_pair_key(key))) == 1
        )
        return GeneOutcomeDistribution(
            gene_name=gene_name,
            allele_pair_probabilities=selected_pairs,
            phenotype_probabilities=selected_phenotypes,
            homozygous_probability=homozygous,
            heterozygous_probability=1.0 - homozygous,
            non_mutation_allele_pair_probabilities=no_mutation_pairs,
            non_mutation_phenotype_probabilities=no_mutation_phenotypes,
            mutation_adjusted_allele_pair_probabilities=adjusted_pairs,
            mutation_adjusted_phenotype_probabilities=adjusted_phenotypes,
            mutation_chance=mutation_chance,
        )

    def evaluate_all_genes(
        self,
        parent_a: genetics.Rock,
        parent_b: genetics.Rock,
        *,
        mutation_chance: float = 0.0,
    ) -> dict[str, GeneOutcomeDistribution]:
        return {
            gene_name: self.evaluate_gene(
                parent_a,
                parent_b,
                gene_name,
                mutation_chance=mutation_chance,
            )
            for gene_name in self.schema.gene_names
        }
