from __future__ import annotations

import math

import Rock_Genetics.rock_genetic_helper as genetics
from Rock_AI.datasets.breeding_record_helper import EncodedBreedingRules
from Rock_AI.evaluation.breeding_expectation_helper import BreedingExpectationEvaluator
from Rock_GameState.rock_game_state_helper import GameMaster


def _controlled_parents(seed=130):
    game = GameMaster(seed=seed)
    parent_a = next(rock for rock in game.rocks.values() if rock.sex == genetics.Sex.MALE)
    parent_b = next(rock for rock in game.rocks.values() if rock.sex == genetics.Sex.FEMALE)
    for rock in (parent_a, parent_b):
        for gene_name, alleles in {"eyes": (0, 1), "splitting": (0, 0)}.items():
            spec = genetics.GENE_SPECS[gene_name]
            rock.genotype.genes[gene_name] = genetics.GenePair(
                allele_a=genetics.Allele(alleles[0]),
                allele_b=genetics.Allele(alleles[1]),
                name_of_gene=gene_name,
                dominance_type=spec.expression_rule,
            )
        game.finalize_rock(rock)
    return parent_a, parent_b


def _controlled_rules(mutation_chance=0.0):
    return EncodedBreedingRules.from_config(
        {
            "mutation_chance": mutation_chance,
            "child_death_chance": 0.0,
            "craisen_chance": 0.0,
            "clutch_mean": 0.0,
            "clutch_std": 0.0,
            "max_clutch_size": 1,
            "spore_death_chance": 0.0,
            "spore_clone_count": 0,
        }
    )


def test_monte_carlo_phenotypes_converge_toward_exact_expectations():
    parent_a, parent_b = _controlled_parents()
    result = BreedingExpectationEvaluator().evaluate(
        parent_a,
        parent_b,
        rules=_controlled_rules(),
        trial_count=1000,
        seed=4000,
    )
    exact = result.per_gene_outcome_distributions["eyes"].phenotype_probabilities

    for phenotype, probability in exact.items():
        sampled = result.phenotype_probability_vector[f"eyes={phenotype}"]
        assert abs(sampled - probability) < 0.06
    assert result.expected_raw_clutch_size.mean == 1.0
    assert result.expected_survivor_count.mean == 1.0
    assert result.expected_child_value.standard_error >= 0.0


def test_mutation_statistics_are_analytical_and_reproducible():
    parent_a, parent_b = _controlled_parents()
    rules = _controlled_rules(mutation_chance=0.15)
    evaluator = BreedingExpectationEvaluator()
    first = evaluator.evaluate(parent_a, parent_b, rules=rules, trial_count=30, seed=4100)
    second = evaluator.evaluate(parent_a, parent_b, rules=rules, trial_count=30, seed=4100)

    attempts = 2 * (len(genetics.GENE_SPECS) + len(genetics.GenomeFactory.death_gene_list))
    assert math.isclose(first.expected_mutations_per_child, attempts * 0.15)
    assert first.mutation_probability > 0.0
    assert first.to_dict() == second.to_dict()
    assert first.field_methods["mutation_probability"] == "analytical"
    assert first.field_methods["expected_child_value"].startswith("monte_carlo")
