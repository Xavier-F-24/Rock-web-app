from __future__ import annotations

import math

import Rock_Genetics.rock_genetic_helper as genetics
from Rock_AI.evaluation.genetics_evaluator import GeneticsEvaluator
from Rock_GameState.rock_game_state_helper import GameMaster


def _parents(seed=120):
    game = GameMaster(seed=seed)
    parent_a = next(rock for rock in game.rocks.values() if rock.sex == genetics.Sex.MALE)
    parent_b = next(rock for rock in game.rocks.values() if rock.sex == genetics.Sex.FEMALE)
    return game, parent_a, parent_b


def _set_gene(game, rock, gene_name, allele_a, allele_b):
    spec = genetics.GENE_SPECS[gene_name]
    rock.genotype.genes[gene_name] = genetics.GenePair(
        allele_a=genetics.Allele(allele_a),
        allele_b=genetics.Allele(allele_b),
        name_of_gene=gene_name,
        dominance_type=spec.expression_rule,
    )
    game.finalize_rock(rock)


def test_homozygous_cross_has_one_exact_outcome():
    game, parent_a, parent_b = _parents()
    _set_gene(game, parent_a, "shape", 0, 0)
    _set_gene(game, parent_b, "shape", 1, 1)

    result = GeneticsEvaluator().evaluate_gene(parent_a, parent_b, "shape")

    assert result.allele_pair_probabilities == {"0|1": 1.0}
    assert result.homozygous_probability == 0.0
    assert result.heterozygous_probability == 1.0


def test_heterozygous_cross_combines_duplicate_transmission_outcomes():
    game, parent_a, parent_b = _parents()
    _set_gene(game, parent_a, "eyes", 0, 1)
    _set_gene(game, parent_b, "eyes", 0, 1)

    result = GeneticsEvaluator().evaluate_gene(parent_a, parent_b, "eyes")

    assert result.allele_pair_probabilities == {
        "0|0": 0.25,
        "0|1": 0.25,
        "1|0": 0.25,
        "1|1": 0.25,
    }
    assert result.homozygous_probability == 0.5
    assert result.heterozygous_probability == 0.5
    assert result.phenotype_probabilities == {"double eye": 0.25, "eye": 0.5, "n/a": 0.25}


def test_codominant_phenotype_comes_from_expression_engine():
    game, parent_a, parent_b = _parents()
    _set_gene(game, parent_a, "color", 0, 0)
    _set_gene(game, parent_b, "color", 1, 1)

    result = GeneticsEvaluator().evaluate_gene(parent_a, parent_b, "color")

    assert result.allele_pair_probabilities == {"0|1": 1.0}
    assert result.phenotype_probabilities == {"silver": 1.0}


def test_mutation_adjustment_is_explicit_and_normalized():
    game, parent_a, parent_b = _parents()
    _set_gene(game, parent_a, "eyes", 0, 0)
    _set_gene(game, parent_b, "eyes", 0, 0)

    no_mutation = GeneticsEvaluator().evaluate_gene(
        parent_a, parent_b, "eyes", mutation_chance=0.0
    )
    mutation = GeneticsEvaluator().evaluate_gene(
        parent_a, parent_b, "eyes", mutation_chance=0.2
    )

    assert no_mutation.allele_pair_probabilities == {"0|0": 1.0}
    assert mutation.allele_pair_probabilities != no_mutation.allele_pair_probabilities
    assert math.isclose(sum(mutation.allele_pair_probabilities.values()), 1.0)
    assert math.isclose(sum(mutation.phenotype_probabilities.values()), 1.0)


def test_every_gene_distribution_sums_to_one_without_mutation():
    _, parent_a, parent_b = _parents()
    distributions = GeneticsEvaluator().evaluate_all_genes(
        parent_a, parent_b, mutation_chance=0.0
    )

    assert set(distributions) == set(genetics.GENE_SPECS)
    for distribution in distributions.values():
        assert math.isclose(sum(distribution.allele_pair_probabilities.values()), 1.0)
        assert math.isclose(sum(distribution.phenotype_probabilities.values()), 1.0)
