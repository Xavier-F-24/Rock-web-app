import random
import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import Rock_Genetics.rock_genetic_helper as genetics


def make_gene_pair(gene_name, allele_a=0, allele_b=0, phenotype=None, money_value=0):
    spec = genetics.GENE_SPECS[gene_name]
    return genetics.GenePair(
        allele_a=genetics.Allele(allele_a),
        allele_b=genetics.Allele(allele_b),
        name_of_gene=gene_name,
        dominance_type=spec.expression_rule,
        phenotype=phenotype,
        money_value=money_value,
    )


def make_death_pair(name, allele_a, allele_b):
    return genetics.GenePair(
        allele_a=genetics.Allele(allele_a),
        allele_b=genetics.Allele(allele_b),
        name_of_gene=name,
        dominance_type="death_genes",
        phenotype="n/a",
        money_value=0,
    )


def make_genome(overrides=None):
    overrides = overrides or {}
    genes = {}

    for gene_name in genetics.GENE_SPECS:
        allele_a, allele_b = overrides.get(gene_name, (0, 0))
        genes[gene_name] = make_gene_pair(gene_name, allele_a, allele_b)

    return genetics.Genome(genes=genes)


def make_death_genes(overrides=None):
    overrides = overrides or {}
    genes = {}

    for index, name in enumerate(genetics.GenomeFactory.death_gene_list, start=1):
        allele_a, allele_b = overrides.get(name, (index, index + 10))
        genes[name] = make_death_pair(name, allele_a, allele_b)

    return genetics.Genome(genes=genes)


def make_rock(
    rock_id=1,
    sex=genetics.Sex.MALE,
    status=genetics.RockStatus.ACTIVE,
    generation=0,
    gene_overrides=None,
    death_overrides=None,
):
    rock = genetics.Rock(
        id=rock_id,
        sex=sex,
        name=genetics.RockName(given=f"Peb{rock_id}"),
        genotype=make_genome(gene_overrides),
        death_genes=make_death_genes(death_overrides),
        generation=generation,
        status=status,
    )

    genetics.ExpressionEngine().instantiate_phenotype(rock)
    genetics.ValueCalculator().set_rock_value(rock)
    return rock


@pytest.fixture
def male_rock():
    return make_rock(rock_id=1, sex=genetics.Sex.MALE)


@pytest.fixture
def female_rock():
    return make_rock(rock_id=2, sex=genetics.Sex.FEMALE)


@pytest.fixture
def active_parent_pair(male_rock, female_rock):
    return male_rock, female_rock


@pytest.fixture
def seeded_genome_factory():
    factory = genetics.GenomeFactory()
    factory.rng = random.Random(1234)
    return factory


@pytest.fixture
def expression_engine():
    return genetics.ExpressionEngine()


@pytest.fixture
def value_calculator():
    return genetics.ValueCalculator()
