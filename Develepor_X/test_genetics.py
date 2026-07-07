import Rock_Genetics.rock_genetic_helper as genetics

from conftest import make_rock


def test_gene_specs_are_internally_consistent():
    assert genetics.GENE_SPECS

    for gene_name, spec in genetics.GENE_SPECS.items():
        assert spec.name == gene_name
        assert spec.options
        assert spec.states

        for allele, option in spec.options.items():
            assert option.allele == allele
            assert isinstance(option.name, str)
            assert option.roll_threshold >= 0

        for state_key, state in spec.states.items():
            assert state.key == state_key
            assert isinstance(state.name, str)


def test_random_genome_includes_every_known_gene(seeded_genome_factory):
    genome = seeded_genome_factory.make_random_rock_genome()

    assert set(genome.genes) == set(genetics.GENE_SPECS)

    for gene_name, gene_pair in genome.genes.items():
        spec = genetics.GENE_SPECS[gene_name]
        assert gene_pair.name_of_gene == gene_name
        assert gene_pair.dominance_type == spec.expression_rule
        assert gene_pair.allele_a.value in spec.options
        assert gene_pair.allele_b.value in spec.options


def test_allele_rolls_map_to_valid_options(seeded_genome_factory):
    for gene_name, spec in genetics.GENE_SPECS.items():
        for roll in range(1, 21):
            allele = seeded_genome_factory.get_allele_from_roll(roll, spec)
            assert allele.value in spec.options, gene_name


def test_phenotype_instantiation_fills_gene_phenotypes(expression_engine):
    rock = make_rock(gene_overrides={"color": (0, 1), "eyes": (1, 1)})
    expression_engine.instantiate_phenotype(rock)

    assert rock.genotype.genes["color"].phenotype == "silver"
    assert rock.genotype.genes["eyes"].phenotype == "double eye"

    for gene_pair in rock.genotype.genes.values():
        assert gene_pair.phenotype is not None
        assert isinstance(gene_pair.money_value, int)


def test_value_calculation_sets_active_sell_and_score_values(value_calculator):
    rock = make_rock(gene_overrides={"eyes": (1, 1), "wings": (1, 1)})

    value_calculator.set_rock_value(rock)

    assert rock.value >= 1
    assert rock.sell_value == rock.value
    assert rock.score_value == rock.value


def test_value_calculation_zeroes_inactive_sell_and_score_values(value_calculator):
    rock = make_rock(status=genetics.RockStatus.SOLD)

    value_calculator.set_rock_value(rock)

    assert rock.value >= 1
    assert rock.sell_value == 0
    assert rock.score_value == 0
