import pytest

import Rock_Genetics.rock_genetic_helper as genetics

from conftest import make_gene_pair, make_rock


def test_rock_name_full_name_combines_optional_parts():
    name = genetics.RockName(
        honorific="Lady",
        given="Pebble",
        family="Moonstone",
        epithet="the Bright",
    )

    assert name.full_name == "Lady Pebble Moonstone the Bright"


def test_rock_name_full_name_omits_missing_optional_parts():
    name = genetics.RockName(given="Pebble")

    assert name.full_name == "Pebble"


def test_rock_is_active_reflects_status():
    rock = make_rock(status=genetics.RockStatus.ACTIVE)
    assert rock.is_active

    rock.change_status(genetics.RockStatus.DEAD)
    assert not rock.is_active


def test_change_status_accepts_rock_status_values():
    rock = make_rock()

    rock.change_status(genetics.RockStatus.BRED)

    assert rock.status == genetics.RockStatus.BRED


def test_change_status_rejects_non_rock_status_values():
    rock = make_rock()

    with pytest.raises(TypeError):
        rock.change_status("dead")


def test_genome_get_gene_returns_expected_pair():
    gene_pair = make_gene_pair("shape", 0, 1)
    genome = genetics.Genome(genes={"shape": gene_pair})

    assert genome.get_gene("shape") is gene_pair
