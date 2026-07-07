import random

import pytest

import Rock_Breeding.rock_breeding_helper as breeding
import Rock_Genetics.rock_genetic_helper as genetics

from conftest import make_rock


def make_master(seed=99):
    master = breeding.BreedingMaster()
    master.rng = random.Random(seed)
    master.GenomeFactory.rng = random.Random(seed + 1)
    master.NameGenerator.rng = random.Random(seed + 2)
    return master


def test_valid_opposite_sex_active_parents_pass_validation(active_parent_pair):
    parent_a, parent_b = active_parent_pair
    result = make_master().validate_breeding_pair(parent_a, parent_b)

    assert result["valid"] is True
    assert result["errors"] == []


def test_same_rock_fails_validation(male_rock):
    result = make_master().validate_breeding_pair(male_rock, male_rock)

    assert result["valid"] is False
    assert any("cannot breed with itself" in error for error in result["errors"])


def test_same_sex_parents_fail_validation():
    parent_a = make_rock(rock_id=1, sex=genetics.Sex.MALE)
    parent_b = make_rock(rock_id=2, sex=genetics.Sex.MALE)

    result = make_master().validate_breeding_pair(parent_a, parent_b)

    assert result["valid"] is False
    assert any("opposite gender" in error for error in result["errors"])


@pytest.mark.parametrize(
    "status",
    [
        genetics.RockStatus.DEAD,
        genetics.RockStatus.SOLD,
        genetics.RockStatus.BRED,
    ],
)
def test_inactive_parent_statuses_fail_validation(status, female_rock):
    parent_a = make_rock(rock_id=1, sex=genetics.Sex.MALE, status=status)

    result = make_master().validate_breeding_pair(parent_a, female_rock)

    assert result["valid"] is False
    assert any("not breedable" in error for error in result["errors"])


def test_breed_child_from_parents_creates_child_with_expected_lineage(active_parent_pair):
    parent_a, parent_b = active_parent_pair
    child = make_master().breed_child_from_parents(
        parent_a=parent_a,
        parent_b=parent_b,
        next_id=10,
        child_generation=3,
        mutation_chance=0,
    )

    assert child.id == 10
    assert child.parent_ids == [parent_a.id, parent_b.id]
    assert child.generation == 3
    assert child.status == genetics.RockStatus.ACTIVE
    assert child.sex in {genetics.Sex.MALE, genetics.Sex.FEMALE}
    assert set(child.genotype.genes) == set(genetics.GENE_SPECS)
    assert set(child.death_genes.genes) == set(genetics.GenomeFactory.death_gene_list)


def test_breed_parent_set_returns_clutch_marks_parents_and_clears_state(active_parent_pair):
    parent_a, parent_b = active_parent_pair
    master = make_master()

    clutch = master.breed_parent_set(
        parent_a=parent_a,
        parent_b=parent_b,
        next_id=20,
        child_generation=1,
        mutation_chance=0,
        death_chance=0,
        craisen_chance=0,
        spore_death_chance=0,
        spore_clone_count=0,
    )

    assert clutch
    assert parent_a.status == genetics.RockStatus.BRED
    assert parent_b.status == genetics.RockStatus.BRED
    assert master.child_bred_for_parents == []

    for child in clutch:
        assert child.parent_ids == [parent_a.id, parent_b.id]
        assert child.generation == 1
        assert child.value >= 1


def test_maybe_kill_child_can_force_death(male_rock):
    child = make_master().maybe_kill_child(male_rock, death_chance=1)

    assert child.status == genetics.RockStatus.DEAD
    assert child.death_reason == "died after birth"


def test_maybe_kill_child_can_force_survival(male_rock):
    child = make_master().maybe_kill_child(male_rock, death_chance=0)

    assert child.status == genetics.RockStatus.ACTIVE
    assert child.death_reason is None


def test_maybe_craisen_child_can_force_craisen_when_death_gene_matches():
    child = make_rock(
        death_overrides={
            "death_gene1": (7, 7),
            "death_gene2": (2, 12),
            "death_gene3": (3, 13),
        }
    )

    child = make_master().maybe_craisen_child(child, craisen_chance=1)

    assert child.status == genetics.RockStatus.CRAISENED
    assert child.checked_craisen is True
    assert child.death_reason == "craisend up, man"


def test_maybe_craisen_child_does_not_craisen_without_matching_death_gene(male_rock):
    child = make_master().maybe_craisen_child(male_rock, craisen_chance=1)

    assert child.status == genetics.RockStatus.ACTIVE
    assert child.checked_craisen is False
