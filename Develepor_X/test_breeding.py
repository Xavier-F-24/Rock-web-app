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
    assert result["warnings"] == []


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


class MiniGame:
    def __init__(self, rocks):
        self.rocks = rocks

    def get_rock(self, rock_id):
        return self.rocks.get(int(rock_id))


def test_relationship_validation_warns_for_siblings_but_allows_pair():
    parent_a = make_rock(rock_id=1, sex=genetics.Sex.MALE)
    parent_b = make_rock(rock_id=2, sex=genetics.Sex.FEMALE)
    sibling_a = make_rock(rock_id=3, sex=genetics.Sex.MALE)
    sibling_b = make_rock(rock_id=4, sex=genetics.Sex.FEMALE)
    sibling_a.parent_ids = [parent_a.id, parent_b.id]
    sibling_b.parent_ids = [parent_a.id, parent_b.id]
    game = MiniGame({rock.id: rock for rock in [parent_a, parent_b, sibling_a, sibling_b]})

    result = make_master().validate_breeding_pair(sibling_a, sibling_b, game=game)

    assert result["valid"] is True
    assert result["errors"] == []
    assert any("full brother and sister" in warning for warning in result["warnings"])
    assert any("R=0.5000" in warning and "F=0.2500" in warning for warning in result["warnings"])


def test_relationship_validation_warns_for_parent_child_but_allows_pair():
    parent = make_rock(rock_id=1, sex=genetics.Sex.MALE)
    child = make_rock(rock_id=2, sex=genetics.Sex.FEMALE)
    child.parent_ids = [parent.id]
    game = MiniGame({parent.id: parent, child.id: child})

    result = make_master().validate_breeding_pair(parent, child, game=game)

    assert result["valid"] is True
    assert result["errors"] == []
    assert any("father and daughter" in warning for warning in result["warnings"])
    assert any("R=0.5000" in warning and "F=0.2500" in warning for warning in result["warnings"])


def test_describe_relationship_identifies_first_cousins():
    grandparent_a = make_rock(rock_id=1, sex=genetics.Sex.MALE)
    grandparent_b = make_rock(rock_id=2, sex=genetics.Sex.FEMALE)
    sibling_parent_a = make_rock(rock_id=3, sex=genetics.Sex.MALE)
    sibling_parent_b = make_rock(rock_id=4, sex=genetics.Sex.FEMALE)
    cousin_a = make_rock(rock_id=5, sex=genetics.Sex.MALE)
    cousin_b = make_rock(rock_id=6, sex=genetics.Sex.FEMALE)

    sibling_parent_a.parent_ids = [grandparent_a.id, grandparent_b.id]
    sibling_parent_b.parent_ids = [grandparent_a.id, grandparent_b.id]
    cousin_a.parent_ids = [sibling_parent_a.id]
    cousin_b.parent_ids = [sibling_parent_b.id]
    rocks = [grandparent_a, grandparent_b, sibling_parent_a, sibling_parent_b, cousin_a, cousin_b]
    game = MiniGame({rock.id: rock for rock in rocks})

    relationship = make_master().describe_relationship(game, cousin_a, cousin_b)

    assert relationship == "first cousins (R=0.1250, F=0.0625)"


def test_describe_relationship_identifies_great_uncle():
    ancestor_a = make_rock(rock_id=1, sex=genetics.Sex.MALE)
    ancestor_b = make_rock(rock_id=2, sex=genetics.Sex.FEMALE)
    grandparent = make_rock(rock_id=3, sex=genetics.Sex.FEMALE)
    great_uncle = make_rock(rock_id=4, sex=genetics.Sex.MALE)
    parent = make_rock(rock_id=5, sex=genetics.Sex.MALE)
    grandniece = make_rock(rock_id=6, sex=genetics.Sex.FEMALE)

    grandparent.parent_ids = [ancestor_a.id, ancestor_b.id]
    great_uncle.parent_ids = [ancestor_a.id, ancestor_b.id]
    parent.parent_ids = [grandparent.id]
    grandniece.parent_ids = [parent.id]
    rocks = [ancestor_a, ancestor_b, grandparent, great_uncle, parent, grandniece]
    game = MiniGame({rock.id: rock for rock in rocks})

    relationship = make_master().describe_relationship(game, great_uncle, grandniece)

    assert relationship == "great-uncle and grandniece (R=0.1250, F=0.0625)"


def test_relationship_warning_can_be_disabled():
    parent_a = make_rock(rock_id=1, sex=genetics.Sex.MALE)
    parent_b = make_rock(rock_id=2, sex=genetics.Sex.FEMALE)
    sibling_a = make_rock(rock_id=3, sex=genetics.Sex.MALE)
    sibling_b = make_rock(rock_id=4, sex=genetics.Sex.FEMALE)
    sibling_a.parent_ids = [parent_a.id, parent_b.id]
    sibling_b.parent_ids = [parent_a.id, parent_b.id]
    game = MiniGame({rock.id: rock for rock in [parent_a, parent_b, sibling_a, sibling_b]})

    result = make_master().validate_breeding_pair(sibling_a, sibling_b, game=game, warn_relatedness=False)

    assert result["valid"] is True
    assert result["warnings"] == []
