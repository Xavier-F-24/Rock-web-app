from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

import Rock_Genetics.rock_genetic_helper as genetics
from Rock_AI.datasets.breeding_record_helper import EncodedBreedingRules
from Rock_AI.environments.breeding_training_environment import BreedingTrainingEnvironment
from Rock_GameState.rock_game_state_helper import GameMaster


def _parents(seed=30):
    game = GameMaster(seed=seed)
    male = next(rock for rock in game.rocks.values() if rock.sex == genetics.Sex.MALE)
    female = next(rock for rock in game.rocks.values() if rock.sex == genetics.Sex.FEMALE)
    return male, female


def test_same_seed_produces_identical_records():
    parents = _parents()
    first = BreedingTrainingEnvironment(seed=700).execute_breeding(*parents).to_dict()
    second = BreedingTrainingEnvironment(seed=700).execute_breeding(*parents).to_dict()
    assert first == second


def test_different_seeds_can_produce_different_stochastic_outcomes():
    parents = _parents()
    environment = BreedingTrainingEnvironment()
    records = [environment.execute_breeding(*parents, seed=seed) for seed in range(701, 706)]
    outcomes = [
        (record.clutch_size, record.child_genotypes, record.child_statuses)
        for record in records
    ]
    assert any(outcome != outcomes[0] for outcome in outcomes[1:])


def test_rule_encoding_includes_mutation_and_split_parameters():
    rules = EncodedBreedingRules.from_config(
        {
            "mutation_chance": 0.37,
            "spore_death_chance": 0.12,
            "spore_clone_count": 5,
            "clutch_plus_one": True,
        }
    )
    record = BreedingTrainingEnvironment(seed=800).execute_breeding(*_parents(), rules=rules)

    assert record.encoded_rule_data["mutation_chance"] == 0.37
    assert record.encoded_rule_data["spore_clone_count"] == 5
    assert record.encoded_rule_data["clutch_plus_one"] is True


def test_record_children_match_children_returned_by_real_engine():
    environment = BreedingTrainingEnvironment(seed=900)
    record = environment.execute_breeding(*_parents())

    assert record.child_ids == tuple(child.id for child in environment.last_children)
    assert record.child_statuses == tuple(child.status.value for child in environment.last_children)
    for recorded, child in zip(record.child_genotypes, environment.last_children):
        actual = {
            name: [pair.allele_a.value, pair.allele_b.value]
            for name, pair in sorted(child.genotype.genes.items())
        }
        assert recorded == actual


def test_environment_does_not_mutate_global_random_state():
    random.seed(12345)
    before = random.getstate()
    BreedingTrainingEnvironment(seed=901).execute_breeding(*_parents())
    assert random.getstate() == before


def test_environment_snapshot_restore_reproduces_local_rng_and_state():
    environment = BreedingTrainingEnvironment(seed=902)
    environment.load_parents(*_parents())
    snapshot = environment.snapshot()
    expected_random_value = environment.rng.random()
    environment.state.parent_a = None

    environment.restore(snapshot)

    assert environment.state.parent_a is not None
    assert environment.rng.random() == expected_random_value


def test_training_package_import_does_not_require_streamlit():
    project_root = Path(__file__).resolve().parents[2]
    code = """
import importlib.abc
import sys

class BlockStreamlit(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'streamlit' or fullname.startswith('streamlit.'):
            raise ImportError('Streamlit import blocked for headless test')
        return None

sys.meta_path.insert(0, BlockStreamlit())
import Rock_AI
from Rock_AI.datasets.breeding_dataset_generator import BreedingDatasetGenerator
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
