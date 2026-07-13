from __future__ import annotations

import json

import Rock_Genetics.rock_genetic_helper as genetics
from Rock_AI.datasets.breeding_dataset_generator import BreedingDatasetGenerator
from Rock_GameState.rock_game_state_helper import GameMaster


def _pair(game: GameMaster):
    male = next(rock for rock in game.rocks.values() if rock.sex == genetics.Sex.MALE)
    female = next(rock for rock in game.rocks.values() if rock.sex == genetics.Sex.FEMALE)
    return male, female


def test_dataset_generation_uses_sequential_reproducible_seeds():
    pair = _pair(GameMaster(seed=40))
    generator = BreedingDatasetGenerator(seed=1000)
    first = generator.generate_from_parent_pairs([pair], trials_per_pair=3)
    second = generator.generate_from_parent_pairs([pair], trials_per_pair=3)

    assert [record.random_seed for record in first] == [1000, 1001, 1002]
    assert [record.to_dict() for record in first] == [record.to_dict() for record in second]


def test_valid_pair_sampling_is_reproducible():
    game = GameMaster(seed=41)
    generator = BreedingDatasetGenerator(seed=1001)
    first = generator.sample_valid_parent_pairs(game, count=2, seed=77, game=game)
    second = generator.sample_valid_parent_pairs(game, count=2, seed=77, game=game)
    assert [(a.id, b.id) for a, b in first] == [(a.id, b.id) for a, b in second]
    assert all(game.breeding_master.validate_breeding_pair(a, b)["valid"] for a, b in first)


def test_jsonl_export_contains_only_serializable_record_data(tmp_path):
    game = GameMaster(seed=42)
    records = BreedingDatasetGenerator(seed=1100).generate_from_parent_pairs(
        [_pair(game)], trials_per_pair=2
    )
    output = BreedingDatasetGenerator.write_jsonl(records, tmp_path / "records.jsonl")
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 2
    assert rows[0]["random_seed"] == 1100
    assert isinstance(rows[0]["child_genotypes"], list)
    assert "encoded_parent_data" in rows[0]
