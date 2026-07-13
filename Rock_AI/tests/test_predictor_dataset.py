from __future__ import annotations

import numpy as np

from Rock_AI.datasets.predictor_dataset_generator import (
    PROCEDURAL_PROFILES,
    PredictorDatasetGenerator,
)
from Rock_AI.training.training_config_helper import TrainingDataConfig
from Rock_GameState.rock_game_state_helper import GameMaster


def _config(**overrides):
    values = {
        "number_of_parent_pairs": 6,
        "trials_per_pair": 2,
        "seed": 6000,
        "mutation_chance_range": (0.0, 0.2),
        "death_chance_range": (0.0, 0.1),
        "value_thresholds": (5.0, 10.0),
    }
    values.update(overrides)
    return TrainingDataConfig(**values)


def test_procedural_generation_covers_edge_profiles_and_fixed_shapes():
    generator = PredictorDatasetGenerator(_config())
    examples = generator.generate_procedural_examples()

    assert len(examples) == 6
    assert {example.metadata["procedural_profile"] for example in examples} == set(
        PROCEDURAL_PROFILES
    )
    parent_widths = {len(example.parent_a_features) for example in examples}
    target_widths = {len(example.target_vector(generator.target_schema)) for example in examples}
    assert len(parent_widths) == 1
    assert target_widths == {len(generator.target_schema.target_names)}
    assert all(np.isfinite(example.target_vector(generator.target_schema)).all() for example in examples)


def test_generation_is_reproducible_and_samples_rule_ranges():
    first_generator = PredictorDatasetGenerator(_config(number_of_parent_pairs=2))
    second_generator = PredictorDatasetGenerator(_config(number_of_parent_pairs=2))
    first = first_generator.generate_procedural_examples()
    second = second_generator.generate_procedural_examples()

    assert [example.target_vector(first_generator.target_schema).tolist() for example in first] == [
        example.target_vector(second_generator.target_schema).tolist() for example in second
    ]
    assert [example.metadata for example in first] == [example.metadata for example in second]
    for example in first:
        rules = example.metadata["rule_encoding"]
        assert 0.0 <= rules["mutation_chance"] <= 0.2
        assert 0.0 <= rules["child_death_chance"] <= 0.1


def test_farm_and_historical_sources_are_supported():
    config = _config(number_of_parent_pairs=2, trials_per_pair=1)
    generator = PredictorDatasetGenerator(config)
    game = GameMaster(seed=6010)

    farm_examples = generator.generate_from_farm(game, pair_count=2, game=game)
    historical_examples = PredictorDatasetGenerator(config).generate_from_historical_rocks(
        list(game.rocks.values()), pair_count=2
    )

    assert len(farm_examples) == 2
    assert len(historical_examples) == 2
    assert {example.metadata["parent_source_type"] for example in farm_examples} == {"farm"}
    assert {example.metadata["parent_source_type"] for example in historical_examples} == {
        "historical"
    }


def test_context_features_can_be_disabled():
    config = _config(
        number_of_parent_pairs=1,
        trials_per_pair=1,
        include_context_features=False,
    )
    example = PredictorDatasetGenerator(config).generate_procedural_examples()[0]
    assert example.context_features.shape == (0,)
