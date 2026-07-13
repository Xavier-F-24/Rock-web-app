from __future__ import annotations

import math

import Rock_Genetics.rock_genetic_helper as genetics
from Rock_AI.datasets.breeding_record_helper import EncodedBreedingRules
from Rock_AI.evaluation.pair_evaluator import PairEvaluator, PairUtilityWeights
from Rock_GameState.rock_game_state_helper import GameMaster


def _fast_rules():
    return EncodedBreedingRules.from_config(
        {
            "mutation_chance": 0.0,
            "child_death_chance": 0.0,
            "craisen_chance": 0.0,
            "clutch_mean": 0.0,
            "clutch_std": 0.0,
            "max_clutch_size": 1,
            "spore_clone_count": 0,
        }
    )


def test_pair_score_is_decomposable():
    game = GameMaster(seed=140)
    parent_a = game.get_rock(1)
    parent_b = game.get_rock(2)
    weights = PairUtilityWeights(expected_value_weight=1.5, rare_trait_weight=4.0)

    result = PairEvaluator().evaluate_pair(
        parent_a,
        parent_b,
        rules=_fast_rules(),
        trial_count=20,
        seed=5000,
        weights=weights,
        game=game,
    )

    assert math.isclose(result.combined_utility_score, sum(result.score_components.values()))
    assert result.explanation_fields["weights"]["expected_value_weight"] == 1.5
    assert result.parent_ids == (1, 2)


def test_rankings_are_deterministic_with_stable_tie_breaking():
    game = GameMaster(seed=141)
    evaluator = PairEvaluator()
    first = evaluator.rank_pairs(game, _fast_rules(), trial_count=15, seed=5100, game=game)
    second = evaluator.rank_pairs(game, _fast_rules(), trial_count=15, seed=5100, game=game)

    assert [result.to_dict() for result in first] == [result.to_dict() for result in second]
    assert all(
        first[index].combined_utility_score >= first[index + 1].combined_utility_score
        for index in range(len(first) - 1)
    )


def test_rank_pairs_excludes_invalid_pairs():
    game = GameMaster(seed=142)
    game.get_rock(2).change_status(genetics.RockStatus.BRED)

    results = PairEvaluator().rank_pairs(
        game,
        _fast_rules(),
        trial_count=10,
        seed=5200,
        game=game,
    )

    assert results
    assert all(2 not in result.parent_ids for result in results)
    for result in results:
        parent_a = game.get_rock(result.parent_ids[0])
        parent_b = game.get_rock(result.parent_ids[1])
        assert game.breeding_master.validate_breeding_pair(parent_a, parent_b)["valid"]
