from __future__ import annotations

import copy

import Rock_Genetics.rock_genetic_helper as genetics
from Rock_GameState.rock_game_state_helper import GameMaster
from Rock_AI.agents.breeding_agent_helper import BreedPairAction, StopGenerationAction
from Rock_AI.environments.breeding_campaign_environment import (
    BreedingCampaignConfig,
    BreedingCampaignEnvironment,
)


def _rock_state(game):
    return [
        (
            rock.id,
            rock.status.value,
            tuple(rock.parent_ids),
            rock.value,
            tuple(
                (name, pair.allele_a.value, pair.allele_b.value)
                for name, pair in sorted(rock.genotype.genes.items())
            ),
        )
        for rock in game.rocks.values()
    ]


def test_campaign_reset_and_snapshot_restore_are_reproducible():
    environment = BreedingCampaignEnvironment(seed=40, config=BreedingCampaignConfig(max_generations=2))
    first = environment.reset(40)
    first_state = _rock_state(first.farm)
    second = environment.reset(40)
    assert _rock_state(second.farm) == first_state

    pair = second.legal_pair_ids[0]
    environment.step(BreedPairAction(*pair), agent_name="test", agent_seed=99)
    snapshot = environment.snapshot()
    first_result = environment.step(StopGenerationAction(), agent_name="test", agent_seed=99)
    first_children = _rock_state(environment.game)
    environment.restore(snapshot)
    second_result = environment.step(StopGenerationAction(), agent_name="test", agent_seed=99)
    assert [child.id for child in first_result.children] == [child.id for child in second_result.children]
    assert _rock_state(environment.game) == first_children


def test_zero_legal_pairs_terminate_cleanly():
    game = GameMaster(seed=41)
    for rock in game.rocks.values():
        rock.sex = genetics.Sex.MALE
    environment = BreedingCampaignEnvironment(config=BreedingCampaignConfig(max_generations=2))
    environment.reset(41, initial_farm=game)
    assert environment.state.terminated
    assert environment.state.termination_reason == "no_legal_pairs"
    assert environment.legal_actions() == ()


def test_maximum_decision_limit_flushes_pending_breed_and_terminates():
    environment = BreedingCampaignEnvironment(
        config=BreedingCampaignConfig(max_decisions=1, max_generations=5)
    )
    observation = environment.reset(42)
    result = environment.step(
        BreedPairAction(*observation.legal_pair_ids[0]),
        agent_name="limit",
        agent_seed=7,
    )
    assert result.terminated
    assert result.termination_reason == "maximum_decisions_reached"
    assert environment.state.decisions[0].resulting_child_ids
    assert environment.state.invalid_decisions == 0
