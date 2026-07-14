from __future__ import annotations

from Rock_AI.agents.breeding_agent_helper import (
    BreedPairAction,
    StopGenerationAction,
)
from Rock_AI.agents.neural_breeding_agent import NeuralBreedingAgent
from Rock_AI.environments.breeding_campaign_environment import (
    BreedingCampaignConfig,
    BreedingCampaignEnvironment,
)
from Rock_AI.policies.neural_pair_ranking_policy import (
    PairRankingDecision,
    RankedPairDecision,
)


class FakePolicy:
    def __init__(self, score=10.0, include_invalid=False):
        self.score = score
        self.include_invalid = include_invalid

    def rank_legal_pairs(self, farm, rules, objective):
        valid = next(
            (pair for pair in ((1, 2), (1, 4), (3, 2), (3, 4)) if farm.breeding_master.validate_breeding_pair(farm.get_rock(pair[0]), farm.get_rock(pair[1]))["valid"]),
            (1, 2),
        )
        rows = []
        if self.include_invalid:
            rows.append(RankedPairDecision((999, 1000), self.score + 100))
        rows.append(RankedPairDecision(valid, self.score))
        return PairRankingDecision(tuple(rows), rows[0].parent_ids, 0.9)


def test_neural_agent_filters_invalid_policy_pairs_and_returns_typed_action():
    environment = BreedingCampaignEnvironment(config=BreedingCampaignConfig(max_generations=2))
    observation = environment.reset(50)
    agent = NeuralBreedingAgent(FakePolicy(include_invalid=True))
    agent.reset(90)
    action = agent.choose_action(observation, environment.legal_actions())
    assert isinstance(action, BreedPairAction)
    assert tuple(sorted(map(str, (action.parent_a_id, action.parent_b_id)))) in {
        tuple(sorted(map(str, pair))) for pair in observation.legal_pair_ids
    }
    result = environment.step(action, agent_name=agent.name, agent_seed=agent.seed)
    assert result.valid
    assert environment.state.invalid_decisions == 0


def test_neural_stopping_threshold_is_conservative_and_configurable():
    environment = BreedingCampaignEnvironment(config=BreedingCampaignConfig(max_generations=2))
    observation = environment.reset(51)
    default_agent = NeuralBreedingAgent(FakePolicy(score=-5.0))
    default_agent.reset(1)
    assert isinstance(default_agent.choose_action(observation, environment.legal_actions()), BreedPairAction)
    stopping_agent = NeuralBreedingAgent(FakePolicy(score=-5.0), utility_threshold=0.0)
    stopping_agent.reset(1)
    action = stopping_agent.choose_action(observation, environment.legal_actions())
    assert isinstance(action, StopGenerationAction)
    assert action.reason == "best_utility_below_threshold"


def test_environment_rejects_an_invalid_policy_action_without_breeding():
    environment = BreedingCampaignEnvironment(config=BreedingCampaignConfig(max_generations=2))
    environment.reset(52)
    initial_ids = set(environment.game.rocks)
    result = environment.step(
        BreedPairAction(999, 1000),
        agent_name="broken-policy",
        agent_seed=1,
    )
    assert not result.valid
    assert result.termination_reason == "invalid_action"
    assert set(environment.game.rocks) == initial_ids
    assert environment.state.invalid_decisions == 1
