from __future__ import annotations

from Rock_AI.agents.breeding_agent_helper import BreedPairAction
from Rock_AI.agents.heuristic_breeding_agent import HeuristicBreedingAgent
from Rock_AI.agents.oracle_breeding_agent import OracleBreedingAgent
from Rock_AI.agents.random_breeding_agent import RandomBreedingAgent
from Rock_AI.environments.breeding_campaign_environment import (
    BreedingCampaignConfig,
    BreedingCampaignEnvironment,
)
from Rock_AI.evaluation.breeding_tournament_helper import BreedingTournament


def test_all_non_neural_agents_return_typed_actions():
    environment = BreedingCampaignEnvironment(config=BreedingCampaignConfig(max_generations=1))
    observation = environment.reset(70)
    actions = environment.legal_actions()
    for agent in (
        RandomBreedingAgent(),
        HeuristicBreedingAgent(),
        OracleBreedingAgent(trial_count=1),
    ):
        agent.reset(100)
        assert isinstance(agent.choose_action(observation, actions), (BreedPairAction,))


def test_smoke_tournament_uses_identical_starts_and_completes(tmp_path):
    tournament = BreedingTournament(
        BreedingCampaignConfig(max_generations=2, max_decisions=20)
    )
    records, summary = tournament.run(
        [RandomBreedingAgent(), HeuristicBreedingAgent()],
        episodes=2,
        seed=71,
    )
    for episode_seed in {record.initial_seed for record in records}:
        starts = [
            record.initial_farm_summary
            for record in records
            if record.initial_seed == episode_seed
        ]
        assert starts[0] == starts[1]
    assert len(records) == 4
    assert summary["episode_count"] == 2
    assert set(summary["agents"]) == {"random", "heuristic"}
    assert all(
        record.final_farm_summary["invalid_decisions_attempted"] == 0
        for record in records
    )
    output = tournament.save(tmp_path / "tournament", records, summary)
    assert (output / "episodes.jsonl").exists()
    assert (output / "summary.json").exists()
