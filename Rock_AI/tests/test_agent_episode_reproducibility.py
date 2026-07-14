from __future__ import annotations

from Rock_AI.agents.random_breeding_agent import RandomBreedingAgent
from Rock_AI.environments.breeding_campaign_environment import BreedingCampaignConfig
from Rock_AI.evaluation.breeding_agent_evaluator import BreedingAgentEvaluator
from Rock_AI.logging.episode_storage_helper import load_episode_records, save_episode_records


def _decision_signature(record):
    return [
        (
            decision.selected_action,
            decision.selected_parent_ids,
            decision.resulting_child_ids,
            decision.resulting_child_values,
            decision.mutation_outcomes,
        )
        for decision in record.decisions
    ]


def test_random_episode_and_recorded_replay_are_reproducible(tmp_path):
    evaluator = BreedingAgentEvaluator(
        BreedingCampaignConfig(max_generations=3, max_decisions=30)
    )
    first = evaluator.run_episode(RandomBreedingAgent(), seed=60)
    second = evaluator.run_episode(RandomBreedingAgent(), seed=60)
    assert first.final_farm_summary == second.final_farm_summary
    assert _decision_signature(first) == _decision_signature(second)
    assert any(
        decision.selected_parent_ids and decision.resulting_child_ids
        for decision in first.decisions
    )

    destination = save_episode_records(tmp_path / "episode.jsonl", [first])
    loaded = load_episode_records(destination)[0]
    replayed = evaluator.replay_episode(loaded)
    assert replayed.final_farm_summary == first.final_farm_summary
    assert replayed.termination_reason == first.termination_reason
    assert _decision_signature(replayed) == _decision_signature(first)
