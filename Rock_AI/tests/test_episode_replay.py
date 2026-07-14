from __future__ import annotations

import copy

from Rock_AI.agents.random_breeding_agent import RandomBreedingAgent
from Rock_AI.environments.breeding_campaign_environment import BreedingCampaignConfig
from Rock_AI.evaluation.breeding_agent_evaluator import BreedingAgentEvaluator
from Rock_AI.replay.episode_replay_helper import EpisodeReplay
from Rock_AI.runtime import AgentRuntimeManager, SeekReplayCommand


def test_replay_matches_each_decision_and_supports_bidirectional_cursor():
    record = BreedingAgentEvaluator(
        BreedingCampaignConfig(max_generations=2, max_decisions=20)
    ).run_episode(RandomBreedingAgent(), seed=140)
    replay = EpisodeReplay.from_episode_record(record)
    assert replay.validation.valid
    assert len(replay.frames) == len(record.decisions) + 1
    assert replay.first().position == 0
    assert replay.last().position == len(record.decisions)
    assert replay.previous().position == len(record.decisions) - 1
    assert replay.next().position == len(record.decisions)


def test_replay_divergence_is_reported_and_runtime_seek_works():
    record = BreedingAgentEvaluator(
        BreedingCampaignConfig(max_generations=2, max_decisions=20)
    ).run_episode(RandomBreedingAgent(), seed=141)
    changed = copy.deepcopy(record)
    changed.decisions[0].immediate_post_action_farm_metrics["rock_count"] += 1
    divergent = EpisodeReplay.from_episode_record(changed)
    assert not divergent.validation.valid
    assert any(row.field_name == "rock_count" for row in divergent.validation.divergences)

    manager = AgentRuntimeManager()
    session = manager.build_replay_session(record, session_id="replay")
    result = manager.apply(session.session_id, SeekReplayCommand("last"))
    assert result.error is None
    assert session.replay_controller.position == len(record.decisions)
    manager.apply(session.session_id, SeekReplayCommand("first"))
    assert session.replay_controller.position == 0
