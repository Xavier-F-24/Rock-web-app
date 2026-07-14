from __future__ import annotations

from Rock_AI.agents.random_breeding_agent import RandomBreedingAgent
from Rock_AI.environments.breeding_campaign_environment import (
    BreedingCampaignConfig,
    BreedingCampaignEnvironment,
)
from Rock_AI.evaluation.breeding_agent_evaluator import BreedingAgentEvaluator
from Rock_AI.evaluation.breeding_agent_metrics import calculate_farm_metrics
from Rock_AI.runtime import (
    AgentRuntimeManager,
    RunToCompletionCommand,
    StartSessionCommand,
    StepSessionCommand,
)
from Rock_AI.runtime.runtime_state_helper import SessionStatus


def test_session_lifecycle_and_one_step_means_one_agent_decision():
    manager = AgentRuntimeManager()
    session = manager.create_session(
        agent=RandomBreedingAgent(), seed=100, session_id="one-step"
    )
    assert session.status == SessionStatus.CREATED
    assert session.current_generation == 0
    manager.apply(session.session_id, StartSessionCommand())
    before = session.current_decision_index
    result = manager.apply(session.session_id, StepSessionCommand())
    assert result.decisions_executed == 1
    assert session.current_decision_index == before + 1
    assert result.decision_explanation.selected_parent_ids is not None
    assert result.event is not None
    assert session.status == SessionStatus.WAITING_FOR_STEP


def test_run_to_completion_matches_direct_environment_execution():
    config = BreedingCampaignConfig(max_generations=3, max_decisions=30)
    manager = AgentRuntimeManager()
    session = manager.create_session(
        agent=RandomBreedingAgent(),
        environment=BreedingCampaignEnvironment(config=config),
        seed=101,
        session_id="runtime-completion",
    )
    manager.apply(session.session_id, StartSessionCommand())
    manager.apply(session.session_id, RunToCompletionCommand())
    direct = BreedingAgentEvaluator(config).run_episode(RandomBreedingAgent(), seed=101)
    assert session.status == SessionStatus.COMPLETED
    assert calculate_farm_metrics(session.current_farm_state) == {
        name: direct.final_farm_summary[name]
        for name in calculate_farm_metrics(session.current_farm_state)
    }
    assert [row.selected_action for row in session.decision_history] == [
        row.selected_action for row in direct.decisions
    ]
