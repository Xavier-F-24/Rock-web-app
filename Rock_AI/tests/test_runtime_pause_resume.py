from __future__ import annotations

from Rock_AI.agents.random_breeding_agent import RandomBreedingAgent
from Rock_AI.datasets.breeding_record_helper import EncodedBreedingRules
from Rock_AI.environments.breeding_campaign_environment import (
    BreedingCampaignConfig,
    BreedingCampaignEnvironment,
)
from Rock_AI.runtime import (
    AgentRuntimeManager,
    PauseSessionCommand,
    ResumeSessionCommand,
    RunGenerationCommand,
    StartSessionCommand,
    StepSessionCommand,
)
from Rock_AI.runtime.runtime_speed_helper import RuntimeSpeedConfig
from Rock_AI.runtime.runtime_state_helper import AgentRuntimeConfig, SessionStatus


def test_pause_blocks_bulk_execution_but_documented_manual_step_is_allowed():
    manager = AgentRuntimeManager()
    session = manager.create_session(agent=RandomBreedingAgent(), seed=120, session_id="pause")
    manager.apply(session.session_id, StartSessionCommand())
    manager.apply(session.session_id, PauseSessionCommand())
    before = session.current_decision_index
    blocked = manager.apply(session.session_id, RunGenerationCommand())
    assert blocked.error
    assert session.current_decision_index == before
    stepped = manager.apply(session.session_id, StepSessionCommand())
    assert stepped.decisions_executed == 1
    assert session.current_decision_index == before + 1
    assert session.status == SessionStatus.WAITING_FOR_STEP
    manager.apply(session.session_id, PauseSessionCommand())
    resumed = manager.apply(session.session_id, ResumeSessionCommand())
    assert resumed.status_after == SessionStatus.WAITING_FOR_STEP


def test_pause_on_mutation_returns_recommendation_without_sleeping():
    rules = EncodedBreedingRules.from_config(
        {
            "mutation_chance": 1.0,
            "child_death_chance": 0.0,
            "craisen_chance": 0.0,
            "clutch_mean": 1.0,
            "clutch_std": 0.0,
            "max_clutch_size": 1,
        }
    )
    manager = AgentRuntimeManager()
    session = manager.create_session(
        agent=RandomBreedingAgent(),
        environment=BreedingCampaignEnvironment(
            config=BreedingCampaignConfig(max_generations=2, max_pairs_per_generation=1)
        ),
        runtime_configuration=AgentRuntimeConfig(
            speed=RuntimeSpeedConfig(pause_on_mutation=True)
        ),
        rules=rules,
        seed=121,
        session_id="mutation-pause",
    )
    manager.apply(session.session_id, StartSessionCommand())
    result = manager.apply(session.session_id, StepSessionCommand())
    assert result.should_pause
    assert "mutation_occurred" in result.pause_reasons
    assert session.status == SessionStatus.PAUSED
