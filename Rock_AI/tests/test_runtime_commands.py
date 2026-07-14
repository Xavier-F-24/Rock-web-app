from __future__ import annotations

from Rock_AI.agents.breeding_agent_helper import (
    BreedPairAction,
    BreedingAgent,
)
from Rock_AI.agents.random_breeding_agent import RandomBreedingAgent
from Rock_AI.environments.breeding_campaign_environment import (
    BreedingCampaignConfig,
    BreedingCampaignEnvironment,
)
from Rock_AI.runtime import (
    AgentRuntimeManager,
    CancelSessionCommand,
    ResetSessionCommand,
    RunGenerationCommand,
    StartSessionCommand,
)
from Rock_AI.runtime.runtime_event_helper import RuntimeEventType
from Rock_AI.runtime.runtime_state_helper import SessionStatus


class InvalidAgent(BreedingAgent):
    def __init__(self):
        super().__init__("invalid-agent")

    def choose_action(self, observation, legal_actions):
        return BreedPairAction(999, 1000)


def test_run_generation_stops_after_exactly_one_generation_transition():
    manager = AgentRuntimeManager()
    session = manager.create_session(
        agent=RandomBreedingAgent(),
        environment=BreedingCampaignEnvironment(
            config=BreedingCampaignConfig(max_generations=4, max_decisions=40)
        ),
        seed=110,
        session_id="generation",
    )
    manager.apply(session.session_id, StartSessionCommand())
    result = manager.apply(session.session_id, RunGenerationCommand())
    assert result.generation_advanced
    assert session.current_generation == 1
    assert result.decisions_executed >= 1


def test_invalid_action_fails_safely_and_reset_reuses_or_changes_seed():
    manager = AgentRuntimeManager()
    session = manager.create_session(agent=InvalidAgent(), seed=111, session_id="invalid")
    manager.apply(session.session_id, StartSessionCommand())
    result = manager.apply(session.session_id, RunGenerationCommand())
    assert session.status == SessionStatus.FAILED
    assert session.current_farm_state.next_rock_id == 5
    assert any(event.event_type == RuntimeEventType.SESSION_FAILED for event in result.events)
    reset = manager.apply(session.session_id, ResetSessionCommand(seed=222))
    assert reset.status_after == SessionStatus.CREATED
    assert session.environment_seed == 222


def test_illegal_status_transitions_return_structured_errors():
    manager = AgentRuntimeManager()
    session = manager.create_session(agent=RandomBreedingAgent(), seed=112, session_id="transitions")
    first = manager.apply(session.session_id, StartSessionCommand())
    second = manager.apply(session.session_id, StartSessionCommand())
    assert first.error is None
    assert second.error
    manager.apply(session.session_id, CancelSessionCommand())
    cancelled_start = manager.apply(session.session_id, StartSessionCommand())
    assert session.status == SessionStatus.CANCELLED
    assert cancelled_start.error
