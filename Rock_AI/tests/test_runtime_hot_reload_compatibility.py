from __future__ import annotations

import pytest

from Rock_AI.agents.random_breeding_agent import RandomBreedingAgent
from Rock_AI.runtime import AgentRuntimeManager, StartSessionCommand, StepSessionCommand


class ReloadedAgentProxy:
    """Structural stand-in for an agent class recreated by Streamlit hot reload."""

    def __init__(self):
        self.delegate = RandomBreedingAgent()
        self.objective_profile = self.delegate.objective_profile
        self.last_decision_context = {}

    @property
    def name(self):
        return self.delegate.name

    def reset(self, seed):
        self.delegate.reset(seed)
        self.last_decision_context = {}

    def choose_action(self, observation, legal_actions):
        action = self.delegate.choose_action(observation, legal_actions)
        self.last_decision_context = self.delegate.last_decision_context
        return action

    def configuration(self):
        return self.delegate.configuration()


def test_runtime_accepts_structurally_valid_agent_after_hot_reload():
    manager = AgentRuntimeManager()
    session = manager.create_session(agent=ReloadedAgentProxy(), seed=912)
    manager.apply(session.session_id, StartSessionCommand())
    result = manager.apply(session.session_id, StepSessionCommand())
    assert result.decisions_executed == 1
    assert result.error is None


def test_runtime_still_rejects_incomplete_agent_objects():
    with pytest.raises(TypeError, match="choose_action"):
        AgentRuntimeManager().create_session(agent=object(), seed=913)
