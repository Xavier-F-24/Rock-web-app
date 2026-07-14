"""Typed persistent ownership boundary for one controllable agent campaign."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from Rock_AI.agents.breeding_agent_helper import BreedingAgent
from Rock_AI.datasets.pair_ranking_record_helper import FarmerObjectiveProfile
from Rock_AI.environments.breeding_campaign_environment import BreedingCampaignEnvironment

from .runtime_event_helper import RuntimeEvent
from .runtime_state_helper import AgentRuntimeConfig, SessionStatus


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class AgentSession:
    session_id: str
    agent: BreedingAgent | None
    environment: BreedingCampaignEnvironment | None
    status: SessionStatus
    runtime_configuration: AgentRuntimeConfig
    objective_profile: FarmerObjectiveProfile
    environment_seed: int
    agent_seed: int
    event_history: list[RuntimeEvent] = field(default_factory=list)
    latest_ranked_candidates: list[Any] = field(default_factory=list)
    latest_decision_explanation: Any = None
    episode_termination_reason: str | None = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    checkpoint_metadata: dict[str, Any] = field(default_factory=dict)
    failure_message: str | None = None
    replay_controller: Any = None
    initial_farm_state: Any = None

    @property
    def current_farm_state(self):
        if self.environment is None:
            return self.replay_controller.current_game if self.replay_controller else None
        return self.environment.game

    @property
    def current_generation(self) -> int:
        game = self.current_farm_state
        return int(getattr(game, "generation", 0))

    @property
    def current_decision_index(self) -> int:
        if self.environment is not None:
            return int(self.environment.state.decision_count)
        return int(getattr(self.replay_controller, "position", 0))

    @property
    def decision_history(self):
        return self.environment.state.decisions if self.environment is not None else []

    def touch(self) -> None:
        self.updated_at = _now()
