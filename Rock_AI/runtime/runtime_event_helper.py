"""Serialization-safe event stream emitted by synchronous runtime commands."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class RuntimeEventType(str, Enum):
    SESSION_STARTED = "session_started"
    SESSION_PAUSED = "session_paused"
    SESSION_RESUMED = "session_resumed"
    DECISION_STARTED = "decision_started"
    CANDIDATE_PAIRS_SCORED = "candidate_pairs_scored"
    PAIR_SELECTED = "pair_selected"
    BREEDING_EXECUTED = "breeding_executed"
    CHILDREN_CREATED = "children_created"
    MUTATION_OCCURRED = "mutation_occurred"
    ROCK_STATUS_CHANGED = "rock_status_changed"
    GENERATION_ADVANCED = "generation_advanced"
    NO_LEGAL_ACTIONS = "no_legal_actions"
    SESSION_COMPLETED = "session_completed"
    SESSION_FAILED = "session_failed"
    SESSION_CANCELLED = "session_cancelled"
    SESSION_RESET = "session_reset"
    REPLAY_SEEKED = "replay_seeked"


@dataclass(frozen=True)
class RuntimeEvent:
    session_id: str
    event_index: int
    decision_index: int
    generation: int
    event_type: RuntimeEventType
    summary: str
    payload: dict[str, Any] = field(default_factory=dict)
    rock_ids: tuple[int | str, ...] = ()
    pre_action_metrics: dict[str, float | int] | None = None
    post_action_metrics: dict[str, float | int] | None = None
    environment_seed: int | None = None
    agent_seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["event_type"] = self.event_type.value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuntimeEvent":
        values = dict(data)
        values["event_type"] = RuntimeEventType(values["event_type"])
        values["rock_ids"] = tuple(values.get("rock_ids", ()))
        return cls(**values)
