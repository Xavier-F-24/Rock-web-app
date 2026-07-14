"""Session status and serializable runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .runtime_speed_helper import RuntimeSpeedConfig


class SessionStatus(str, Enum):
    CREATED = "created"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    WAITING_FOR_STEP = "waiting_for_step"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_SESSION_STATUSES = {
    SessionStatus.COMPLETED,
    SessionStatus.FAILED,
    SessionStatus.CANCELLED,
}


@dataclass(frozen=True)
class AgentRuntimeConfig:
    speed: RuntimeSpeedConfig = RuntimeSpeedConfig()
    retain_top_candidates: int = 5
    allow_step_while_paused: bool = True

    def __post_init__(self) -> None:
        if self.retain_top_candidates <= 0:
            raise ValueError("retain_top_candidates must be positive")

    def to_dict(self) -> dict:
        return {
            "speed": self.speed.to_dict(),
            "retain_top_candidates": self.retain_top_candidates,
            "allow_step_while_paused": self.allow_step_while_paused,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentRuntimeConfig":
        return cls(
            speed=RuntimeSpeedConfig.from_dict(data.get("speed", {})),
            retain_top_candidates=int(data.get("retain_top_candidates", 5)),
            allow_step_while_paused=bool(data.get("allow_step_while_paused", True)),
        )
