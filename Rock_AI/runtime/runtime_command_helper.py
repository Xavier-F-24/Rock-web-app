"""Explicit synchronous commands and structured runtime results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

from .runtime_event_helper import RuntimeEvent
from .runtime_state_helper import SessionStatus


@dataclass(frozen=True)
class StartSessionCommand:
    pass


@dataclass(frozen=True)
class PauseSessionCommand:
    reason: str = "user_requested"


@dataclass(frozen=True)
class ResumeSessionCommand:
    pass


@dataclass(frozen=True)
class StepSessionCommand:
    pass


@dataclass(frozen=True)
class RunGenerationCommand:
    pass


@dataclass(frozen=True)
class RunToCompletionCommand:
    pass


@dataclass(frozen=True)
class CancelSessionCommand:
    reason: str = "user_cancelled"


@dataclass(frozen=True)
class ResetSessionCommand:
    seed: int | None = None


@dataclass(frozen=True)
class SeekReplayCommand:
    position: int | str


RuntimeCommand: TypeAlias = (
    StartSessionCommand
    | PauseSessionCommand
    | ResumeSessionCommand
    | StepSessionCommand
    | RunGenerationCommand
    | RunToCompletionCommand
    | CancelSessionCommand
    | ResetSessionCommand
    | SeekReplayCommand
)


@dataclass(frozen=True)
class RuntimeCommandResult:
    session_id: str
    command_name: str
    status_before: SessionStatus
    status_after: SessionStatus
    events: tuple[RuntimeEvent, ...] = ()
    decision_explanation: Any = None
    decisions_executed: int = 0
    generation_advanced: bool = False
    should_pause: bool = False
    pause_reasons: tuple[str, ...] = ()
    termination_reason: str | None = None
    error: str | None = None

    @property
    def event(self) -> RuntimeEvent | None:
        return self.events[-1] if self.events else None
