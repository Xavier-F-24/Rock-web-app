"""Persistent, synchronous control plane for headless Rock AI agents."""

from .agent_session_helper import AgentSession
from .runtime_command_helper import (
    CancelSessionCommand,
    PauseSessionCommand,
    ResetSessionCommand,
    ResumeSessionCommand,
    RunGenerationCommand,
    RunToCompletionCommand,
    SeekReplayCommand,
    StartSessionCommand,
    StepSessionCommand,
)
from .runtime_state_helper import SessionStatus

__all__ = [
    "AgentRuntimeManager",
    "AgentSession",
    "CancelSessionCommand",
    "PauseSessionCommand",
    "ResetSessionCommand",
    "ResumeSessionCommand",
    "RunGenerationCommand",
    "RunToCompletionCommand",
    "SeekReplayCommand",
    "SessionStatus",
    "StartSessionCommand",
    "StepSessionCommand",
]


def __getattr__(name: str):
    if name == "AgentRuntimeManager":
        from .agent_runtime_manager import AgentRuntimeManager

        return AgentRuntimeManager
    raise AttributeError(name)
