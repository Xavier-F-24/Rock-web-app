"""Structured authoritative result for one submitted action."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ActionResult:
    success: bool
    action_hash: str
    transaction_id: str
    summary: str
    actor_farm_id: str
    world_turn: int
    public_payload: dict[str, Any] = field(default_factory=dict)
    private_payload: dict[str, Any] = field(default_factory=dict)
    affected_rock_ids: tuple[int, ...] = ()
    error_code: str | None = None
    idempotent_replay: bool = False
