from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TransactionRecord:
    transaction_id: str
    action_hash: str
    world_turn: int
    actor_farm_id: str
    action_type: str
    success: bool
    money_changes: tuple[tuple[str, int], ...] = ()
    rock_transfers: tuple[tuple[int, str, str], ...] = ()
    payload: dict[str, Any] = field(default_factory=dict)
