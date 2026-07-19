from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PublicWorldEventRecord:
    event_id: str
    world_turn: int
    event_type: str
    summary: str
    farm_ids: tuple[str, ...] = ()
    rock_ids: tuple[int, ...] = ()
    payload: dict[str, Any] = field(default_factory=dict)
