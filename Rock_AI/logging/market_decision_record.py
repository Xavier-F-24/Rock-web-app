from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class MarketDecisionRecord:
    episode_id: str
    world_turn: int
    farm_id: str
    observation_hash: str
    legal_action_count: int
    selected_action_hash: str
    selected_action: dict[str, Any]
    ranked_candidates: tuple[dict[str, Any], ...] = ()
    result: dict[str, Any] = field(default_factory=dict)
    recurrent_state_before: dict[str, Any] | None = None
    recurrent_state_after: dict[str, Any] | None = None

    def to_dict(self):
        return asdict(self)
