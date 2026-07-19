from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class MultiFarmEpisodeRecord:
    episode_id: str
    seed: int
    world_template: str
    initial_world: dict[str, Any]
    decisions: list[dict[str, Any]] = field(default_factory=list)
    rounds: list[dict[str, Any]] = field(default_factory=list)
    final_world: dict[str, Any] | None = None
    termination_reason: str | None = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)
