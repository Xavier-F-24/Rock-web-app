"""A legal action plus its player-visible fixed-width representation."""

from dataclasses import dataclass, field

from .farmer_action import FarmerAction


@dataclass(frozen=True)
class ActionCandidate:
    action: FarmerAction
    candidate_hash: str
    values: tuple[float, ...]
    visibility_mask: tuple[bool, ...]
    feature_names: tuple[str, ...]
    generation_reasons: tuple[str, ...] = ()
    pruning_rank: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.values) != len(self.visibility_mask) or len(self.values) != len(self.feature_names):
            raise ValueError("Action candidate values, mask, and feature names must align")

    def model_values(self) -> tuple[float, ...]:
        return self.values + tuple(float(value) for value in self.visibility_mask)
