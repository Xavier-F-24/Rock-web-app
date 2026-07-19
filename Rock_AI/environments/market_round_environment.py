"""Round result records for market-sensitive simultaneous decisions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MarketRoundResult:
    world_turn: int
    acting_order: tuple[str, ...]
    action_results: tuple[object, ...]
    generation_advanced: bool
