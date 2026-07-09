"""Farm profile definitions for NPC rock farms."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FarmProfile:
    """
    Stable identity and strategy preferences for an NPC farm.

    The numeric strategy fields are intentionally heuristic knobs, not learned
    weights. Farmer policy code can later use them to score breeding, buying,
    selling, and offer decisions without changing the saved identity shape.
    """

    farm_id: str
    farm_name: str
    owner_name: str
    region: str
    difficulty: str = "medium"
    personality: str = "balanced breeder"
    trait_preferences: tuple[str, ...] = field(default_factory=tuple)
    relatedness_tolerance: float = 0.125
    risk_tolerance: float = 0.5
    market_aggression: float = 0.5
    cash_reserve: int = 10
    starting_generation_offset: int = 1

    @property
    def owner_id(self) -> str:
        return f"farm:{self.farm_id}"


def create_default_farm_profiles() -> list[FarmProfile]:
    """
    Return the initial three NPC farm personalities.
    """

    return [
        FarmProfile(
            farm_id="mossy_knoll",
            farm_name="Mossy Knoll Farm",
            owner_name="Mira Gravel",
            region="North Ridge",
            difficulty="easy",
            personality="friendly collector",
            trait_preferences=("color", "eyes"),
            relatedness_tolerance=0.25,
            risk_tolerance=0.65,
            market_aggression=0.35,
            cash_reserve=8,
            starting_generation_offset=1,
        ),
        FarmProfile(
            farm_id="basalt_yard",
            farm_name="Basalt Yard",
            owner_name="Basil Basalt",
            region="Old Quarry Road",
            difficulty="medium",
            personality="value-minded breeder",
            trait_preferences=("size", "shape", "hair"),
            relatedness_tolerance=0.125,
            risk_tolerance=0.45,
            market_aggression=0.55,
            cash_reserve=14,
            starting_generation_offset=1,
        ),
        FarmProfile(
            farm_id="feldspar_house",
            farm_name="Feldspar House",
            owner_name="Orin Feldspar",
            region="Sunset Cut",
            difficulty="hard",
            personality="careful lineage planner",
            trait_preferences=("color", "eye_color", "special"),
            relatedness_tolerance=0.0625,
            risk_tolerance=0.3,
            market_aggression=0.75,
            cash_reserve=22,
            starting_generation_offset=2,
        ),
    ]
