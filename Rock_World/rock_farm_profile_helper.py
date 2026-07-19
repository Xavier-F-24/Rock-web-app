"""Data-driven farmer objectives, never hard-coded action algorithms."""

import random
from dataclasses import asdict, dataclass, replace
from enum import Enum


class FarmObjective(str, Enum):
    PLAYER = "player"
    PROFIT_TRADER = "profit_trader"
    DIVERSITY_COLLECTOR = "diversity_collector"
    MUTATION_GAMBLER = "mutation_gambler"


@dataclass(frozen=True)
class FarmProfile:
    profile_id: str
    display_name: str
    objective: FarmObjective
    profit_weight: float = 1.0
    diversity_weight: float = 1.0
    rare_trait_weight: float = 1.0
    mutation_weight: float = 0.5
    maximum_value_weight: float = 0.5
    liquidity_weight: float = 0.5
    risk_aversion_weight: float = 0.5

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["objective"] = self.objective.value
        return data


def create_default_farm_profiles() -> tuple[FarmProfile, ...]:
    return (
        FarmProfile("profit_trader", "Ledger & Stone", FarmObjective.PROFIT_TRADER, 2.5, .4, .4, .2, 1.0, 1.5, 1.0),
        FarmProfile("diversity_collector", "The Varied Quarry", FarmObjective.DIVERSITY_COLLECTOR, .5, 2.5, 2.0, .5, .7, .5, .5),
        FarmProfile("mutation_gambler", "Bright Fault Farm", FarmObjective.MUTATION_GAMBLER, .6, 1.0, 1.2, 2.5, 2.0, .3, .1),
    )


FARM_NAME_PREFIXES = (
    "Amber", "Blue", "Clever", "Dancing", "Emerald", "Far", "Golden", "Hidden",
    "Ivory", "Juniper", "Kindred", "Lucky", "Mossy", "Northern", "Old", "Quiet",
)
FARM_NAME_SUFFIXES = (
    "Boulder Farm", "Fault", "Gravel Works", "Hill Quarry", "Pebble House",
    "Rockery", "Stone Yard", "Vale", "Vein", "Works",
)


def create_farm_profiles(count: int, rng: random.Random) -> tuple[FarmProfile, ...]:
    if count < 1:
        return ()
    templates = create_default_farm_profiles()
    combinations = [f"{left} {right}" for left in FARM_NAME_PREFIXES for right in FARM_NAME_SUFFIXES]
    rng.shuffle(combinations)
    profiles = []
    for index in range(count):
        template = templates[index % len(templates)]
        profiles.append(replace(
            template,
            profile_id=f"{template.profile_id}_{index + 1}",
            display_name=combinations[index],
        ))
    return tuple(profiles)
