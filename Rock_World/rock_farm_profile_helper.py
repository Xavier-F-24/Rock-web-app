"""Data-driven farmer objectives, never hard-coded action algorithms."""

from dataclasses import asdict, dataclass
from enum import Enum


class FarmObjective(str, Enum):
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
