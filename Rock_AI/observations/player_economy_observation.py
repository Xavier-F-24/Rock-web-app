from dataclasses import dataclass

from .player_inventory_observation import PlayerInventoryObservation
from .player_market_observation import PlayerMarketObservation
from .player_offer_observation import PlayerOfferObservation
from .player_opponent_observation import PublicFarmObservation
from .player_potion_observation import PlayerPotionObservation


ECONOMY_OBSERVATION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PlayerEconomyObservation:
    schema_version: int
    actor_farm_id: str
    world_turn: int
    generation: int
    inventory: PlayerInventoryObservation
    market: PlayerMarketObservation
    opponents: tuple[PublicFarmObservation, ...]
    potions: PlayerPotionObservation
    offers: PlayerOfferObservation
    public_rule_version: str
    observation_hash: str
