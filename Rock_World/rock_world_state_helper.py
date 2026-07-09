"""Core world-state dataclasses for NPC farms and player-facing trade."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import Rock_Genetics.rock_genetic_helper as genetics
from Rock_GameState.rock_game_state_helper import Inventory, QueuedPair
from Rock_World.rock_farm_profile_helper import FarmProfile


PLAYER_OWNER_ID = "player"
FARM_OWNER_PREFIX = "farm:"


def farm_owner_id(farm_id: str) -> str:
    return f"{FARM_OWNER_PREFIX}{farm_id}"


def is_farm_owner(owner_id: str) -> bool:
    return str(owner_id).startswith(FARM_OWNER_PREFIX)


@dataclass
class FarmState:
    """
    Mutable state for one NPC farm.

    Rock IDs are stored in the farm-local ``rocks`` dictionary for now, but the
    world layer also tracks a global ``next_world_rock_id`` so future creation
    and transfer helpers can guarantee no duplicate IDs across all owners.
    """

    profile: FarmProfile
    rocks: dict[int, genetics.Rock] = field(default_factory=dict)
    inventory: Inventory = field(default_factory=Inventory)
    generation: int = 0
    next_rock_id: int = 1
    breeding_queue: list[QueuedPair] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    active: bool = True

    @property
    def farm_id(self) -> str:
        return self.profile.farm_id

    @property
    def owner_id(self) -> str:
        return self.profile.owner_id

    @property
    def money(self) -> int:
        return self.inventory.money

    @money.setter
    def money(self, value: int) -> None:
        self.inventory.money = int(value)

    @property
    def potions(self) -> dict[str, int]:
        return self.inventory.potions

    def get_rock(self, rock_id: int | genetics.Rock | None) -> genetics.Rock | None:
        if rock_id is None:
            return None
        if hasattr(rock_id, "id"):
            rock_id = rock_id.id
        return self.rocks.get(int(rock_id))


@dataclass
class MarketListing:
    listing_id: str
    seller_owner_id: str
    rock_id: int
    price: int
    created_generation: int
    status: str = "open"
    note: str = ""

    @property
    def is_open(self) -> bool:
        return self.status == "open"


@dataclass
class TradeOffer:
    offer_id: str
    from_owner_id: str
    to_owner_id: str
    offered_money: int = 0
    offered_rock_ids: list[int] = field(default_factory=list)
    requested_rock_ids: list[int] = field(default_factory=list)
    message: str = ""
    created_generation: int = 0
    status: str = "pending"

    @property
    def is_pending(self) -> bool:
        return self.status == "pending"


@dataclass
class FarmMessage:
    message_id: str
    from_owner_id: str
    to_owner_id: str = PLAYER_OWNER_ID
    kind: str = "info"
    text: str = ""
    created_generation: int = 0
    related_offer_id: str | None = None
    read: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorldState:
    """
    Persistent NPC-world container.
    """

    farms: dict[str, FarmState] = field(default_factory=dict)
    market_listings: list[MarketListing] = field(default_factory=list)
    trade_offers: list[TradeOffer] = field(default_factory=list)
    messages: list[FarmMessage] = field(default_factory=list)
    world_generation: int = 0
    next_world_rock_id: int = 1
    next_listing_id: int = 1
    next_offer_id: int = 1
    next_message_id: int = 1

    def get_farm(self, farm_id: str) -> FarmState | None:
        return self.farms.get(str(farm_id))

    def add_farm(self, farm: FarmState) -> FarmState:
        self.farms[farm.farm_id] = farm
        return farm

    def reserve_world_rock_id(self) -> int:
        rock_id = self.next_world_rock_id
        self.next_world_rock_id += 1
        return rock_id

    def reserve_listing_id(self) -> str:
        listing_id = f"listing_{self.next_listing_id}"
        self.next_listing_id += 1
        return listing_id

    def reserve_offer_id(self) -> str:
        offer_id = f"offer_{self.next_offer_id}"
        self.next_offer_id += 1
        return offer_id

    def reserve_message_id(self) -> str:
        message_id = f"message_{self.next_message_id}"
        self.next_message_id += 1
        return message_id
