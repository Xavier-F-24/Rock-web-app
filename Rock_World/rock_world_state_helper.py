"""Typed ownership and public economy state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from Rock_GameState.rock_game_state_helper import GameMaster

from .rock_farm_profile_helper import FarmProfile


WORLD_SAVE_VERSION = 2


@dataclass
class FarmerControllerSpec:
    policy_id: str = "heuristic"
    seed: int = 0
    policy_state: dict[str, Any] = field(default_factory=dict)
    warning: str | None = None


@dataclass
class FarmState:
    farm_id: str
    profile: FarmProfile
    game: GameMaster
    visible_rock_ids: set[int] = field(default_factory=set)
    committed_money: int = 0
    observable_history: list[dict[str, Any]] = field(default_factory=list)
    private_messages: list[str] = field(default_factory=list)
    controller: FarmerControllerSpec = field(default_factory=FarmerControllerSpec)

    @property
    def rocks(self):
        return self.game.rock_list

    @property
    def money(self) -> int:
        return self.game.money

    @money.setter
    def money(self, value: int) -> None:
        self.game.money = value

    @property
    def potions(self) -> dict[str, int]:
        return self.game.potions

    @property
    def generation(self) -> int:
        return self.game.generation

    def get_rock(self, rock_id: int):
        return self.game.get_rock(rock_id)

    @property
    def available_money(self) -> int:
        return self.money - self.committed_money


@dataclass
class WorldState:
    farms: dict[str, FarmState]
    owner_by_rock_id: dict[int, str]
    seed: int
    turn: int = 0
    generation: int = 0
    listings: dict[str, Any] = field(default_factory=dict)
    bids: dict[str, Any] = field(default_factory=dict)
    trade_offers: dict[str, Any] = field(default_factory=dict)
    family_pods: dict[str, Any] = field(default_factory=dict)
    messages: list[Any] = field(default_factory=list)
    public_events: list[Any] = field(default_factory=list)
    completed_transaction_ids: set[str] = field(default_factory=set)
    reserved_rock_ids: dict[int, str] = field(default_factory=dict)
    rule_version: str = "economy-1"
    save_version: int = WORLD_SAVE_VERSION
    resolved_npc_count: int = 0

    def __post_init__(self) -> None:
        if not self.resolved_npc_count:
            self.resolved_npc_count = len([farm_id for farm_id in self.farms if farm_id != "player"])
        self.validate_ownership()

    def validate_ownership(self) -> None:
        observed: dict[int, str] = {}
        for farm_id, farm in self.farms.items():
            for rock_id in farm.rocks:
                if rock_id in observed:
                    raise ValueError(f"Rock ID {rock_id} exists in multiple farms")
                observed[rock_id] = farm_id
        if observed != self.owner_by_rock_id:
            raise ValueError("owner_by_rock_id does not match farm inventories")

    def owner_of(self, rock_id: int) -> str | None:
        return self.owner_by_rock_id.get(int(rock_id))

    def farm(self, farm_id: str) -> FarmState:
        try:
            return self.farms[str(farm_id)]
        except KeyError as error:
            raise ValueError(f"Unknown farm: {farm_id}") from error

    def reserve_rock(self, rock_id: int, reservation_id: str) -> None:
        current = self.reserved_rock_ids.get(int(rock_id))
        if current is not None and current != reservation_id:
            raise ValueError(f"Rock #{rock_id} is already reserved")
        self.reserved_rock_ids[int(rock_id)] = reservation_id

    def release_rock(self, rock_id: int, reservation_id: str | None = None) -> None:
        current = self.reserved_rock_ids.get(int(rock_id))
        if current is not None and (reservation_id is None or current == reservation_id):
            self.reserved_rock_ids.pop(int(rock_id), None)

    def transfer_rock(self, rock_id: int, source_farm_id: str, target_farm_id: str) -> None:
        if source_farm_id == target_farm_id:
            raise ValueError("Cannot transfer a rock to its current owner")
        if self.owner_of(rock_id) != source_farm_id:
            raise ValueError(f"Farm {source_farm_id} does not own rock #{rock_id}")
        source = self.farm(source_farm_id)
        target = self.farm(target_farm_id)
        rock = source.rocks.pop(int(rock_id))
        if int(rock_id) in target.rocks:
            source.rocks[int(rock_id)] = rock
            raise ValueError(f"Rock ID collision during transfer: {rock_id}")
        target.rocks[int(rock_id)] = rock
        self.owner_by_rock_id[int(rock_id)] = target_farm_id
        self.release_rock(int(rock_id))
