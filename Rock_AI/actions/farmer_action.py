"""Immutable serialization-safe farmer actions.

Actions contain only choices from authoritative menus. Hidden import outcomes and
arbitrary object references are deliberately absent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar

from .farmer_action_type import FarmerActionType


@dataclass(frozen=True)
class FarmerAction:
    actor_farm_id: str
    world_turn: int
    action_type: ClassVar[FarmerActionType]

    def to_dict(self) -> dict[str, Any]:
        return {"action_type": self.action_type.value, **asdict(self)}


@dataclass(frozen=True)
class BreedPairAction(FarmerAction):
    parent_a_id: int
    parent_b_id: int
    potion_keys: tuple[str, ...] = ()
    action_type: ClassVar[FarmerActionType] = FarmerActionType.BREED_PAIR


@dataclass(frozen=True)
class StopBreedingAction(FarmerAction):
    action_type: ClassVar[FarmerActionType] = FarmerActionType.STOP_BREEDING


@dataclass(frozen=True)
class ImportRandomRockAction(FarmerAction):
    quoted_cost: int
    requested_sex: str | None = None
    action_type: ClassVar[FarmerActionType] = FarmerActionType.IMPORT_RANDOM_ROCK


@dataclass(frozen=True)
class ImportRequestedRockAction(FarmerAction):
    selected_traits: tuple[tuple[str, str], ...]
    quoted_cost: int
    action_type: ClassVar[FarmerActionType] = FarmerActionType.IMPORT_REQUESTED_ROCK


@dataclass(frozen=True)
class BuyPotionAction(FarmerAction):
    potion_type: str
    quantity: int
    quoted_cost: int
    action_type: ClassVar[FarmerActionType] = FarmerActionType.BUY_POTION


@dataclass(frozen=True)
class SellRockAction(FarmerAction):
    rock_id: int
    quoted_sale_value: int
    action_type: ClassVar[FarmerActionType] = FarmerActionType.SELL_ROCK


@dataclass(frozen=True)
class CreateListingAction(FarmerAction):
    rock_id: int
    asking_price: int
    action_type: ClassVar[FarmerActionType] = FarmerActionType.CREATE_LISTING


@dataclass(frozen=True)
class CancelListingAction(FarmerAction):
    listing_id: str
    action_type: ClassVar[FarmerActionType] = FarmerActionType.CANCEL_LISTING


@dataclass(frozen=True)
class PlaceBidAction(FarmerAction):
    listing_id: str
    bid_amount: int
    action_type: ClassVar[FarmerActionType] = FarmerActionType.PLACE_BID


@dataclass(frozen=True)
class AcceptBidAction(FarmerAction):
    listing_id: str
    bid_id: str
    action_type: ClassVar[FarmerActionType] = FarmerActionType.ACCEPT_BID


@dataclass(frozen=True)
class RejectBidAction(FarmerAction):
    listing_id: str
    bid_id: str
    action_type: ClassVar[FarmerActionType] = FarmerActionType.REJECT_BID


@dataclass(frozen=True)
class CreateTradeOfferAction(FarmerAction):
    recipient_farm_id: str
    offered_rock_ids: tuple[int, ...] = ()
    requested_rock_ids: tuple[int, ...] = ()
    offered_money: int = 0
    requested_money: int = 0
    expires_turn: int = 0
    action_type: ClassVar[FarmerActionType] = FarmerActionType.CREATE_TRADE_OFFER


@dataclass(frozen=True)
class AcceptTradeOfferAction(FarmerAction):
    offer_id: str
    action_type: ClassVar[FarmerActionType] = FarmerActionType.ACCEPT_TRADE_OFFER


@dataclass(frozen=True)
class RejectTradeOfferAction(FarmerAction):
    offer_id: str
    action_type: ClassVar[FarmerActionType] = FarmerActionType.REJECT_TRADE_OFFER


@dataclass(frozen=True)
class PassTurnAction(FarmerAction):
    reason: str = "no_preferred_action"
    action_type: ClassVar[FarmerActionType] = FarmerActionType.PASS_TURN


ACTION_CLASSES = {
    cls.action_type: cls for cls in (
        BreedPairAction, StopBreedingAction, ImportRandomRockAction,
        ImportRequestedRockAction, BuyPotionAction, SellRockAction,
        CreateListingAction, CancelListingAction, PlaceBidAction,
        AcceptBidAction, RejectBidAction, CreateTradeOfferAction,
        AcceptTradeOfferAction, RejectTradeOfferAction, PassTurnAction,
    )
}


def farmer_action_from_dict(data: dict[str, Any]) -> FarmerAction:
    payload = dict(data)
    action_type = FarmerActionType(payload.pop("action_type"))
    cls = ACTION_CLASSES[action_type]
    for key in ("potion_keys", "offered_rock_ids", "requested_rock_ids"):
        if key in payload:
            payload[key] = tuple(payload[key])
    if "selected_traits" in payload:
        payload["selected_traits"] = tuple(tuple(row) for row in payload["selected_traits"])
    return cls(**payload)
