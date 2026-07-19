"""Typed player-only economy commands kept outside neural action schemas."""

from dataclasses import asdict, dataclass
from enum import Enum


class PlayerMarketActionType(str, Enum):
    CANCEL_TRADE_OFFER = "cancel_trade_offer"
    PURCHASE_FAMILY_POD_CHILD = "purchase_family_pod_child"


@dataclass(frozen=True)
class CancelTradeOfferAction:
    actor_farm_id: str
    world_turn: int
    offer_id: str
    action_type = PlayerMarketActionType.CANCEL_TRADE_OFFER

    def to_dict(self):
        return {"action_type": self.action_type.value, **asdict(self)}


@dataclass(frozen=True)
class PurchaseFamilyPodChildAction:
    actor_farm_id: str
    world_turn: int
    pod_id: str
    child_id: int
    quoted_price: int
    action_type = PlayerMarketActionType.PURCHASE_FAMILY_POD_CHILD

    def to_dict(self):
        return {"action_type": self.action_type.value, **asdict(self)}
