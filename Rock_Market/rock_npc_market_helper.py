"""Transactional public listings, bids, and direct trade records."""

from dataclasses import dataclass, field
from enum import Enum


class ListingStatus(str, Enum):
    ACTIVE = "active"
    SOLD = "sold"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class OfferStatus(str, Enum):
    OPEN = "open"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    EXPIRED = "expired"


class FamilyPodStatus(str, Enum):
    ACTIVE = "active"
    SOLD = "sold"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


@dataclass
class MarketBid:
    bid_id: str
    listing_id: str
    bidder_farm_id: str
    amount: int
    created_turn: int
    active: bool = True


@dataclass
class MarketListing:
    listing_id: str
    seller_farm_id: str
    rock_id: int
    asking_price: int
    appraised_value: int
    created_turn: int
    expires_turn: int
    status: ListingStatus = ListingStatus.ACTIVE
    bids: dict[str, MarketBid] = field(default_factory=dict)


@dataclass
class TradeOffer:
    offer_id: str
    sender_farm_id: str
    recipient_farm_id: str
    offered_rock_ids: tuple[int, ...]
    requested_rock_ids: tuple[int, ...]
    offered_money: int
    requested_money: int
    created_turn: int
    expires_turn: int
    status: OfferStatus = OfferStatus.OPEN


@dataclass
class FamilyPodListing:
    pod_id: str
    seller_farm_id: str
    parent_ids: tuple[int, int]
    child_ids: tuple[int, ...]
    price: int
    created_turn: int
    expires_turn: int
    status: FamilyPodStatus = FamilyPodStatus.ACTIVE


@dataclass
class FarmMessage:
    message_id: str
    sender_farm_id: str
    recipient_farm_id: str
    created_turn: int
    kind: str
    text: str
    related_id: str | None = None
    read: bool = False
    requires_response: bool = False
