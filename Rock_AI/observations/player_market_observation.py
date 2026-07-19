from dataclasses import dataclass


@dataclass(frozen=True)
class PublicListingObservation:
    listing_id: str
    seller_farm_id: str
    rock_id: int
    asking_price: int
    appraised_value: int
    created_turn: int
    expires_turn: int
    visible_bid_count: int
    highest_visible_bid: int


@dataclass(frozen=True)
class PlayerMarketObservation:
    listings: tuple[PublicListingObservation, ...]
    random_import_cost: int
    potion_shop: tuple[tuple[str, int, str], ...]
    comparable_price_mean: float
