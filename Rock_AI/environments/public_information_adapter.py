"""Public world view that intentionally omits private money and hidden genetics."""

import hashlib
import json

from Rock_AI.observations.player_market_observation import PublicListingObservation
from Rock_AI.observations.player_opponent_observation import PublicFarmObservation
from Rock_Market.rock_market_helper import POTION_SHOP, RANDOM_ROCK_COST
from Rock_Market.rock_npc_market_helper import ListingStatus


class PublicInformationAdapter:
    @staticmethod
    def public_farms(world, viewer_farm_id: str) -> tuple[PublicFarmObservation, ...]:
        rows = []
        for farm_id, farm in sorted(world.farms.items()):
            if farm_id == viewer_farm_id:
                continue
            visible = sorted(farm.visible_rock_ids & set(farm.rocks))
            rows.append(PublicFarmObservation(
                farm_id, farm.profile.display_name, farm.generation, tuple(visible),
                tuple(int(farm.rocks[rock_id].value) for rock_id in visible),
                tuple(event.summary for event in world.public_events[-5:] if farm_id in event.farm_ids),
            ))
        return tuple(rows)

    @staticmethod
    def listings(world) -> tuple[PublicListingObservation, ...]:
        rows = []
        for listing in sorted(world.listings.values(), key=lambda item: item.listing_id):
            if listing.status != ListingStatus.ACTIVE or listing.expires_turn < world.turn:
                continue
            active_bids = [bid.amount for bid in listing.bids.values() if bid.active]
            rows.append(PublicListingObservation(
                listing.listing_id, listing.seller_farm_id, listing.rock_id,
                listing.asking_price, listing.appraised_value, listing.created_turn,
                listing.expires_turn, len(active_bids), max(active_bids, default=0),
            ))
        return tuple(rows)

    @staticmethod
    def public_hash(payload: object) -> str:
        raw = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def potion_shop():
        return tuple((key, int(value["cost"]), str(value["description"])) for key, value in sorted(POTION_SHOP.items()))

    @staticmethod
    def random_import_cost() -> int:
        return RANDOM_ROCK_COST
