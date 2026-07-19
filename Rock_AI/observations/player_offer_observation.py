from dataclasses import dataclass


@dataclass(frozen=True)
class PlayerOfferObservation:
    incoming_offer_ids: tuple[str, ...]
    outgoing_offer_ids: tuple[str, ...]
    own_bid_ids: tuple[str, ...]
    public_bid_ids: tuple[str, ...]
