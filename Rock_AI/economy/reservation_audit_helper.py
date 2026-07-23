"""Reservation-ledger validation and repair for the shared rock economy."""

from __future__ import annotations

from dataclasses import dataclass

from Rock_Market.rock_npc_market_helper import FamilyPodStatus, ListingStatus, OfferStatus


@dataclass(frozen=True)
class ReservationAuditReport:
    valid_before_repair: bool
    repaired: bool
    released_rock_ids: tuple[int, ...]
    restored_rock_ids: tuple[int, ...]
    corrected_farm_ids: tuple[str, ...]
    issues: tuple[str, ...]


def audit_transaction_reservations(world, *, repair: bool = True) -> ReservationAuditReport:
    """Reconcile reservations and committed cash with active economy records."""
    issues: list[str] = []
    expected_rocks: dict[int, str] = {}
    expected_money = {farm_id: 0 for farm_id in world.farms}

    for listing in world.listings.values():
        if listing.status == ListingStatus.ACTIVE and listing.expires_turn < world.turn:
            issues.append(f"expired_listing:{listing.listing_id}")
            if repair:
                listing.status = ListingStatus.EXPIRED
        if listing.status == ListingStatus.ACTIVE:
            if world.owner_of(listing.rock_id) != listing.seller_farm_id:
                issues.append(f"listing_owner_mismatch:{listing.listing_id}")
                if repair:
                    listing.status = ListingStatus.CANCELLED
            else:
                expected_rocks[int(listing.rock_id)] = listing.listing_id
        for bid in listing.bids.values():
            should_be_active = listing.status == ListingStatus.ACTIVE and bid.active
            if should_be_active:
                expected_money[bid.bidder_farm_id] += int(bid.amount)
            elif bid.active:
                issues.append(f"orphan_bid:{bid.bid_id}")
                if repair:
                    bid.active = False

    for offer in world.trade_offers.values():
        if offer.status == OfferStatus.OPEN and offer.expires_turn < world.turn:
            issues.append(f"expired_offer:{offer.offer_id}")
            if repair:
                offer.status = OfferStatus.EXPIRED
        if offer.status != OfferStatus.OPEN:
            continue
        sender_valid = all(world.owner_of(rock_id) == offer.sender_farm_id for rock_id in offer.offered_rock_ids)
        if not sender_valid:
            issues.append(f"offer_owner_mismatch:{offer.offer_id}")
            if repair:
                offer.status = OfferStatus.REJECTED
            continue
        expected_money[offer.sender_farm_id] += int(offer.offered_money)
        for rock_id in offer.offered_rock_ids:
            expected_rocks[int(rock_id)] = offer.offer_id

    for pod in world.family_pods.values():
        if pod.status == FamilyPodStatus.ACTIVE and pod.expires_turn < world.turn:
            issues.append(f"expired_pod:{pod.pod_id}")
            if repair:
                pod.status = FamilyPodStatus.EXPIRED
        if pod.status != FamilyPodStatus.ACTIVE:
            continue
        valid_children = [
            int(rock_id) for rock_id in pod.child_ids
            if world.owner_of(rock_id) == pod.seller_farm_id
        ]
        if len(valid_children) != len(pod.child_ids):
            issues.append(f"pod_owner_mismatch:{pod.pod_id}")
            if repair:
                pod.status = FamilyPodStatus.CANCELLED
            continue
        for rock_id in valid_children:
            expected_rocks[rock_id] = pod.pod_id

    released: list[int] = []
    restored: list[int] = []
    for rock_id, reservation_id in tuple(world.reserved_rock_ids.items()):
        if expected_rocks.get(int(rock_id)) != reservation_id:
            issues.append(f"orphan_reservation:{rock_id}:{reservation_id}")
            if repair:
                world.release_rock(int(rock_id))
                released.append(int(rock_id))
    for rock_id, reservation_id in expected_rocks.items():
        if world.reserved_rock_ids.get(rock_id) != reservation_id:
            issues.append(f"missing_reservation:{rock_id}:{reservation_id}")
            if repair:
                world.release_rock(rock_id)
                world.reserve_rock(rock_id, reservation_id)
                restored.append(rock_id)

    corrected: list[str] = []
    for farm_id, expected in expected_money.items():
        actual = int(world.farm(farm_id).committed_money)
        if actual != expected:
            issues.append(f"committed_money:{farm_id}:{actual}:{expected}")
            if repair:
                world.farm(farm_id).committed_money = max(0, expected)
                corrected.append(farm_id)

    return ReservationAuditReport(
        valid_before_repair=not issues,
        repaired=bool(repair and issues),
        released_rock_ids=tuple(sorted(released)),
        restored_rock_ids=tuple(sorted(restored)),
        corrected_farm_ids=tuple(sorted(corrected)),
        issues=tuple(issues),
    )
