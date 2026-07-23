"""Atomic, idempotent execution for shared-world economy actions."""

from __future__ import annotations

import hashlib

import Rock_Genetics.rock_genetic_helper as genetics
from Rock_AI.actions.action_result import ActionResult
from Rock_AI.actions.farmer_action import (
    AcceptBidAction, AcceptTradeOfferAction, BreedPairAction, BuyPotionAction,
    CancelListingAction, CreateListingAction, CreateTradeOfferAction, FarmerAction,
    ImportRandomRockAction, ImportRequestedRockAction, PassTurnAction, PlaceBidAction,
    RejectBidAction, RejectTradeOfferAction, SellRockAction, StopBreedingAction,
)
from Rock_AI.logging.public_world_event_record import PublicWorldEventRecord
from Rock_Market.rock_market_helper import POTION_SHOP, RANDOM_ROCK_COST
from Rock_Market.rock_npc_market_helper import (
    FamilyPodStatus, FarmMessage, ListingStatus, MarketBid, MarketListing, OfferStatus, TradeOffer,
)
from Rock_Market.rock_player_market_action_helper import CancelTradeOfferAction, PurchaseFamilyPodChildAction
from Rock_World.rock_world_state_helper import WorldState
from .reservation_audit_helper import audit_transaction_reservations


def _stable_id(prefix: str, *parts: object) -> str:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


class EconomyTransactionManager:
    """Validate first, then commit each action as one authoritative transaction."""

    def execute(self, world: WorldState, action: FarmerAction, action_hash: str) -> ActionResult:
        audit_transaction_reservations(world, repair=True)
        transaction_id = _stable_id("tx", action_hash, action.actor_farm_id, action.world_turn)
        if transaction_id in world.completed_transaction_ids:
            return ActionResult(True, action_hash, transaction_id, "Transaction already applied.", action.actor_farm_id, action.world_turn, idempotent_replay=True)
        if action.world_turn != world.turn:
            return self._failure(action, action_hash, transaction_id, "stale_action", "Action belongs to a stale world turn.")
        try:
            payload, rock_ids, summary = self._dispatch(world, action)
        except (ValueError, TypeError, KeyError, IndexError) as error:
            audit_transaction_reservations(world, repair=True)
            return self._failure(action, action_hash, transaction_id, "validation_failed", str(error))
        except Exception:
            audit_transaction_reservations(world, repair=True)
            raise
        world.completed_transaction_ids.add(transaction_id)
        event = PublicWorldEventRecord(
            _stable_id("event", transaction_id), world.turn, action.action_type.value,
            summary, (action.actor_farm_id,), tuple(rock_ids), payload,
        )
        world.public_events.append(event)
        audit_transaction_reservations(world, repair=True)
        return ActionResult(True, action_hash, transaction_id, summary, action.actor_farm_id, world.turn, payload, affected_rock_ids=tuple(rock_ids))

    @staticmethod
    def _failure(action, action_hash, transaction_id, code, message):
        return ActionResult(False, action_hash, transaction_id, message, action.actor_farm_id, action.world_turn, error_code=code)

    def _dispatch(self, world: WorldState, action: FarmerAction):
        actor = world.farm(action.actor_farm_id)
        if isinstance(action, BreedPairAction):
            if action.parent_a_id in world.reserved_rock_ids or action.parent_b_id in world.reserved_rock_ids:
                raise ValueError("A selected parent is reserved by a market transaction")
            pair = actor.game.add_pair_to_queue(action.parent_a_id, action.parent_b_id, potion_keys=list(action.potion_keys))
            return {"parent_a_id": pair.parent_a_id, "parent_b_id": pair.parent_b_id, "potion_keys": list(pair.potion_keys)}, (pair.parent_a_id, pair.parent_b_id), f"Queued rocks #{pair.parent_a_id} and #{pair.parent_b_id} for breeding."
        if isinstance(action, (StopBreedingAction, PassTurnAction)):
            return {"reason": getattr(action, "reason", "stop_breeding")}, (), "Farmer passed this turn."
        if isinstance(action, ImportRandomRockAction):
            if action.quoted_cost != RANDOM_ROCK_COST:
                raise ValueError("Random import quote is no longer valid")
            self._prepare_global_id(world, actor)
            before = set(actor.rocks)
            sex = genetics.Sex(action.requested_sex) if action.requested_sex else None
            rock = actor.game.buy_random_rock(cost=action.quoted_cost, sex=sex)
            self._register_new_rocks(world, actor, before)
            return {"cost": action.quoted_cost, "revealed_rock_id": rock.id}, (rock.id,), f"Imported random rock #{rock.id} for ${action.quoted_cost}."
        if isinstance(action, ImportRequestedRockAction):
            self._prepare_global_id(world, actor)
            selected = dict(action.selected_traits)
            authoritative_cost = actor.game.market_manager.quote_defined_trait_request(selected)
            if action.quoted_cost != authoritative_cost:
                raise ValueError("Requested import quote is no longer valid")
            before = set(actor.rocks)
            rock = actor.game.buy_defined_trait_rock(selected, cost=authoritative_cost)
            self._register_new_rocks(world, actor, before)
            return {"cost": authoritative_cost, "selected_traits": selected, "revealed_rock_id": rock.id}, (rock.id,), f"Imported requested rock #{rock.id} for ${authoritative_cost}."
        if isinstance(action, BuyPotionAction):
            definition = POTION_SHOP.get(action.potion_type)
            if definition is None or action.quantity < 1:
                raise ValueError("Unknown potion or invalid quantity")
            expected = int(definition["cost"]) * action.quantity
            if action.quoted_cost != expected or actor.available_money < expected:
                raise ValueError("Potion quote is invalid or unaffordable")
            for _ in range(action.quantity):
                actor.game.buy_potion(action.potion_type)
            return {"potion_type": action.potion_type, "quantity": action.quantity, "cost": expected}, (), f"Bought {action.quantity} {definition['name']}."
        if isinstance(action, SellRockAction):
            self._require_owned_transferable(world, actor.farm_id, action.rock_id)
            rock = actor.get_rock(action.rock_id)
            actor.game.finalize_rock(rock)
            if action.quoted_sale_value != rock.sell_value:
                raise ValueError("Sale quote is no longer valid")
            value = actor.game.sell_rock(action.rock_id)
            return {"proceeds": value}, (action.rock_id,), f"Sold rock #{action.rock_id} for ${value}."
        if isinstance(action, CreateListingAction):
            self._require_owned_transferable(world, actor.farm_id, action.rock_id)
            rock = actor.get_rock(action.rock_id)
            if action.asking_price not in self.legal_price_menu(rock.value):
                raise ValueError("Listing price is not in the authoritative menu")
            listing_id = _stable_id("listing", actor.farm_id, action.rock_id, world.turn)
            world.reserve_rock(action.rock_id, listing_id)
            listing = MarketListing(listing_id, actor.farm_id, action.rock_id, action.asking_price, int(rock.value), world.turn, world.turn + 5)
            world.listings[listing_id] = listing
            return {"listing_id": listing_id, "asking_price": action.asking_price}, (action.rock_id,), f"Listed rock #{action.rock_id} for ${action.asking_price}."
        if isinstance(action, CancelListingAction):
            listing = self._active_listing(world, action.listing_id)
            if listing.seller_farm_id != actor.farm_id:
                raise ValueError("Only the seller can cancel this listing")
            self._release_listing_commitments(world, listing)
            listing.status = ListingStatus.CANCELLED
            world.release_rock(listing.rock_id, listing.listing_id)
            return {"listing_id": listing.listing_id}, (listing.rock_id,), f"Cancelled listing {listing.listing_id}."
        if isinstance(action, PlaceBidAction):
            listing = self._active_listing(world, action.listing_id)
            if listing.seller_farm_id == actor.farm_id:
                raise ValueError("A farm cannot bid on its own listing")
            minimum = max(listing.asking_price, max((bid.amount + 1 for bid in listing.bids.values() if bid.active), default=listing.asking_price))
            if action.bid_amount not in self.legal_bid_menu(listing, actor.available_money) or action.bid_amount < minimum:
                raise ValueError("Bid is not an affordable authoritative price candidate")
            actor.committed_money += action.bid_amount
            bid_id = _stable_id("bid", listing.listing_id, actor.farm_id, world.turn, action.bid_amount)
            bid = MarketBid(bid_id, listing.listing_id, actor.farm_id, action.bid_amount, world.turn)
            listing.bids[bid_id] = bid
            world.bids[bid_id] = bid
            self._message(world, actor.farm_id, listing.seller_farm_id, "bid_received", f"{actor.profile.display_name} bid ${action.bid_amount} on rock #{listing.rock_id}.", bid_id, True)
            return {"listing_id": listing.listing_id, "bid_id": bid_id, "amount": action.bid_amount}, (listing.rock_id,), f"Bid ${action.bid_amount} on rock #{listing.rock_id}."
        if isinstance(action, (AcceptBidAction, RejectBidAction)):
            listing = self._active_listing(world, action.listing_id)
            if listing.seller_farm_id != actor.farm_id:
                raise ValueError("Only the seller can respond to bids")
            bid = listing.bids.get(action.bid_id)
            if bid is None or not bid.active:
                raise ValueError("Bid is no longer active")
            bidder = world.farm(bid.bidder_farm_id)
            if isinstance(action, RejectBidAction):
                bid.active = False
                bidder.committed_money = max(0, bidder.committed_money - bid.amount)
                self._message(world, actor.farm_id, bidder.farm_id, "bid_rejected", f"Your bid on rock #{listing.rock_id} was rejected.", bid.bid_id)
                return {"bid_id": bid.bid_id}, (listing.rock_id,), f"Rejected bid {bid.bid_id}."
            if bidder.money < bid.amount:
                raise ValueError("Bidder can no longer honor the bid")
            if world.owner_of(listing.rock_id) != actor.farm_id or world.reserved_rock_ids.get(listing.rock_id) != listing.listing_id:
                raise ValueError("Listed rock is no longer reserved for this listing")
            if listing.rock_id in bidder.rocks:
                raise ValueError("Rock ID collision at buyer farm")
            bidder.money -= bid.amount
            bidder.committed_money -= bid.amount
            actor.money += bid.amount
            world.transfer_rock(listing.rock_id, actor.farm_id, bidder.farm_id)
            listing.status = ListingStatus.SOLD
            self._release_listing_commitments(world, listing, except_bid_id=bid.bid_id)
            bid.active = False
            self._message(world, actor.farm_id, bidder.farm_id, "bid_accepted", f"Your bid purchased rock #{listing.rock_id} for ${bid.amount}.", bid.bid_id)
            return {"bid_id": bid.bid_id, "price": bid.amount, "buyer_farm_id": bidder.farm_id}, (listing.rock_id,), f"Sold listed rock #{listing.rock_id} to {bidder.farm_id} for ${bid.amount}."
        if isinstance(action, CreateTradeOfferAction):
            if action.recipient_farm_id == actor.farm_id:
                raise ValueError("Self-trading is prohibited")
            recipient = world.farm(action.recipient_farm_id)
            if len(action.offered_rock_ids) > 1 or len(action.requested_rock_ids) > 1:
                raise ValueError("Initial trade system supports at most one rock per side")
            if action.expires_turn <= world.turn:
                raise ValueError("Trade offer must expire in a future turn")
            for rock_id in action.offered_rock_ids:
                self._require_owned_transferable(world, actor.farm_id, rock_id)
            for rock_id in action.requested_rock_ids:
                self._require_owned_transferable(world, recipient.farm_id, rock_id)
            if actor.available_money < action.offered_money:
                raise ValueError("Insufficient uncommitted money for trade offer")
            offer_id = _stable_id("offer", actor.farm_id, recipient.farm_id, world.turn, action.offered_rock_ids, action.requested_rock_ids, action.offered_money, action.requested_money)
            for rock_id in action.offered_rock_ids:
                world.reserve_rock(rock_id, offer_id)
            actor.committed_money += action.offered_money
            offer = TradeOffer(offer_id, actor.farm_id, recipient.farm_id, action.offered_rock_ids, action.requested_rock_ids, action.offered_money, action.requested_money, world.turn, action.expires_turn)
            world.trade_offers[offer_id] = offer
            self._message(world, actor.farm_id, recipient.farm_id, "trade_received", f"{actor.profile.display_name} sent a direct trade offer.", offer_id, True)
            return {"offer_id": offer_id}, action.offered_rock_ids + action.requested_rock_ids, f"Proposed trade {offer_id} to {recipient.farm_id}."
        if isinstance(action, (AcceptTradeOfferAction, RejectTradeOfferAction)):
            offer = world.trade_offers.get(action.offer_id)
            if offer is None or offer.status != OfferStatus.OPEN or offer.expires_turn < world.turn:
                raise ValueError("Trade offer is no longer open")
            if offer.recipient_farm_id != actor.farm_id:
                raise ValueError("Only the recipient can respond to this offer")
            sender = world.farm(offer.sender_farm_id)
            if isinstance(action, RejectTradeOfferAction):
                self._release_trade(world, offer)
                offer.status = OfferStatus.REJECTED
                self._message(world, actor.farm_id, sender.farm_id, "trade_rejected", f"{actor.profile.display_name} rejected trade {offer.offer_id}.", offer.offer_id)
                return {"offer_id": offer.offer_id}, offer.offered_rock_ids + offer.requested_rock_ids, f"Rejected trade {offer.offer_id}."
            for rock_id in offer.offered_rock_ids:
                if world.owner_of(rock_id) != sender.farm_id or world.reserved_rock_ids.get(rock_id) != offer.offer_id:
                    raise ValueError("Offered rock is no longer available")
            for rock_id in offer.requested_rock_ids:
                self._require_owned_transferable(world, actor.farm_id, rock_id)
            if sender.money < offer.offered_money or actor.available_money < offer.requested_money:
                raise ValueError("Trade money can no longer be honored")
            sender.money += offer.requested_money - offer.offered_money
            actor.money += offer.offered_money - offer.requested_money
            sender.committed_money -= offer.offered_money
            for rock_id in offer.offered_rock_ids:
                world.transfer_rock(rock_id, sender.farm_id, actor.farm_id)
            for rock_id in offer.requested_rock_ids:
                world.transfer_rock(rock_id, actor.farm_id, sender.farm_id)
            offer.status = OfferStatus.ACCEPTED
            self._message(world, actor.farm_id, sender.farm_id, "trade_accepted", f"{actor.profile.display_name} accepted trade {offer.offer_id}.", offer.offer_id)
            return {"offer_id": offer.offer_id}, offer.offered_rock_ids + offer.requested_rock_ids, f"Completed trade {offer.offer_id}."
        if isinstance(action, CancelTradeOfferAction):
            offer = world.trade_offers.get(action.offer_id)
            if offer is None or offer.status != OfferStatus.OPEN:
                raise ValueError("Trade offer is no longer open")
            if offer.sender_farm_id != actor.farm_id:
                raise ValueError("Only the sender can cancel this trade")
            self._release_trade(world, offer)
            offer.status = OfferStatus.REJECTED
            self._message(world, actor.farm_id, offer.recipient_farm_id, "trade_cancelled", f"Trade {offer.offer_id} was cancelled.", offer.offer_id)
            return {"offer_id": offer.offer_id}, offer.offered_rock_ids + offer.requested_rock_ids, f"Cancelled trade {offer.offer_id}."
        if isinstance(action, PurchaseFamilyPodChildAction):
            pod = world.family_pods.get(action.pod_id)
            if pod is None or pod.status != FamilyPodStatus.ACTIVE or pod.expires_turn < world.turn:
                raise ValueError("Family pod is no longer active")
            if pod.seller_farm_id == actor.farm_id or action.child_id not in pod.child_ids:
                raise ValueError("Invalid family pod child selection")
            if action.quoted_price != pod.price or actor.available_money < pod.price:
                raise ValueError("Family pod quote is invalid or unaffordable")
            seller = world.farm(pod.seller_farm_id)
            if world.owner_of(action.child_id) != seller.farm_id or world.reserved_rock_ids.get(action.child_id) != pod.pod_id:
                raise ValueError("Selected pod child is no longer available")
            actor.money -= pod.price
            seller.money += pod.price
            world.transfer_rock(action.child_id, seller.farm_id, actor.farm_id)
            for child_id in pod.child_ids:
                world.release_rock(child_id, pod.pod_id)
            pod.status = FamilyPodStatus.SOLD
            self._message(world, actor.farm_id, seller.farm_id, "pod_purchased", f"{actor.profile.display_name} purchased rock #{action.child_id} from family pod {pod.pod_id}.", pod.pod_id)
            return {"pod_id": pod.pod_id, "price": pod.price, "seller_farm_id": seller.farm_id}, (action.child_id,), f"Purchased family pod child #{action.child_id} for ${pod.price}."
        raise ValueError(f"Unsupported action: {action.action_type.value}")

    @staticmethod
    def _message(world, sender, recipient, kind, text, related_id=None, requires_response=False):
        message_id = _stable_id("message", sender, recipient, kind, related_id, world.turn, len(world.messages))
        world.messages.append(FarmMessage(message_id, sender, recipient, world.turn, kind, text, related_id, False, requires_response))

    @staticmethod
    def _price_points(appraised_value: int) -> tuple[int, ...]:
        return tuple(sorted({1, *(max(1, round(appraised_value * factor)) for factor in (.75, 1.0, 1.25, 1.5))}))

    @classmethod
    def legal_price_menu(cls, appraised_value: int) -> tuple[int, ...]:
        return cls._price_points(int(appraised_value))

    @classmethod
    def legal_bid_menu(cls, listing, available_money: int) -> tuple[int, ...]:
        minimum = max(listing.asking_price, max((bid.amount + 1 for bid in listing.bids.values() if bid.active), default=listing.asking_price))
        points = {minimum, *(max(1, round(listing.appraised_value * factor)) for factor in (.8, .9, 1.0, 1.1))}
        return tuple(sorted(value for value in points if minimum <= value <= available_money))

    @staticmethod
    def _active_listing(world, listing_id):
        listing = world.listings.get(listing_id)
        if listing is None or listing.status != ListingStatus.ACTIVE or listing.expires_turn < world.turn:
            raise ValueError("Listing is no longer active")
        return listing

    @staticmethod
    def _require_owned_transferable(world, farm_id, rock_id, allow_unreserved=False):
        if world.owner_of(rock_id) != farm_id:
            raise ValueError(f"Farm {farm_id} does not own rock #{rock_id}")
        reservation = world.reserved_rock_ids.get(int(rock_id))
        if reservation is not None and not allow_unreserved:
            raise ValueError(f"Rock #{rock_id} is reserved")
        rock = world.farm(farm_id).get_rock(rock_id)
        if rock is None or rock.status != genetics.RockStatus.ACTIVE:
            raise ValueError(f"Rock #{rock_id} is not transferable")
        queued = {queued_id for pair in world.farm(farm_id).game.breeding_queue for queued_id in (pair.parent_a_id, pair.parent_b_id)}
        if int(rock_id) in queued:
            raise ValueError(f"Rock #{rock_id} is queued for breeding")

    @staticmethod
    def _prepare_global_id(world, actor):
        actor.game.next_rock_id = max([*world.owner_by_rock_id, actor.game.next_rock_id - 1], default=0) + 1

    @staticmethod
    def _register_new_rocks(world, actor, before):
        for rock_id in set(actor.rocks) - before:
            if rock_id in world.owner_by_rock_id:
                raise ValueError(f"Generated duplicate rock ID {rock_id}")
            world.owner_by_rock_id[rock_id] = actor.farm_id
            actor.visible_rock_ids.add(rock_id)

    @staticmethod
    def _release_listing_commitments(world, listing, except_bid_id=None):
        for bid in listing.bids.values():
            if bid.active and bid.bid_id != except_bid_id:
                bidder = world.farm(bid.bidder_farm_id)
                bidder.committed_money -= bid.amount
                bid.active = False

    @staticmethod
    def _release_trade(world, offer):
        sender = world.farm(offer.sender_farm_id)
        sender.committed_money = max(0, sender.committed_money - offer.offered_money)
        for rock_id in offer.offered_rock_ids:
            world.release_rock(rock_id, offer.offer_id)
