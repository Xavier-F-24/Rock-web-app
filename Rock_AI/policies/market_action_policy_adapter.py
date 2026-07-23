"""Bounded authoritative legal-action generation for the shared economy."""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass

import Rock_Genetics.rock_genetic_helper as genetics
from Rock_AI.actions.action_encoder import ActionEncoder
from Rock_AI.actions.action_mask import ActionAvailability
from Rock_AI.actions.farmer_action import (
    AcceptBidAction, AcceptTradeOfferAction, BreedPairAction, BuyPotionAction,
    CancelListingAction, CreateListingAction, CreateTradeOfferAction,
    ImportRandomRockAction, ImportRequestedRockAction, PassTurnAction, PlaceBidAction,
    RejectBidAction, RejectTradeOfferAction, SellRockAction,
    StopBreedingAction,
)
from Rock_AI.actions.farmer_action_type import FarmerActionType
from Rock_AI.economy.transaction_validator import EconomyTransactionManager
from Rock_Market.rock_market_helper import POTION_SHOP, RANDOM_ROCK_COST
from Rock_Market.rock_npc_market_helper import ListingStatus, OfferStatus


@dataclass(frozen=True)
class ActionCandidateLimits:
    maximum_active_rocks_considered: int = 16
    maximum_breeding_pairs: int = 32
    maximum_import_actions: int = 3
    maximum_potion_actions: int = 8
    maximum_listing_actions: int = 24
    maximum_bid_actions: int = 24
    maximum_trade_actions: int = 16
    maximum_response_actions: int = 24
    maximum_listed_rocks_considered: int = 8
    maximum_opponent_rocks_considered: int = 8
    maximum_trade_bundles: int = 8
    maximum_price_candidates_per_transaction: int = 4
    maximum_total_legal_actions: int = 128
    maximum_potion_allocations_per_pair: int = 6
    maximum_candidate_generation_seconds: float = 5.0


class LegalFarmerActionGenerator:
    def __init__(self, limits=None, availability=None, encoder=None, heartbeat_callback=None, clock=None):
        self.limits = limits or ActionCandidateLimits()
        self.availability = availability or ActionAvailability.all()
        self.encoder = encoder or ActionEncoder()
        self.transactions = EconomyTransactionManager()
        self.last_pruning_record: dict[str, object] = {}
        self.pruning_records: dict[str, dict[str, object]] = {}
        self.heartbeat_callback = heartbeat_callback
        self.clock = clock or time.monotonic
        self._last_heartbeat = 0.0

    def _heartbeat(self, farm_id: str, operation: str, *, force: bool = False) -> None:
        now = self.clock()
        if self.heartbeat_callback and (force or now - self._last_heartbeat >= 1.0):
            self._last_heartbeat = now
            self.heartbeat_callback("candidate_generation", {"farm_id": farm_id, "operation": operation})

    def _enabled(self, action) -> bool:
        return self.availability.permits(action.action_type)

    def generate(self, world, farm_id: str):
        started = self.clock()
        self._heartbeat(farm_id, "started", force=True)
        farm = world.farm(farm_id)
        game = farm.game
        actions = [PassTurnAction(farm_id, world.turn, "strategic_wait")]
        stop = StopBreedingAction(farm_id, world.turn)
        if self._enabled(stop):
            actions.append(stop)
        reserved = set(world.reserved_rock_ids)
        queued = {rock_id for pair in game.breeding_queue for rock_id in (pair.parent_a_id, pair.parent_b_id)}
        active = sorted(
            (rock for rock in farm.rocks.values() if rock.status == genetics.RockStatus.ACTIVE and rock.id not in reserved and rock.id not in queued),
            key=lambda rock: rock.id,
        )[: self.limits.maximum_active_rocks_considered]
        breeding_actions = 0
        if len(game.breeding_queue) < game.max_pairs_per_generation:
            for left, right in itertools.combinations(active, 2):
                self._heartbeat(farm_id, "breeding_pairs")
                if breeding_actions >= self.limits.maximum_breeding_pairs:
                    break
                if left.id in queued or right.id in queued:
                    continue
                validation = game.breeding_master.validate_breeding_pair(left, right, game=game, warn_relatedness=False)
                if not validation["valid"]:
                    continue
                allocations = [()]
                owned = tuple(sorted(key for key, count in farm.potions.items() if count > 0))
                allocations.extend((key,) for key in owned)
                if len(owned) > 1:
                    allocations.append(owned)
                for potion_keys in allocations[: self.limits.maximum_potion_allocations_per_pair]:
                    action = BreedPairAction(farm_id, world.turn, left.id, right.id, potion_keys)
                    if self._enabled(action):
                        actions.append(action)
                        breeding_actions += 1
                    if breeding_actions >= self.limits.maximum_breeding_pairs:
                        break
        random_import = ImportRandomRockAction(farm_id, world.turn, RANDOM_ROCK_COST)
        import_actions = 0
        if farm.available_money >= RANDOM_ROCK_COST and self._enabled(random_import):
            actions.append(random_import)
            import_actions += 1
        for sex in ("male", "female"):
            if import_actions >= self.limits.maximum_import_actions:
                break
            traits = (("sex", sex),)
            cost = game.market_manager.quote_defined_trait_request(dict(traits))
            action = ImportRequestedRockAction(farm_id, world.turn, traits, cost)
            if farm.available_money >= cost and self._enabled(action):
                actions.append(action)
                import_actions += 1
        for potion_key, definition in sorted(POTION_SHOP.items())[: self.limits.maximum_potion_actions]:
            action = BuyPotionAction(farm_id, world.turn, potion_key, 1, int(definition["cost"]))
            if farm.available_money >= action.quoted_cost and self._enabled(action):
                actions.append(action)
        listing_actions = 0
        for rock in active:
            self._heartbeat(farm_id, "listing_candidates")
            if rock.sell_value > 0:
                action = SellRockAction(farm_id, world.turn, rock.id, int(rock.sell_value))
                if self._enabled(action):
                    actions.append(action)
            for price in self.transactions.legal_price_menu(rock.value)[: self.limits.maximum_price_candidates_per_transaction]:
                action = CreateListingAction(farm_id, world.turn, rock.id, price)
                if self._enabled(action):
                    actions.append(action)
                    listing_actions += 1
                if listing_actions >= self.limits.maximum_listing_actions:
                    break
            if listing_actions >= self.limits.maximum_listing_actions:
                break
        own_listings = [listing for listing in world.listings.values() if listing.seller_farm_id == farm_id and listing.status == ListingStatus.ACTIVE]
        response_actions = 0
        for listing in sorted(own_listings, key=lambda row: row.listing_id)[: self.limits.maximum_listed_rocks_considered]:
            cancel = CancelListingAction(farm_id, world.turn, listing.listing_id)
            if self._enabled(cancel):
                actions.append(cancel)
                response_actions += 1
            for bid in sorted(listing.bids.values(), key=lambda row: (-row.amount, row.bid_id)):
                if not bid.active:
                    continue
                for response in (AcceptBidAction(farm_id, world.turn, listing.listing_id, bid.bid_id), RejectBidAction(farm_id, world.turn, listing.listing_id, bid.bid_id)):
                    if self._enabled(response):
                        actions.append(response)
                        response_actions += 1
                    if response_actions >= self.limits.maximum_response_actions:
                        break
                if response_actions >= self.limits.maximum_response_actions:
                    break
            if response_actions >= self.limits.maximum_response_actions:
                break
        other_listings = [listing for listing in world.listings.values() if listing.seller_farm_id != farm_id and listing.status == ListingStatus.ACTIVE]
        bid_actions = 0
        for listing in sorted(other_listings, key=lambda row: (row.asking_price, row.listing_id))[: self.limits.maximum_listed_rocks_considered]:
            self._heartbeat(farm_id, "bid_candidates")
            for amount in self.transactions.legal_bid_menu(listing, farm.available_money)[: self.limits.maximum_price_candidates_per_transaction]:
                bid = PlaceBidAction(farm_id, world.turn, listing.listing_id, amount)
                if self._enabled(bid):
                    actions.append(bid)
                    bid_actions += 1
                if bid_actions >= self.limits.maximum_bid_actions:
                    break
            if bid_actions >= self.limits.maximum_bid_actions:
                break
        incoming = [offer for offer in world.trade_offers.values() if offer.recipient_farm_id == farm_id and offer.status == OfferStatus.OPEN and offer.expires_turn >= world.turn]
        for offer in sorted(incoming, key=lambda row: row.offer_id)[: self.limits.maximum_response_actions]:
            for response in (AcceptTradeOfferAction(farm_id, world.turn, offer.offer_id), RejectTradeOfferAction(farm_id, world.turn, offer.offer_id)):
                if self._enabled(response):
                    actions.append(response)
        trade_count = 0
        for other_id, other in sorted(world.farms.items()):
            if other_id == farm_id:
                continue
            other_queued = {rock_id for pair in other.game.breeding_queue for rock_id in (pair.parent_a_id, pair.parent_b_id)}
            their_rocks = [other.rocks[rock_id] for rock_id in sorted(other.visible_rock_ids & set(other.rocks)) if rock_id not in reserved and rock_id not in other_queued and other.rocks[rock_id].status == genetics.RockStatus.ACTIVE]
            for ours in active[: self.limits.maximum_opponent_rocks_considered]:
                for theirs in their_rocks[: self.limits.maximum_opponent_rocks_considered]:
                    action = CreateTradeOfferAction(farm_id, world.turn, other_id, (ours.id,), (theirs.id,), 0, 0, world.turn + 3)
                    if self._enabled(action):
                        actions.append(action)
                        trade_count += 1
                    if trade_count >= min(self.limits.maximum_trade_bundles, self.limits.maximum_trade_actions):
                        break
                if trade_count >= min(self.limits.maximum_trade_bundles, self.limits.maximum_trade_actions):
                    break
            if trade_count >= min(self.limits.maximum_trade_bundles, self.limits.maximum_trade_actions):
                break
            if self.clock() - started > self.limits.maximum_candidate_generation_seconds:
                break
        original_count = len(actions)
        pass_action = next((action for action in actions if isinstance(action, PassTurnAction)), PassTurnAction(farm_id, world.turn, "no_other_legal_action"))
        actions = sorted(
            (action for action in actions if not isinstance(action, PassTurnAction)),
            key=lambda action: (action.action_type.value, str(action.to_dict())),
        )
        actions = [pass_action, *actions[: max(0, self.limits.maximum_total_legal_actions - 1)]]
        candidates = []
        for action in actions:
            rock_a, rock_b, listing, relatedness = self._entities(world, farm, action)
            candidate = self.encoder.encode(
                action, actor=farm, world=world, objective=farm.profile,
                rock_a=rock_a, rock_b=rock_b, listing=listing, relatedness_r=relatedness,
                reasons=("authoritative_legal_candidate",),
            )
            candidates.append(candidate)
        self.last_pruning_record = {
            "farm_id": farm_id, "world_turn": world.turn, "generated": original_count,
            "retained": len(candidates), "pruned": max(0, original_count - len(candidates)),
            "limit": self.limits.maximum_total_legal_actions,
            "elapsed_seconds": self.clock() - started,
            "counts_by_type": {
                kind.value: sum(candidate.action.action_type == kind for candidate in candidates)
                for kind in FarmerActionType
            },
        }
        self.pruning_records[farm_id] = dict(self.last_pruning_record)
        self._heartbeat(farm_id, "completed", force=True)
        return tuple(candidates)

    @staticmethod
    def _entities(world, farm, action):
        rock_a = rock_b = listing = None
        relatedness = 0.0
        if isinstance(action, BreedPairAction):
            rock_a, rock_b = farm.get_rock(action.parent_a_id), farm.get_rock(action.parent_b_id)
            relatedness, _ = farm.game.breeding_master.calculate_relatedness(farm.game, rock_a, rock_b)
        elif hasattr(action, "rock_id"):
            rock_a = farm.get_rock(action.rock_id)
        if hasattr(action, "listing_id"):
            listing = world.listings.get(action.listing_id)
            if listing is not None:
                rock_a = world.farm(listing.seller_farm_id).get_rock(listing.rock_id)
        if isinstance(action, CreateTradeOfferAction):
            rock_a = farm.get_rock(action.offered_rock_ids[0]) if action.offered_rock_ids else None
            other = world.farm(action.recipient_farm_id)
            rock_b = other.get_rock(action.requested_rock_ids[0]) if action.requested_rock_ids else None
        return rock_a, rock_b, listing, relatedness
