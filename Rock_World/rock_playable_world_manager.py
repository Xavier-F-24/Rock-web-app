"""Authoritative explicit-turn runtime for the persistent playable world."""

from __future__ import annotations

import hashlib

from Rock_AI.actions.action_mask import ActionAvailability
from Rock_AI.actions.farmer_action import PassTurnAction
from Rock_AI.actions.farmer_action_type import FarmerActionType
from Rock_AI.economy.transaction_validator import EconomyTransactionManager
from Rock_AI.environments.private_farmer_observation_adapter import PrivateFarmerObservationAdapter
from Rock_AI.environments.simultaneous_action_resolver import ActionIntent, SimultaneousActionResolver
from Rock_AI.policies.market_action_policy_adapter import LegalFarmerActionGenerator
from Rock_Market.rock_npc_market_helper import FamilyPodListing, FamilyPodStatus, ListingStatus, MarketListing, OfferStatus

from .rock_farmer_policy_registry import FarmerPolicyRegistry


PRODUCTION_ACTIONS = frozenset({
    action for action in FarmerActionType
    if action not in {FarmerActionType.IMPORT_RANDOM_ROCK, FarmerActionType.IMPORT_REQUESTED_ROCK, FarmerActionType.SELL_ROCK}
})


def _stable_id(prefix: str, *parts) -> str:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


class PlayableWorldManager:
    def __init__(self, registry=None):
        self.registry = registry or FarmerPolicyRegistry()
        self.transactions = EconomyTransactionManager()
        self.generator = LegalFarmerActionGenerator(availability=ActionAvailability(PRODUCTION_ACTIONS))
        self.observations = PrivateFarmerObservationAdapter(self.generator)
        self.resolver = SimultaneousActionResolver(self.transactions)

    @staticmethod
    def world(game):
        if game.world is None:
            raise RuntimeError("Game has no persistent rock world")
        return game.world

    def execute_player_action(self, game, action, action_hash: str):
        if action.actor_farm_id != "player":
            raise ValueError("Player controller may only execute actions for the player farm")
        return self.transactions.execute(self.world(game), action, action_hash)

    def bootstrap_market(self, game):
        world = self.world(game)
        for farm_id, farm in sorted(world.farms.items()):
            if farm_id == "player":
                continue
            self._create_family_pod(world, farm_id, list(farm.rocks.values()))
            candidates = [rock for rock in farm.rocks.values() if rock.is_active and rock.id not in world.reserved_rock_ids]
            if not candidates:
                continue
            rock = sorted(candidates, key=lambda row: (-row.generation, row.id))[0]
            listing_id = _stable_id("listing", farm_id, rock.id, world.turn, "bootstrap")
            world.reserve_rock(rock.id, listing_id)
            asking = self.transactions.legal_price_menu(rock.value)[1 if len(self.transactions.legal_price_menu(rock.value)) > 1 else 0]
            world.listings[listing_id] = MarketListing(listing_id, farm_id, rock.id, asking, int(rock.value), world.turn, world.turn + 6)
        return world

    def end_turn(self, game):
        world = self.world(game)
        intents = []
        agents = {}
        chosen_candidates = {}
        for farm_id in sorted(world.farms):
            candidates = self.generator.generate(world, farm_id)
            if farm_id == "player":
                selected = next(row for row in candidates if isinstance(row.action, PassTurnAction))
            else:
                farm = world.farm(farm_id)
                agent = self.registry.build(farm)
                agents[farm_id] = agent
                selected = agent.choose_candidate(self.observations.build(world, farm_id, recurrent_state=getattr(getattr(agent, "policy", None), "state", None)))
            intents.append(ActionIntent(farm_id, selected.action, selected.candidate_hash))
            chosen_candidates[farm_id] = selected
        results = self.resolver.resolve(world, tuple(intents))
        by_farm = {result.actor_farm_id: result for result in results}
        for farm_id, agent in agents.items():
            agent.observe_result(chosen_candidates[farm_id], by_farm[farm_id])
            self.registry.save_state(world.farm(farm_id), agent)
        world.turn += 1
        self._expire(world)
        return {
            "turn": world.turn,
            "results": tuple(results),
            "successful_actions": sum(result.success for result in results),
            "failed_actions": sum(not result.success for result in results),
        }

    def advance_after_player_generation(self, game):
        world = self.world(game)
        player = world.farm("player")
        for rock_id in set(player.rocks) - set(world.owner_by_rock_id):
            world.owner_by_rock_id[rock_id] = "player"
            player.visible_rock_ids.add(rock_id)
        created = []
        for farm_id, farm in sorted(world.farms.items()):
            if farm_id == "player":
                continue
            farm.game.next_rock_id = max(world.owner_by_rock_id, default=0) + 1
            before = set(farm.rocks)
            children = farm.game.breed_queue()
            farm.game.generation += 1
            for rock_id in sorted(set(farm.rocks) - before):
                if rock_id in world.owner_by_rock_id:
                    raise ValueError(f"Duplicate child rock ID {rock_id}")
                world.owner_by_rock_id[rock_id] = farm_id
                farm.visible_rock_ids.add(rock_id)
            created.extend(children)
            self._create_family_pod(world, farm_id, children)
        world.generation = game.generation
        world.validate_ownership()
        self._expire(world)
        return created

    @staticmethod
    def _create_family_pod(world, farm_id, children):
        active_existing = any(pod.seller_farm_id == farm_id and pod.status == FamilyPodStatus.ACTIVE for pod in world.family_pods.values())
        if active_existing:
            return None
        sibling_groups = {}
        for child in children:
            if child.is_active and len(child.parent_ids) == 2:
                sibling_groups.setdefault(tuple(child.parent_ids), []).append(child)
        eligible = [group for group in sibling_groups.values() if len(group) >= 2]
        if not eligible:
            return None
        siblings = sorted(max(eligible, key=len), key=lambda rock: rock.id)
        pod_id = _stable_id("pod", farm_id, world.turn, tuple(rock.id for rock in siblings))
        for rock in siblings:
            world.reserve_rock(rock.id, pod_id)
        price = max(1, round(sum(max(1, rock.sell_value) for rock in siblings) / len(siblings)))
        pod = FamilyPodListing(pod_id, farm_id, tuple(siblings[0].parent_ids), tuple(rock.id for rock in siblings), price, world.turn, world.turn + 6)
        world.family_pods[pod_id] = pod
        return pod

    def _expire(self, world):
        for listing in world.listings.values():
            if listing.status == ListingStatus.ACTIVE and listing.expires_turn < world.turn:
                self.transactions._release_listing_commitments(world, listing)
                world.release_rock(listing.rock_id, listing.listing_id)
                listing.status = ListingStatus.EXPIRED
        for offer in world.trade_offers.values():
            if offer.status == OfferStatus.OPEN and offer.expires_turn < world.turn:
                self.transactions._release_trade(world, offer)
                offer.status = OfferStatus.EXPIRED
        for pod in world.family_pods.values():
            if pod.status == FamilyPodStatus.ACTIVE and pod.expires_turn < world.turn:
                for child_id in pod.child_ids:
                    world.release_rock(child_id, pod.pod_id)
                pod.status = FamilyPodStatus.EXPIRED
