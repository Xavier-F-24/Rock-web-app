"""Authoritative deterministic three-farm breeding and economy environment."""

from __future__ import annotations

import copy
from dataclasses import dataclass

from Rock_AI.actions.farmer_action import PassTurnAction, StopBreedingAction
from Rock_AI.economy.transaction_validator import EconomyTransactionManager
from Rock_AI.policies.market_action_policy_adapter import ActionCandidateLimits, LegalFarmerActionGenerator
from Rock_World.rock_world_manager_helper import create_starter_world

from .market_round_environment import MarketRoundResult
from .private_farmer_observation_adapter import PrivateFarmerObservationAdapter
from .simultaneous_action_resolver import ActionIntent, SimultaneousActionResolver
from .world_turn_scheduler import SchedulerMode, WorldTurnScheduler


@dataclass(frozen=True)
class MultiFarmEconomyConfig:
    max_generations: int = 7
    max_world_turns: int = 100
    turns_per_generation: int = 6
    scheduler_mode: SchedulerMode = SchedulerMode.SIMULTANEOUS
    candidate_limits: ActionCandidateLimits = ActionCandidateLimits()


class MultiFarmEconomyEnvironment:
    def __init__(self, seed: int = 0, config: MultiFarmEconomyConfig | None = None):
        self.seed = int(seed)
        self.config = config or MultiFarmEconomyConfig()
        self.transaction_manager = EconomyTransactionManager()
        self.candidate_generator = LegalFarmerActionGenerator(self.config.candidate_limits)
        self.observation_adapter = PrivateFarmerObservationAdapter(self.candidate_generator)
        self.scheduler = WorldTurnScheduler(self.config.scheduler_mode)
        self.resolver = SimultaneousActionResolver(self.transaction_manager)
        self.world = None
        self.turn_in_generation = 0
        self.terminated = False
        self.termination_reason = None
        self.action_history = []

    def reset(self, seed: int | None = None, initial_world=None):
        if seed is not None:
            self.seed = int(seed)
        self.world = copy.deepcopy(initial_world) if initial_world is not None else create_starter_world(seed=self.seed)
        self.turn_in_generation = 0
        self.terminated = False
        self.termination_reason = None
        self.action_history = []
        return self.world

    def snapshot(self):
        return copy.deepcopy((self.world, self.turn_in_generation, self.terminated, self.termination_reason, self.action_history))

    def restore(self, snapshot) -> None:
        self.world, self.turn_in_generation, self.terminated, self.termination_reason, self.action_history = copy.deepcopy(snapshot)

    def observe(self, farm_id: str, recurrent_state=None):
        self._require_ready()
        return self.observation_adapter.build(self.world, farm_id, recurrent_state=recurrent_state)

    def legal_candidates(self, farm_id: str):
        return self.observe(farm_id).legal_candidates

    def execute(self, candidate):
        self._require_ready()
        legal = {row.candidate_hash: row for row in self.legal_candidates(candidate.action.actor_farm_id)}
        if candidate.candidate_hash not in legal:
            raise ValueError("Selected action is not in the authoritative legal candidate set")
        result = self.transaction_manager.execute(self.world, legal[candidate.candidate_hash].action, candidate.candidate_hash)
        self.action_history.append(result)
        return result

    def resolve_round(self, candidates_by_farm: dict[str, object]) -> MarketRoundResult:
        self._require_ready()
        opening_candidates = {farm_id: {row.candidate_hash: row for row in self.legal_candidates(farm_id)} for farm_id in self.world.farms}
        intents = []
        order = self.scheduler.order(self.world)
        for farm_id in order:
            selected = candidates_by_farm[farm_id]
            candidate = opening_candidates[farm_id].get(selected.candidate_hash)
            if candidate is None:
                raise ValueError(f"Farm {farm_id} submitted a non-legal opening-state action")
            intents.append(ActionIntent(farm_id, candidate.action, candidate.candidate_hash))
        results = self.resolver.resolve(self.world, tuple(intents))
        self.action_history.extend(results)
        all_pass = all(isinstance(intent.action, (PassTurnAction, StopBreedingAction)) for intent in intents)
        self.world.turn += 1
        self.turn_in_generation += 1
        advanced = all_pass or self.turn_in_generation >= self.config.turns_per_generation
        if advanced:
            self.advance_generation()
        self._check_termination()
        return MarketRoundResult(self.world.turn - 1, order, results, advanced)

    def advance_generation(self):
        created = []
        for farm in self.world.farms.values():
            farm.game.next_rock_id = max([*self.world.owner_by_rock_id, farm.game.next_rock_id - 1], default=0) + 1
            before = set(farm.rocks)
            children = farm.game.breed_queue()
            farm.game.generation += 1
            farm.game.game_over = farm.game.generation >= self.config.max_generations
            for rock_id in set(farm.rocks) - before:
                if rock_id in self.world.owner_by_rock_id:
                    raise ValueError(f"Duplicate child rock ID {rock_id}")
                self.world.owner_by_rock_id[rock_id] = farm.farm_id
                farm.visible_rock_ids.add(rock_id)
            created.extend(children)
        self.world.generation += 1
        self.turn_in_generation = 0
        self._expire_economy_items()
        return created

    def _expire_economy_items(self):
        from Rock_Market.rock_npc_market_helper import ListingStatus, OfferStatus
        for listing in self.world.listings.values():
            if listing.status == ListingStatus.ACTIVE and listing.expires_turn < self.world.turn:
                self.transaction_manager._release_listing_commitments(self.world, listing)
                self.world.release_rock(listing.rock_id, listing.listing_id)
                listing.status = ListingStatus.EXPIRED
        for offer in self.world.trade_offers.values():
            if offer.status == OfferStatus.OPEN and offer.expires_turn < self.world.turn:
                self.transaction_manager._release_trade(self.world, offer)
                offer.status = OfferStatus.EXPIRED

    def _check_termination(self):
        if self.world.generation >= self.config.max_generations:
            self.terminated, self.termination_reason = True, "final_generation_reached"
        elif self.world.turn >= self.config.max_world_turns:
            self.terminated, self.termination_reason = True, "maximum_world_turns_reached"

    def _require_ready(self):
        if self.world is None:
            raise RuntimeError("Call reset() before using the economy environment")
        if self.terminated:
            raise RuntimeError(f"Environment terminated: {self.termination_reason}")
