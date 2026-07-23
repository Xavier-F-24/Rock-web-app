"""Authoritative deterministic three-farm breeding and economy environment."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

from Rock_AI.actions.farmer_action import PassTurnAction, StopBreedingAction
from Rock_AI.economy.transaction_validator import EconomyTransactionManager
from Rock_AI.economy.reservation_audit_helper import audit_transaction_reservations
from Rock_AI.policies.market_action_policy_adapter import ActionCandidateLimits, LegalFarmerActionGenerator
from Rock_World.rock_world_manager_helper import create_starter_world

from .market_round_environment import MarketRoundResult
from .episode_liveness_helper import (
    EpisodeLivenessLimits, EpisodeLivenessState, EpisodeTerminationReason,
    world_progress_signature,
)
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
    liveness_limits: EpisodeLivenessLimits = field(default_factory=EpisodeLivenessLimits)


class MultiFarmEconomyEnvironment:
    def __init__(self, seed: int = 0, config: MultiFarmEconomyConfig | None = None, *, heartbeat_callback=None, clock=None):
        self.seed = int(seed)
        self.config = config or MultiFarmEconomyConfig()
        self.transaction_manager = EconomyTransactionManager()
        self.heartbeat_callback = heartbeat_callback
        self.clock = clock or time.monotonic
        self.candidate_generator = LegalFarmerActionGenerator(
            self.config.candidate_limits,
            heartbeat_callback=self._heartbeat,
            clock=self.clock,
        )
        self.observation_adapter = PrivateFarmerObservationAdapter(self.candidate_generator)
        self.scheduler = WorldTurnScheduler(self.config.scheduler_mode)
        self.resolver = SimultaneousActionResolver(self.transaction_manager)
        self.world = None
        self.turn_in_generation = 0
        self.terminated = False
        self.termination_reason = None
        self.action_history = []
        self.liveness = EpisodeLivenessState(self.config.liveness_limits, clock=self.clock)

    def reset(self, seed: int | None = None, initial_world=None):
        if seed is not None:
            self.seed = int(seed)
        self.world = copy.deepcopy(initial_world) if initial_world is not None else create_starter_world(seed=self.seed)
        self.turn_in_generation = 0
        self.terminated = False
        self.termination_reason = None
        self.action_history = []
        self.liveness = EpisodeLivenessState(self.config.liveness_limits, clock=self.clock)
        self.liveness.record_signature(world_progress_signature(self.world))
        audit_transaction_reservations(self.world, repair=True)
        return self.world

    def snapshot(self):
        return copy.deepcopy((self.world, self.turn_in_generation, self.terminated, self.termination_reason, self.action_history, self.liveness))

    def restore(self, snapshot) -> None:
        self.world, self.turn_in_generation, self.terminated, self.termination_reason, self.action_history, self.liveness = copy.deepcopy(snapshot)

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
        self._heartbeat("world_episode", {"operation": "round_started", "world_turn": self.world.turn})
        before_signature = world_progress_signature(self.world)
        opening_candidates = {farm_id: {row.candidate_hash: row for row in self.legal_candidates(farm_id)} for farm_id in self.world.farms}
        intents = []
        order = self.scheduler.order(self.world)
        for farm_id in order:
            selected = candidates_by_farm.get(farm_id)
            candidate = opening_candidates[farm_id].get(getattr(selected, "candidate_hash", None))
            if candidate is None:
                self.liveness.failed_transactions += 1
                candidate = self._pass_candidate(opening_candidates[farm_id])
            intents.append(ActionIntent(farm_id, candidate.action, candidate.candidate_hash))
            self.liveness.decisions_by_farm[farm_id] = self.liveness.decisions_by_farm.get(farm_id, 0) + 1
        try:
            results = self.resolver.resolve(self.world, tuple(intents))
        except Exception:
            audit_transaction_reservations(self.world, repair=True)
            self._terminate(EpisodeTerminationReason.ENVIRONMENT_FAILURE)
            raise
        self.action_history.extend(results)
        self.liveness.failed_transactions += sum(not row.success for row in results)
        all_pass = all(isinstance(intent.action, (PassTurnAction, StopBreedingAction)) for intent in intents)
        self.world.turn += 1
        self.turn_in_generation += 1
        advanced = all_pass or self.turn_in_generation >= self.config.turns_per_generation
        if advanced:
            self.advance_generation()
        audit_transaction_reservations(self.world, repair=True)
        after_signature = world_progress_signature(self.world)
        self.liveness.no_progress_rounds = self.liveness.no_progress_rounds + 1 if after_signature == before_signature else 0
        self.liveness.consecutive_pass_rounds = self.liveness.consecutive_pass_rounds + 1 if all_pass else 0
        self.liveness.record_signature(after_signature)
        self.liveness.last_completed_operation = "round_resolved"
        meaningful = (PassTurnAction, StopBreedingAction)
        all_blocked = all(not any(not isinstance(row.action, meaningful) for row in rows.values()) for rows in opening_candidates.values())
        if all_blocked:
            self._terminate(EpisodeTerminationReason.ALL_FARMS_BLOCKED)
        self._check_termination()
        self._heartbeat("world_episode", {"operation": "round_completed", "world_turn": self.world.turn}, force=True)
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
        audit_transaction_reservations(self.world, repair=True)

    def _check_termination(self):
        if self.terminated:
            return
        limits = self.liveness.limits
        if self.world.generation >= self.config.max_generations:
            self._terminate(EpisodeTerminationReason.FINAL_GENERATION)
        elif self.world.turn >= min(self.config.max_world_turns, limits.maximum_world_turns):
            self._terminate(EpisodeTerminationReason.MAX_WORLD_TURNS)
        elif any(count >= limits.maximum_decisions_per_farm for count in self.liveness.decisions_by_farm.values()):
            self._terminate(EpisodeTerminationReason.MAX_DECISIONS_PER_FARM)
        elif self.liveness.failed_transactions >= limits.maximum_failed_transactions:
            self._terminate(EpisodeTerminationReason.MAX_FAILED_TRANSACTIONS)
        elif self.liveness.no_progress_rounds >= limits.maximum_no_progress_rounds:
            self._terminate(EpisodeTerminationReason.ECONOMY_STALLED)
        elif self.liveness.consecutive_pass_rounds >= limits.maximum_consecutive_passes:
            self._terminate(EpisodeTerminationReason.MAX_CONSECUTIVE_PASSES)
        elif self.liveness.elapsed_seconds >= limits.maximum_wall_clock_seconds:
            self._terminate(EpisodeTerminationReason.WALL_CLOCK_TIMEOUT)
        elif any(count >= limits.cycle_repeat_limit for count in self.liveness.state_hash_counts.values()):
            self._terminate(EpisodeTerminationReason.STATE_CYCLE)

    @staticmethod
    def _pass_candidate(candidates):
        for candidate in candidates.values():
            if isinstance(candidate.action, PassTurnAction):
                return candidate
        raise RuntimeError("Authoritative candidate generation omitted PassTurnAction")

    def _terminate(self, reason: EpisodeTerminationReason) -> None:
        if not self.terminated:
            self.terminated = True
            self.termination_reason = reason

    def terminal_fitness(self, base_fitness: float = 0.0) -> float:
        penalty = {
            EpisodeTerminationReason.ENVIRONMENT_FAILURE: 2.0,
            EpisodeTerminationReason.MAX_FAILED_TRANSACTIONS: 1.0,
            EpisodeTerminationReason.WALL_CLOCK_TIMEOUT: 1.0,
            EpisodeTerminationReason.STATE_CYCLE: 0.5,
            EpisodeTerminationReason.ECONOMY_STALLED: 0.25,
        }.get(self.termination_reason, 0.0)
        return self.liveness.finite_terminal_fitness(base_fitness - penalty)

    def diagnostic_snapshot(self) -> dict[str, object]:
        counts = {}
        if self.world is not None:
            for farm_id in sorted(self.world.farms):
                try:
                    record = self.candidate_generator.pruning_records.get(farm_id, {})
                    counts[farm_id] = dict(record.get("counts_by_type", {}))
                except Exception:
                    counts[farm_id] = {}
        return {
            "world_turn": None if self.world is None else self.world.turn,
            "termination_reason": None if self.termination_reason is None else self.termination_reason.value,
            "legal_action_counts_by_type": counts,
            "current_reservations": {} if self.world is None else dict(self.world.reserved_rock_ids),
            "decisions_by_farm": dict(self.liveness.decisions_by_farm),
            "failed_transactions": self.liveness.failed_transactions,
            "consecutive_passes": self.liveness.consecutive_pass_rounds,
            "no_progress_counter": self.liveness.no_progress_rounds,
            "recent_state_hashes": list(self.liveness.recent_state_hashes),
            "last_completed_operation": self.liveness.last_completed_operation,
            "elapsed_seconds": self.liveness.elapsed_seconds,
        }

    def _heartbeat(self, phase, payload, *, force=False):
        if self.heartbeat_callback:
            self.heartbeat_callback(phase, payload)

    def _require_ready(self):
        if self.world is None:
            raise RuntimeError("Call reset() before using the economy environment")
        if self.terminated:
            raise RuntimeError(f"Environment terminated: {self.termination_reason}")
