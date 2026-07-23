"""Bounded liveness accounting for multi-farm economy episodes."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class EpisodeTerminationReason(str, Enum):
    FINAL_GENERATION = "final_generation_reached"
    MAX_WORLD_TURNS = "maximum_world_turns_reached"
    MAX_DECISIONS_PER_FARM = "maximum_decisions_per_farm_reached"
    MAX_NO_PROGRESS = "maximum_no_progress_rounds_reached"
    MAX_CONSECUTIVE_PASSES = "maximum_consecutive_passes_reached"
    MAX_FAILED_TRANSACTIONS = "maximum_failed_transactions_reached"
    WALL_CLOCK_TIMEOUT = "episode_wall_clock_timeout"
    ALL_FARMS_BLOCKED = "all_farms_blocked"
    ECONOMY_STALLED = "economy_stalled"
    STATE_CYCLE = "world_state_cycle_detected"
    ENVIRONMENT_FAILURE = "environment_failure"


@dataclass(frozen=True)
class EpisodeLivenessLimits:
    maximum_world_turns: int = 100
    maximum_decisions_per_farm: int = 100
    maximum_no_progress_rounds: int = 8
    maximum_consecutive_passes: int = 6
    maximum_failed_transactions: int = 12
    maximum_wall_clock_seconds: float = 300.0
    cycle_history_size: int = 12
    cycle_repeat_limit: int = 3

    def __post_init__(self) -> None:
        integer_limits = (
            self.maximum_world_turns,
            self.maximum_decisions_per_farm,
            self.maximum_no_progress_rounds,
            self.maximum_consecutive_passes,
            self.maximum_failed_transactions,
            self.cycle_history_size,
            self.cycle_repeat_limit,
        )
        if min(integer_limits) <= 0 or self.maximum_wall_clock_seconds <= 0:
            raise ValueError("Episode liveness limits must be positive")


@dataclass
class EpisodeLivenessState:
    limits: EpisodeLivenessLimits
    clock: Callable[[], float] = time.monotonic
    started_at: float = field(init=False)
    decisions_by_farm: dict[str, int] = field(default_factory=dict)
    failed_transactions: int = 0
    no_progress_rounds: int = 0
    consecutive_pass_rounds: int = 0
    recent_state_hashes: deque[str] = field(init=False)
    state_hash_counts: Counter[str] = field(default_factory=Counter)
    last_completed_operation: str = "reset"

    def __post_init__(self) -> None:
        self.started_at = self.clock()
        self.recent_state_hashes = deque(maxlen=self.limits.cycle_history_size)

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, self.clock() - self.started_at)

    def record_signature(self, signature: str) -> None:
        if len(self.recent_state_hashes) == self.recent_state_hashes.maxlen:
            removed = self.recent_state_hashes[0]
            self.state_hash_counts[removed] -= 1
            if self.state_hash_counts[removed] <= 0:
                del self.state_hash_counts[removed]
        self.recent_state_hashes.append(signature)
        self.state_hash_counts[signature] += 1

    def finite_terminal_fitness(self, base_fitness: float = 0.0) -> float:
        value = float(base_fitness)
        return value if math.isfinite(value) else -10.0


def world_progress_signature(world) -> str:
    """Hash meaningful world state while deliberately excluding the turn counter."""
    farms = []
    for farm_id, farm in sorted(world.farms.items()):
        rocks = tuple(
            (int(rock.id), str(rock.status.value), int(getattr(rock, "value", 0)), int(getattr(rock, "generation", 0)))
            for rock in sorted(farm.rocks.values(), key=lambda row: row.id)
        )
        queue = tuple(
            (int(pair.parent_a_id), int(pair.parent_b_id), tuple(pair.potion_keys))
            for pair in farm.game.breeding_queue
        )
        farms.append(
            (
                farm_id,
                int(farm.money),
                int(farm.committed_money),
                int(farm.generation),
                rocks,
                queue,
                tuple(sorted(farm.potions.items())),
            )
        )
    listings = tuple(
        (
            key,
            str(row.status.value),
            int(row.rock_id),
            tuple(sorted((bid_id, bid.active, int(bid.amount)) for bid_id, bid in row.bids.items())),
        )
        for key, row in sorted(world.listings.items())
    )
    offers = tuple(
        (key, str(row.status.value), tuple(row.offered_rock_ids), tuple(row.requested_rock_ids))
        for key, row in sorted(world.trade_offers.items())
    )
    payload = {
        "generation": int(world.generation),
        "farms": farms,
        "listings": listings,
        "offers": offers,
        "reservations": tuple(sorted((int(key), value) for key, value in world.reserved_rock_ids.items())),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
