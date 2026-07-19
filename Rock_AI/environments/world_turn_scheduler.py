"""Deterministic farm scheduling modes."""

import hashlib
import random
from dataclasses import dataclass
from enum import Enum


class SchedulerMode(str, Enum):
    SEQUENTIAL_SEEDED = "sequential_seeded"
    SIMULTANEOUS = "simultaneous"


@dataclass(frozen=True)
class WorldTurnScheduler:
    mode: SchedulerMode = SchedulerMode.SEQUENTIAL_SEEDED

    def order(self, world) -> tuple[str, ...]:
        farms = sorted(world.farms)
        seed = int(hashlib.sha256(f"{world.seed}:{world.turn}".encode()).hexdigest()[:16], 16)
        random.Random(seed).shuffle(farms)
        return tuple(farms)
