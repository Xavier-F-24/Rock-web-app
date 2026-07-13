"""Framework-neutral deterministic environment base class."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EnvironmentSnapshot:
    seed: int
    rng_state: object
    state: Any


class RockTrainingEnvironment:
    """Own a local RNG and copyable episode state without touching global random."""

    def __init__(self, seed: int = 0):
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.state: Any = None

    def reset(self, seed: int | None = None) -> Any:
        if seed is not None:
            self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.state = None
        return self.state

    def snapshot(self) -> EnvironmentSnapshot:
        return EnvironmentSnapshot(
            seed=self.seed,
            rng_state=copy.deepcopy(self.rng.getstate()),
            state=copy.deepcopy(self.state),
        )

    def clone_state(self) -> EnvironmentSnapshot:
        return self.snapshot()

    def restore(self, snapshot: EnvironmentSnapshot) -> Any:
        if not isinstance(snapshot, EnvironmentSnapshot):
            raise TypeError("snapshot must be an EnvironmentSnapshot")
        self.seed = int(snapshot.seed)
        self.rng = random.Random()
        self.rng.setstate(copy.deepcopy(snapshot.rng_state))
        self.state = copy.deepcopy(snapshot.state)
        return self.state

    def restore_state(self, snapshot: EnvironmentSnapshot) -> Any:
        return self.restore(snapshot)
