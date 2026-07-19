"""Deterministic mixed baseline and frozen-champion opponent selection."""

import random
from dataclasses import dataclass, field

from Rock_AI.agents.heuristic_full_farmer_agent import HeuristicFullFarmerAgent
from Rock_AI.agents.random_full_farmer_agent import RandomFullFarmerAgent
from Rock_AI.agents.scripted_market_baseline_agent import ScriptedMarketBaselineAgent
from Rock_AI.agents.full_neat_farmer_agent import FullNeatFarmerAgent
from Rock_AI.policies.recurrent_neat_farmer_policy import RecurrentNeatFarmerPolicy


@dataclass
class OpponentPool:
    historical_artifacts: list[str] = field(default_factory=list)
    recent_artifacts: list[str] = field(default_factory=list)
    baseline_fraction: float = .20
    historical_fraction: float = .30
    recent_fraction: float = .30

    def baseline_agents(self):
        return (RandomFullFarmerAgent(), HeuristicFullFarmerAgent(), ScriptedMarketBaselineAgent())

    def select(self, seed: int, count: int = 2):
        rng = random.Random(seed)
        selected = []
        for index in range(count):
            roll = rng.random()
            artifact = None
            if roll < self.baseline_fraction:
                selected.append(self.baseline_agents()[(seed + index) % 3])
                continue
            if roll < self.baseline_fraction + self.historical_fraction and self.historical_artifacts:
                artifact = self.historical_artifacts[(seed + index) % len(self.historical_artifacts)]
            elif roll < self.baseline_fraction + self.historical_fraction + self.recent_fraction and self.recent_artifacts:
                artifact = self.recent_artifacts[(seed + index) % len(self.recent_artifacts)]
            if artifact:
                try:
                    selected.append(FullNeatFarmerAgent(RecurrentNeatFarmerPolicy.load(artifact), f"frozen_champion_{index}"))
                    continue
                except (OSError, ValueError, KeyError):
                    pass
            # The final bucket deliberately varies baseline style and scenario seed.
            selected.append(self.baseline_agents()[(seed * 7 + index * 5) % 3])
        return tuple(selected)
