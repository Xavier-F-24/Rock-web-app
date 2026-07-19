"""Deterministic compatible baseline and champion opponent selection."""

from dataclasses import dataclass, field

from Rock_AI.agents.heuristic_full_farmer_agent import HeuristicFullFarmerAgent
from Rock_AI.agents.random_full_farmer_agent import RandomFullFarmerAgent
from Rock_AI.agents.scripted_market_baseline_agent import ScriptedMarketBaselineAgent


@dataclass
class OpponentPool:
    historical_artifacts: list[str] = field(default_factory=list)

    def baseline_agents(self):
        return (RandomFullFarmerAgent(), HeuristicFullFarmerAgent(), ScriptedMarketBaselineAgent())

    def select(self, seed: int, count: int = 2):
        baselines = self.baseline_agents()
        return tuple(baselines[(seed + index) % len(baselines)] for index in range(count))
