"""Reconstruct production farmers from safe policy IDs, never saved objects."""

from __future__ import annotations

from pathlib import Path

from Rock_AI.agents.full_neat_farmer_agent import FullNeatFarmerAgent
from Rock_AI.agents.heuristic_full_farmer_agent import HeuristicFullFarmerAgent
from Rock_AI.policies.recurrent_neat_farmer_policy import RecurrentNeatFarmerPolicy


class FarmerPolicyRegistry:
    def __init__(self, repository_root: str | Path | None = None):
        self.root = Path(repository_root).resolve() if repository_root else Path(__file__).resolve().parents[1]

    def safe_champions(self) -> tuple[Path, ...]:
        return tuple(sorted(self.root.glob("training_runs/*/champions/best_validation/network.json")))

    def build(self, farm):
        spec = farm.controller
        if spec.policy_id in {"auto", "heuristic"}:
            champions = self.safe_champions() if spec.policy_id == "auto" else ()
            if champions and spec.seed % 3 == 0:
                try:
                    policy = RecurrentNeatFarmerPolicy.load(champions[spec.seed % len(champions)])
                    agent = FullNeatFarmerAgent(policy, f"neural_{farm.farm_id}")
                    agent.reset(spec.seed, "playable_world")
                    if spec.policy_state:
                        policy.import_state(spec.policy_state)
                    return agent
                except (OSError, ValueError, KeyError) as error:
                    spec.warning = f"Neural policy unavailable; using heuristic: {error}"
            return HeuristicFullFarmerAgent(f"heuristic_{farm.farm_id}")
        spec.warning = f"Unknown policy ID {spec.policy_id!r}; using heuristic"
        return HeuristicFullFarmerAgent(f"heuristic_{farm.farm_id}")

    @staticmethod
    def save_state(farm, agent) -> None:
        policy = getattr(agent, "policy", None)
        farm.controller.policy_state = policy.export_state() if policy and hasattr(policy, "export_state") else {}
