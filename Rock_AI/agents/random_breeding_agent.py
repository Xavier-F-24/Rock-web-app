"""Uniform deterministic baseline over authoritative legal breeding pairs."""

from __future__ import annotations

from Rock_AI.agents.breeding_agent_helper import (
    AgentAction,
    BreedPairAction,
    BreedingAgent,
    StopGenerationAction,
)
from Rock_AI.datasets.pair_ranking_record_helper import FarmerObjectiveProfile


class RandomBreedingAgent(BreedingAgent):
    def __init__(
        self,
        objective_profile: FarmerObjectiveProfile | None = None,
        *,
        stop_chance: float = 0.0,
        agent_id: str = "random",
    ):
        super().__init__(agent_id, objective_profile)
        if not 0.0 <= stop_chance <= 1.0:
            raise ValueError("stop_chance must be in [0, 1]")
        self.stop_chance = float(stop_chance)

    def choose_action(self, observation, legal_actions) -> AgentAction:
        breed_actions = [action for action in legal_actions if isinstance(action, BreedPairAction)]
        if observation.remaining_breeding_actions <= 0:
            return StopGenerationAction("no_remaining_breeding_actions")
        if not breed_actions:
            return StopGenerationAction("no_legal_pairs")
        if self.stop_chance and self.rng.random() < self.stop_chance:
            return StopGenerationAction("random_stop")
        selected = self.rng.choice(breed_actions)
        self.last_decision_context = {
            "ranked_candidate_pairs": [
                {"parent_ids": [action.parent_a_id, action.parent_b_id]}
                for action in breed_actions
            ]
        }
        return selected

    def configuration(self):
        return {**super().configuration(), "stop_chance": self.stop_chance}
