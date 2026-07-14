"""Fast transparent pair heuristic used when no world FarmerPolicy is available."""

from __future__ import annotations

from Rock_AI.agents.breeding_agent_helper import (
    AgentAction,
    BreedPairAction,
    BreedingAgent,
    StopGenerationAction,
)
from Rock_AI.datasets.pair_ranking_record_helper import FarmerObjectiveProfile
from Rock_AI.models.pair_scoring_helper import pair_diversity_features


class HeuristicBreedingAgent(BreedingAgent):
    """Inference-only fallback; the repository currently has no FarmerPolicy class."""

    def __init__(self, objective_profile: FarmerObjectiveProfile | None = None, agent_id: str = "heuristic"):
        super().__init__(agent_id, objective_profile)

    def _score(self, farm, action: BreedPairAction) -> float:
        parent_a = farm.get_rock(action.parent_a_id)
        parent_b = farm.get_rock(action.parent_b_id)
        allele_diversity, phenotype_diversity = pair_diversity_features(parent_a, parent_b)
        objective = self.objective_profile
        return float(
            (parent_a.value + parent_b.value) * objective.immediate_expected_value_weight
            + max(parent_a.value, parent_b.value) * objective.maximum_offspring_value_weight
            + allele_diversity * objective.genotype_diversity_weight
            + phenotype_diversity * objective.phenotype_diversity_weight
        )

    def choose_action(self, observation, legal_actions) -> AgentAction:
        candidates = [action for action in legal_actions if isinstance(action, BreedPairAction)]
        if observation.remaining_breeding_actions <= 0 or not candidates:
            return StopGenerationAction("no_heuristic_candidate")
        scored = sorted(
            ((self._score(observation.farm, action), action) for action in candidates),
            key=lambda item: (-item[0], str(item[1].parent_a_id), str(item[1].parent_b_id)),
        )
        score, selected = scored[0]
        self.last_decision_context = {
            "selected_score": score,
            "scores": {"heuristic_score": score},
            "ranked_candidate_pairs": [
                {"parent_ids": [action.parent_a_id, action.parent_b_id], "score": value}
                for value, action in scored
            ],
        }
        return selected
