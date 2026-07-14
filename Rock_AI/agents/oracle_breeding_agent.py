"""Expensive PairEvaluator reference agent for small evaluation campaigns only."""

from __future__ import annotations

from Rock_AI.agents.breeding_agent_helper import (
    AgentAction,
    BreedPairAction,
    BreedingAgent,
    StopGenerationAction,
)
from Rock_AI.datasets.pair_ranking_record_helper import FarmerObjectiveProfile
from Rock_AI.evaluation.pair_evaluator import PairEvaluator
from Rock_AI.models.pair_scoring_helper import score_pair_evaluation


class OracleBreedingAgent(BreedingAgent):
    production_safe = False

    def __init__(
        self,
        objective_profile: FarmerObjectiveProfile | None = None,
        *,
        trial_count: int = 100,
        agent_id: str = "oracle",
        evaluator: PairEvaluator | None = None,
    ):
        super().__init__(agent_id, objective_profile)
        if trial_count <= 0:
            raise ValueError("trial_count must be positive")
        self.trial_count = int(trial_count)
        self.evaluator = evaluator or PairEvaluator()
        self._decision_index = 0

    def reset(self, seed: int) -> None:
        super().reset(seed)
        self._decision_index = 0

    def choose_action(self, observation, legal_actions) -> AgentAction:
        candidates = [action for action in legal_actions if isinstance(action, BreedPairAction)]
        if observation.remaining_breeding_actions <= 0 or not candidates:
            return StopGenerationAction("no_oracle_candidate")
        scored = []
        for index, action in enumerate(candidates):
            parent_a = observation.farm.get_rock(action.parent_a_id)
            parent_b = observation.farm.get_rock(action.parent_b_id)
            evaluation = self.evaluator.evaluate_pair(
                parent_a,
                parent_b,
                rules=observation.breeding_rules,
                trial_count=self.trial_count,
                seed=self.seed + self._decision_index * 10_000 + index,
                game=observation.farm,
            )
            utility = score_pair_evaluation(evaluation, self.objective_profile)
            scored.append((utility.score, action, evaluation, utility))
        scored.sort(key=lambda item: (-item[0], str(item[1].parent_a_id), str(item[1].parent_b_id)))
        self._decision_index += 1
        score, selected, evaluation, utility = scored[0]
        self.last_decision_context = {
            "selected_score": score,
            "pair_evaluator_utility": score,
            "scores": {"oracle_pair_utility": score, **utility.contributions},
            "ranked_candidate_pairs": [
                {"parent_ids": [action.parent_a_id, action.parent_b_id], "score": candidate_score}
                for candidate_score, action, _, _ in scored
            ],
            "predictor_outputs": {
                "expected_child_value": evaluation.expectation.expected_child_value.mean,
                "expected_survivor_count": evaluation.expectation.expected_survivor_count.mean,
                "expected_maximum_child_value": evaluation.expectation.expected_maximum_child_value.mean,
            },
        }
        return selected

    def configuration(self):
        return {**super().configuration(), "trial_count": self.trial_count, "production_safe": False}
