"""Breeding agent backed by NeuralPairRankingPolicy."""

from __future__ import annotations

import numpy as np

from Rock_AI.agents.breeding_agent_helper import (
    AgentAction,
    BreedPairAction,
    BreedingAgent,
    StopGenerationAction,
)
from Rock_AI.datasets.pair_ranking_record_helper import FarmerObjectiveProfile
from Rock_AI.policies.neural_pair_ranking_policy import NeuralPairRankingPolicy


class NeuralBreedingAgent(BreedingAgent):
    def __init__(
        self,
        policy: NeuralPairRankingPolicy,
        objective_profile: FarmerObjectiveProfile | None = None,
        *,
        utility_threshold: float | None = None,
        confidence_threshold: float | None = None,
        temperature: float = 0.0,
        agent_id: str = "neural",
    ):
        super().__init__(agent_id, objective_profile)
        if temperature < 0:
            raise ValueError("temperature cannot be negative")
        if confidence_threshold is not None and not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        self.policy = policy
        self.utility_threshold = utility_threshold
        self.confidence_threshold = confidence_threshold
        self.temperature = float(temperature)

    def choose_action(self, observation, legal_actions) -> AgentAction:
        legal_pairs = {
            tuple(sorted((str(action.parent_a_id), str(action.parent_b_id)))): action
            for action in legal_actions
            if isinstance(action, BreedPairAction)
        }
        if observation.remaining_breeding_actions <= 0:
            return StopGenerationAction("no_remaining_breeding_actions")
        if not legal_pairs:
            return StopGenerationAction("no_legal_pairs")
        decision = self.policy.rank_legal_pairs(
            observation.farm,
            observation.breeding_rules,
            self.objective_profile,
        )
        ranked = [
            row
            for row in decision.ranked_pairs
            if tuple(sorted(map(str, row.parent_ids))) in legal_pairs
        ]
        if not ranked:
            return StopGenerationAction("policy_returned_no_legal_candidate")
        best_score = ranked[0].neural_score
        if self.utility_threshold is not None and best_score < self.utility_threshold:
            return StopGenerationAction("best_utility_below_threshold")
        if self.confidence_threshold is not None and decision.confidence_proxy < self.confidence_threshold:
            return StopGenerationAction("policy_confidence_below_threshold")
        if self.temperature > 0 and len(ranked) > 1:
            scores = np.asarray([row.neural_score for row in ranked], dtype=np.float64)
            logits = (scores - scores.max()) / self.temperature
            probabilities = np.exp(logits) / np.exp(logits).sum()
            selected_index = self.rng.choices(range(len(ranked)), weights=probabilities, k=1)[0]
        else:
            selected_index = 0
        selected = ranked[selected_index]
        action = legal_pairs[tuple(sorted(map(str, selected.parent_ids)))]
        self.last_decision_context = {
            "selected_score": selected.neural_score,
            "scores": {
                "neural_score": selected.neural_score,
                "confidence_proxy": decision.confidence_proxy,
            },
            "predictor_outputs": selected.predicted_breeding_outcomes,
            "ranked_candidate_pairs": [
                {
                    "parent_ids": list(row.parent_ids),
                    "score": row.neural_score,
                    "predicted_breeding_outcomes": row.predicted_breeding_outcomes,
                    "score_components": row.score_components,
                }
                for row in ranked
            ],
        }
        return action

    def configuration(self):
        return {
            **super().configuration(),
            "utility_threshold": self.utility_threshold,
            "confidence_threshold": self.confidence_threshold,
            "temperature": self.temperature,
        }
