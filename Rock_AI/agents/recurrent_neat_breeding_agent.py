"""Breeding agent backed by an open-topology recurrent NEAT policy."""

from __future__ import annotations

from Rock_AI.agents.breeding_agent_helper import BreedPairAction, BreedingAgent, StopGenerationAction
from Rock_AI.datasets.pair_ranking_record_helper import FarmerObjectiveProfile
from Rock_AI.policies.recurrent_neat_pair_ranking_policy import RecurrentNeatPairRankingPolicy
from Rock_AI.representations.player_observation_adapter import PlayerObservationAdapter


class RecurrentNeatBreedingAgent(BreedingAgent):
    def __init__(
        self, policy: RecurrentNeatPairRankingPolicy,
        objective_profile: FarmerObjectiveProfile | None = None, *,
        utility_threshold: float | None = None,
        agent_id: str = "recurrent_neat",
        observation_adapter: PlayerObservationAdapter | None = None,
    ):
        super().__init__(agent_id, objective_profile)
        self.policy = policy
        self.utility_threshold = utility_threshold
        self.observation_adapter = observation_adapter or PlayerObservationAdapter()

    def reset(self, seed: int) -> None:
        super().reset(seed)
        self.policy.reset(f"{self.agent_id}-{seed}")

    def choose_action(self, observation, legal_actions):
        legal_pairs = {
            tuple(sorted((str(action.parent_a_id), str(action.parent_b_id)))): action
            for action in legal_actions if isinstance(action, BreedPairAction)
        }
        if observation.remaining_breeding_actions <= 0:
            return StopGenerationAction("no_remaining_breeding_actions")
        if not legal_pairs:
            return StopGenerationAction("no_legal_pairs")
        recurrent_observation = self.observation_adapter.build_recurrent(observation)
        decision = self.policy.rank_observation(recurrent_observation)
        ranked = [row for row in decision.ranked_pairs if tuple(sorted(map(str, row.parent_ids))) in legal_pairs]
        if not ranked:
            return StopGenerationAction("policy_returned_no_legal_candidate")
        if self.utility_threshold is not None and ranked[0].neural_score < self.utility_threshold:
            return StopGenerationAction("best_utility_below_threshold")
        selected = ranked[0]
        self.last_decision_context = {
            "selected_score": selected.neural_score,
            "scores": {"neural_score": selected.neural_score, "confidence_proxy": decision.confidence_proxy},
            "player_observation_hash": recurrent_observation.player_observation.observation_hash,
            "observation_schema_version": recurrent_observation.player_observation.schema_version,
            "recurrent_state": self.policy.export_state(),
            "ranked_candidate_pairs": [
                {"parent_ids": list(row.parent_ids), "score": row.neural_score, "score_components": row.score_components}
                for row in ranked
            ],
        }
        if self.policy.latest_model_trace is not None:
            self.last_decision_context["model_trace"] = self.policy.latest_model_trace.to_dict()
        return legal_pairs[tuple(sorted(map(str, selected.parent_ids)))]

    def configuration(self):
        return {
            **super().configuration(), "utility_threshold": self.utility_threshold,
            "network_artifact": self.policy.checkpoint_id,
            "topology_id": self.policy.artifact.topology_id,
            "recurrent": True,
        }
