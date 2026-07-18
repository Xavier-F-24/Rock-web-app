"""Breeding agent controlled by a separately evolved NEAT policy."""

from __future__ import annotations

from Rock_AI.agents.neural_breeding_agent import NeuralBreedingAgent
from Rock_AI.policies.neat_pair_ranking_policy import NeatPairRankingPolicy


class NeatBreedingAgent(NeuralBreedingAgent):
    def __init__(self, policy: NeatPairRankingPolicy, *args, agent_id: str = "neat", **kwargs):
        super().__init__(policy, *args, agent_id=agent_id, **kwargs)

    def choose_action(self, observation, legal_actions):
        action = super().choose_action(observation, legal_actions)
        if self.policy.latest_model_trace is not None:
            self.last_decision_context["model_trace"] = self.policy.latest_model_trace.to_dict()
        return action

    def configuration(self):
        return {
            **super().configuration(),
            "agent_type": type(self).__name__,
            "network_artifact": self.policy.checkpoint_id,
            "topology_id": self.policy.artifact.topology_id,
        }
