"""Deterministic random legal-action baseline."""

import random


class RandomFullFarmerAgent:
    def __init__(self, agent_id="random_full_farmer"):
        self.agent_id = agent_id
        self.rng = random.Random()
        self.latest_decision = None

    @property
    def name(self):
        return self.agent_id

    def reset(self, seed=0, episode_id="economy"):
        self.rng.seed(seed)

    def choose_candidate(self, observation):
        if not observation.legal_candidates:
            return None
        return self.rng.choice(observation.legal_candidates)

    def observe_result(self, candidate, result):
        return None
