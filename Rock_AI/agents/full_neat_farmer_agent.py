"""Runtime agent using one recurrent network for all legal economy actions."""

from dataclasses import dataclass, field


@dataclass
class FullNeatFarmerAgent:
    policy: object
    agent_id: str = "recurrent_neat_full_farmer"
    latest_decision: object = field(default=None, init=False)
    pending_resolutions: dict[str, object] = field(default_factory=dict, init=False)

    @property
    def name(self):
        return self.agent_id

    def reset(self, seed: int = 0, episode_id: str = "economy"):
        self.policy.reset(f"{episode_id}:{seed}")
        self.latest_decision = None
        self.pending_resolutions.clear()

    def choose_candidate(self, observation):
        self.latest_decision = self.policy.rank_actions(observation)
        return self.latest_decision.selected

    def observe_result(self, candidate, result):
        self.policy.commit_selected(candidate, result)
        for key in ("bid_id", "offer_id", "listing_id"):
            related_id = result.public_payload.get(key)
            if related_id:
                self.pending_resolutions[str(related_id)] = candidate

    def observe_public_resolution(self, event):
        related = [str(value) for key, value in event.payload.items() if key in {"bid_id", "offer_id", "listing_id"}]
        for related_id in related:
            candidate = self.pending_resolutions.pop(related_id, None)
            if candidate is not None:
                self.policy.commit_visible_resolution(candidate, event)
