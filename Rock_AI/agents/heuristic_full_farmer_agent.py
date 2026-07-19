"""Conservative player-visible full-farmer baseline."""

from Rock_AI.actions.farmer_action import (
    AcceptBidAction, AcceptTradeOfferAction, BreedPairAction, BuyPotionAction,
    ImportRandomRockAction, PassTurnAction, PlaceBidAction, SellRockAction,
)


class HeuristicFullFarmerAgent:
    def __init__(self, agent_id="heuristic_full_farmer"):
        self.agent_id = agent_id

    @property
    def name(self):
        return self.agent_id

    def reset(self, seed=0, episode_id="economy"):
        return None

    def choose_candidate(self, observation):
        def score(candidate):
            action = candidate.action
            if isinstance(action, AcceptTradeOfferAction): return 9.0
            if isinstance(action, AcceptBidAction): return 8.0
            if isinstance(action, BreedPairAction): return 7.0
            if isinstance(action, SellRockAction): return 4.0 + action.quoted_sale_value / 100.0
            if isinstance(action, PlaceBidAction): return 3.0
            if isinstance(action, ImportRandomRockAction): return 2.0
            if isinstance(action, BuyPotionAction): return 1.5
            if isinstance(action, PassTurnAction): return -1.0
            return 0.0
        return max(observation.legal_candidates, key=lambda row: (score(row), row.candidate_hash), default=None)

    def observe_result(self, candidate, result):
        return None
