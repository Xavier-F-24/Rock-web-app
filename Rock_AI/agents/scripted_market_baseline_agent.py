"""Profit-oriented market baseline for league evaluation."""

from Rock_AI.actions.farmer_action import AcceptBidAction, CreateListingAction, PlaceBidAction
from .heuristic_full_farmer_agent import HeuristicFullFarmerAgent


class ScriptedMarketBaselineAgent(HeuristicFullFarmerAgent):
    def __init__(self):
        super().__init__("scripted_profit_trader")

    def choose_candidate(self, observation):
        preferred = [candidate for candidate in observation.legal_candidates if isinstance(candidate.action, (AcceptBidAction, PlaceBidAction, CreateListingAction))]
        return min(preferred, key=lambda row: row.candidate_hash) if preferred else super().choose_candidate(observation)
