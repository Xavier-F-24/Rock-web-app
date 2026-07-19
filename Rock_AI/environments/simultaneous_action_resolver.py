"""Resolve private intents against one round-opening state."""

from dataclasses import dataclass

from Rock_AI.actions.farmer_action import AcceptBidAction, AcceptTradeOfferAction, CancelListingAction, FarmerAction, RejectBidAction, RejectTradeOfferAction


@dataclass(frozen=True)
class ActionIntent:
    farm_id: str
    action: FarmerAction
    action_hash: str


class SimultaneousActionResolver:
    def __init__(self, transaction_manager):
        self.transaction_manager = transaction_manager

    def resolve(self, world, intents: tuple[ActionIntent, ...]):
        response_types = (AcceptBidAction, AcceptTradeOfferAction, RejectBidAction, RejectTradeOfferAction, CancelListingAction)
        ordered = sorted(intents, key=lambda intent: (0 if isinstance(intent.action, response_types) else 1, intent.farm_id, intent.action_hash))
        return tuple(self.transaction_manager.execute(world, intent.action, intent.action_hash) for intent in ordered)
