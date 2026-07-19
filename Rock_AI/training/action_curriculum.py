"""Stable-width action curriculum for full-farmer evolution."""

from enum import IntEnum

from Rock_AI.actions.action_mask import ActionAvailability
from Rock_AI.actions.farmer_action_type import FarmerActionType


class ActionCurriculumStage(IntEnum):
    BREEDING = 1
    IMPORTS = 2
    POTIONS = 3
    SELLING_LISTINGS = 4
    BIDS = 5
    TRADES = 6
    FULL_ECONOMY = 7
    OPPONENT_GENERALIZATION = 8


def availability_for_stage(stage: ActionCurriculumStage) -> ActionAvailability:
    enabled = {FarmerActionType.BREED_PAIR, FarmerActionType.STOP_BREEDING, FarmerActionType.PASS_TURN}
    if stage >= ActionCurriculumStage.IMPORTS:
        enabled.update({FarmerActionType.IMPORT_RANDOM_ROCK, FarmerActionType.IMPORT_REQUESTED_ROCK})
    if stage >= ActionCurriculumStage.POTIONS:
        enabled.add(FarmerActionType.BUY_POTION)
    if stage >= ActionCurriculumStage.SELLING_LISTINGS:
        enabled.update({FarmerActionType.SELL_ROCK, FarmerActionType.CREATE_LISTING, FarmerActionType.CANCEL_LISTING})
    if stage >= ActionCurriculumStage.BIDS:
        enabled.update({FarmerActionType.PLACE_BID, FarmerActionType.ACCEPT_BID, FarmerActionType.REJECT_BID})
    if stage >= ActionCurriculumStage.TRADES:
        enabled.update({FarmerActionType.CREATE_TRADE_OFFER, FarmerActionType.ACCEPT_TRADE_OFFER, FarmerActionType.REJECT_TRADE_OFFER})
    return ActionAvailability(frozenset(enabled))
