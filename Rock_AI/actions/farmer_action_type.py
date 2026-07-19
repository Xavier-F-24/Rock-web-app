"""Stable action type vocabulary shared by training and runtime."""

from enum import Enum


class FarmerActionType(str, Enum):
    BREED_PAIR = "breed_pair"
    STOP_BREEDING = "stop_breeding"
    IMPORT_RANDOM_ROCK = "import_random_rock"
    IMPORT_REQUESTED_ROCK = "import_requested_rock"
    BUY_POTION = "buy_potion"
    SELL_ROCK = "sell_rock"
    CREATE_LISTING = "create_listing"
    CANCEL_LISTING = "cancel_listing"
    PLACE_BID = "place_bid"
    ACCEPT_BID = "accept_bid"
    REJECT_BID = "reject_bid"
    CREATE_TRADE_OFFER = "create_trade_offer"
    ACCEPT_TRADE_OFFER = "accept_trade_offer"
    REJECT_TRADE_OFFER = "reject_trade_offer"
    PASS_TURN = "pass_turn"


ACTION_TYPE_ORDER = tuple(FarmerActionType)
