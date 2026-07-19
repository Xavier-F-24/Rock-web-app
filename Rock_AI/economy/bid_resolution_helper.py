from .transaction_validator import EconomyTransactionManager


def legal_bid_amounts(listing, available_money: int):
    return EconomyTransactionManager.legal_bid_menu(listing, available_money)
