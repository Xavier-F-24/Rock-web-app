from Rock_AI.actions.farmer_action import AcceptTradeOfferAction, CreateTradeOfferAction
from Rock_AI.environments.multi_farm_economy_environment import MultiFarmEconomyEnvironment


def test_one_for_one_trade_reserves_then_transfers_each_rock_once():
    environment = MultiFarmEconomyEnvironment(seed=60)
    environment.reset()
    sender_id, recipient_id = sorted(environment.world.farms)[:2]
    offer = next(row for row in environment.legal_candidates(sender_id) if isinstance(row.action, CreateTradeOfferAction) and row.action.recipient_farm_id == recipient_id)
    offered = offer.action.offered_rock_ids[0]
    requested = offer.action.requested_rock_ids[0]
    assert environment.execute(offer).success
    assert environment.world.reserved_rock_ids[offered].startswith("offer_")
    accept = next(row for row in environment.legal_candidates(recipient_id) if isinstance(row.action, AcceptTradeOfferAction))
    assert environment.execute(accept).success
    assert environment.world.owner_of(offered) == recipient_id
    assert environment.world.owner_of(requested) == sender_id
    environment.world.validate_ownership()
