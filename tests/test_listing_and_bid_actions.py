from Rock_AI.actions.farmer_action import AcceptBidAction, CreateListingAction, PlaceBidAction
from Rock_AI.environments.multi_farm_economy_environment import MultiFarmEconomyEnvironment


def test_listing_bid_and_acceptance_transfer_atomically():
    environment = MultiFarmEconomyEnvironment(seed=50)
    environment.reset()
    seller_id, buyer_id = sorted(environment.world.farms)[:2]
    listing_action = next(row for row in environment.legal_candidates(seller_id) if isinstance(row.action, CreateListingAction))
    rock_id = listing_action.action.rock_id
    assert environment.execute(listing_action).success
    assert rock_id in environment.world.reserved_rock_ids
    bid_action = next(row for row in environment.legal_candidates(buyer_id) if isinstance(row.action, PlaceBidAction))
    assert environment.execute(bid_action).success
    buyer_before = environment.world.farm(buyer_id).money
    seller_before = environment.world.farm(seller_id).money
    accept = next(row for row in environment.legal_candidates(seller_id) if isinstance(row.action, AcceptBidAction))
    price = accept.metadata.get("price", bid_action.action.bid_amount)
    result = environment.execute(accept)
    assert result.success
    assert environment.world.owner_of(rock_id) == buyer_id
    assert environment.world.farm(buyer_id).money == buyer_before - bid_action.action.bid_amount
    assert environment.world.farm(seller_id).money == seller_before + bid_action.action.bid_amount
    environment.world.validate_ownership()


def test_idempotency_prevents_duplicate_market_transfer():
    environment = MultiFarmEconomyEnvironment(seed=51)
    environment.reset()
    seller_id = sorted(environment.world.farms)[0]
    candidate = next(row for row in environment.legal_candidates(seller_id) if isinstance(row.action, CreateListingAction))
    first = environment.execute(candidate)
    second = environment.transaction_manager.execute(environment.world, candidate.action, candidate.candidate_hash)
    assert first.success and second.success and second.idempotent_replay
    assert len(environment.world.listings) == 1
