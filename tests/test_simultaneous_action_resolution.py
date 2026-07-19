import copy

from Rock_AI.actions.farmer_action import PassTurnAction
from Rock_AI.environments.multi_farm_economy_environment import MultiFarmEconomyEnvironment


def test_all_farms_observe_same_opening_turn_and_pending_intents_are_not_visible():
    environment = MultiFarmEconomyEnvironment(seed=70)
    environment.reset()
    observations = {farm_id: environment.observe(farm_id) for farm_id in environment.world.farms}
    assert {row.economy.world_turn for row in observations.values()} == {0}
    assert all(not row.economy.offers.public_bid_ids for row in observations.values())
    selected = {farm_id: next(row for row in observation.legal_candidates if isinstance(row.action, PassTurnAction)) for farm_id, observation in observations.items()}
    result = environment.resolve_round(selected)
    assert all(row.success for row in result.action_results)
    assert result.generation_advanced
