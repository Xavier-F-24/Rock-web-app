from Rock_AI.actions.farmer_action import BreedPairAction, BuyPotionAction
from Rock_AI.environments.multi_farm_economy_environment import MultiFarmEconomyEnvironment


def test_potion_purchase_and_pair_allocation_use_core_inventory_rules():
    environment = MultiFarmEconomyEnvironment(seed=40)
    environment.reset()
    farm_id = sorted(environment.world.farms)[0]
    buy = next(row for row in environment.legal_candidates(farm_id) if isinstance(row.action, BuyPotionAction) and row.action.potion_type == "mutation")
    money = environment.world.farm(farm_id).money
    assert environment.execute(buy).success
    farm = environment.world.farm(farm_id)
    assert farm.money == money - buy.action.quoted_cost and farm.potions["mutation"] == 1
    breed = next(row for row in environment.legal_candidates(farm_id) if isinstance(row.action, BreedPairAction) and row.action.potion_keys == ("mutation",))
    assert environment.execute(breed).success
    assert "mutation" not in farm.potions
    assert farm.game.breeding_queue[0].potion_keys == ["mutation"]
