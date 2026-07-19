def money_asset_flow_rows(world):
    return tuple({"farm_id": farm_id, "cash": farm.money, "committed_cash": farm.committed_money, "rock_value": sum(rock.value for rock in farm.rocks.values() if rock.status.value == "active"), "potion_count": sum(farm.potions.values())} for farm_id, farm in sorted(world.farms.items()))
