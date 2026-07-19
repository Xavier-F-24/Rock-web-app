def available_liquidity(farm) -> int:
    return int(farm.money - farm.committed_money)


def public_asset_value(farm) -> int:
    return int(sum(rock.value for rock in farm.rocks.values()))
