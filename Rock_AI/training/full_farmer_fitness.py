"""Normalized strategic outcome fitness and anti-spam accounting."""

from collections import Counter


def farm_objective_utility(farm) -> float:
    rocks = [rock for rock in farm.rocks.values() if rock.status.value == "active"]
    asset_value = sum(float(rock.value) for rock in rocks)
    maximum = max((float(rock.value) for rock in rocks), default=0.0)
    visible_traits = {(name, str(pair.phenotype)) for rock in rocks for name, pair in rock.genotype.genes.items() if pair.phenotype is not None}
    profile = farm.profile
    return (
        profile.profit_weight * (farm.money + asset_value)
        + profile.maximum_value_weight * maximum
        + profile.diversity_weight * len(visible_traits)
        + profile.liquidity_weight * farm.available_money
    )


def normalized_campaign_fitness(world, controlled_farm_id: str, initial_utility: float) -> float:
    final = farm_objective_utility(world.farm(controlled_farm_id))
    peers = [farm_objective_utility(farm) for farm_id, farm in world.farms.items() if farm_id != controlled_farm_id]
    gain = (final - initial_utility) / max(1.0, abs(initial_utility))
    relative = final / max(1.0, max([final, *peers]))
    return .7 * gain + .3 * relative


def action_spam_diagnostics(decisions):
    counts = Counter(row["selected_action"]["action_type"] for row in decisions)
    total = max(1, sum(counts.values()))
    return {"counts": dict(counts), "pass_fraction": counts.get("pass_turn", 0) / total}
