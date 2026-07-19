from Rock_AI.evaluation.economy_metrics import economy_metrics


def farmer_comparison_rows(world):
    return tuple({"farm_id": farm_id, "name": farm.profile.display_name, "objective": farm.profile.objective.value, "generation": farm.generation, **economy_metrics(world, farm_id)} for farm_id, farm in sorted(world.farms.items()))
