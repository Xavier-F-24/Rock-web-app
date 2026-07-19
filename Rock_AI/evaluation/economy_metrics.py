from Rock_AI.training.full_farmer_fitness import farm_objective_utility


def economy_metrics(world, farm_id: str):
    farm = world.farm(farm_id)
    active = [rock for rock in farm.rocks.values() if rock.status.value == "active"]
    return {
        "cash": farm.money, "committed_cash": farm.committed_money,
        "active_rock_value": sum(rock.value for rock in active),
        "maximum_rock_value": max((rock.value for rock in active), default=0),
        "objective_utility": farm_objective_utility(farm),
        "imports": sum(event.event_type.startswith("import_") and farm_id in event.farm_ids for event in world.public_events),
        "transactions": sum(farm_id in event.farm_ids for event in world.public_events),
    }
