import json

from Rock_AI.agents.heuristic_full_farmer_agent import HeuristicFullFarmerAgent
from Rock_AI.environments.multi_farm_economy_environment import MultiFarmEconomyEnvironment
from Rock_AI.environments.world_episode_runner import MultiFarmEpisodeRunner
from Rock_Serialization.rock_serialization_helper import world_from_dict, world_to_dict


def test_episode_snapshots_roundtrip_and_do_not_duplicate_transactions():
    environment = MultiFarmEconomyEnvironment(seed=80)
    environment.reset()
    agents = {farm_id: HeuristicFullFarmerAgent(f"heuristic-{farm_id}") for farm_id in environment.world.farms}
    record = MultiFarmEpisodeRunner(environment, agents).run(seed=80, max_rounds=2)
    json.dumps(record.to_dict())
    final = world_from_dict(record.final_world)
    assert world_to_dict(final) == record.final_world
    final.validate_ownership()
