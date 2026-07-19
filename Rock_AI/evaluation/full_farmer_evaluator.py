"""Paired multi-farm evaluation on identical worlds and seeds."""

from statistics import mean, pstdev

from Rock_AI.agents.full_neat_farmer_agent import FullNeatFarmerAgent
from Rock_AI.agents.heuristic_full_farmer_agent import HeuristicFullFarmerAgent
from Rock_AI.agents.random_full_farmer_agent import RandomFullFarmerAgent
from Rock_AI.environments.multi_farm_economy_environment import MultiFarmEconomyEnvironment
from Rock_AI.environments.world_episode_runner import MultiFarmEpisodeRunner
from Rock_AI.policies.recurrent_neat_farmer_policy import RecurrentNeatFarmerPolicy
from Rock_Serialization.rock_serialization_helper import world_from_dict

from .economy_metrics import economy_metrics


class FullFarmerEvaluator:
    def evaluate(self, champion_path, *, episodes: int, seed: int):
        rows = []
        for episode_index in range(episodes):
            episode_seed = seed + episode_index * 1009
            environment = MultiFarmEconomyEnvironment(episode_seed)
            environment.reset()
            farm_ids = sorted(environment.world.farms)
            agents = {
                farm_ids[0]: FullNeatFarmerAgent(RecurrentNeatFarmerPolicy.load(champion_path)),
                farm_ids[1]: HeuristicFullFarmerAgent(),
                farm_ids[2]: RandomFullFarmerAgent(),
            }
            record = MultiFarmEpisodeRunner(environment, agents).run(seed=episode_seed, max_rounds=6)
            final_world = world_from_dict(record.final_world)
            for farm_id, agent in agents.items():
                rows.append({"episode": episode_index, "farm_id": farm_id, "agent": agent.name, **economy_metrics(final_world, farm_id)})
        aggregate = {}
        for agent_name in sorted({row["agent"] for row in rows}):
            values = [row["objective_utility"] for row in rows if row["agent"] == agent_name]
            aggregate[agent_name] = {"mean_utility": mean(values), "std_utility": pstdev(values) if len(values) > 1 else 0.0, "episodes": len(values)}
        return {"rows": rows, "aggregate": aggregate}
