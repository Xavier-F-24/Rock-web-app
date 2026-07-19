"""Run one visible deterministic three-farm economy episode."""

import argparse
import json
from pathlib import Path

from Rock_AI.agents.full_neat_farmer_agent import FullNeatFarmerAgent
from Rock_AI.agents.heuristic_full_farmer_agent import HeuristicFullFarmerAgent
from Rock_AI.agents.random_full_farmer_agent import RandomFullFarmerAgent
from Rock_AI.environments.multi_farm_economy_environment import MultiFarmEconomyEnvironment
from Rock_AI.environments.world_episode_runner import MultiFarmEpisodeRunner
from Rock_AI.policies.recurrent_neat_farmer_policy import RecurrentNeatFarmerPolicy


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--champion", required=True)
    parser.add_argument("--seed", type=int, default=2468)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    environment = MultiFarmEconomyEnvironment(args.seed)
    environment.reset()
    farm_ids = sorted(environment.world.farms)
    agents = {
        farm_ids[0]: FullNeatFarmerAgent(RecurrentNeatFarmerPolicy.load(args.champion)),
        farm_ids[1]: HeuristicFullFarmerAgent(),
        farm_ids[2]: RandomFullFarmerAgent(),
    }
    record = MultiFarmEpisodeRunner(environment, agents).run(seed=args.seed, max_rounds=args.rounds)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    print(f"Episode: {output}")
    print(f"Rounds: {len(record.rounds)} | Decisions: {len(record.decisions)} | Termination: {record.termination_reason}")


if __name__ == "__main__":
    main()
