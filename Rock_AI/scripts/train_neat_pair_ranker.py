"""CLI for deterministic player-visible NEAT pair-ranker training."""

from __future__ import annotations

import argparse

from Rock_AI.training.neat_training_helper import NeatPairRankerTrainer, NeatTrainingConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--training-scenarios", type=int, default=12)
    parser.add_argument("--validation-scenarios", type=int, default=12)
    parser.add_argument("--complexity-penalty", type=float, default=0.00001)
    args = parser.parse_args()
    NeatPairRankerTrainer(NeatTrainingConfig(
        dataset_path=args.dataset,
        output_directory=args.output,
        population=args.population,
        generations=args.generations,
        seed=args.seed,
        training_scenarios_per_generation=args.training_scenarios,
        validation_scenarios=args.validation_scenarios,
        complexity_penalty=args.complexity_penalty,
    )).train()


if __name__ == "__main__":
    main()
