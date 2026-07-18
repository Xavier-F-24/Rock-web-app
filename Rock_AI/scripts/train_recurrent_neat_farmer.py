"""CLI for bounded open-topology recurrent NEAT training."""

from __future__ import annotations

import argparse

from Rock_AI.training.recurrent_neat_training_helper import RecurrentNeatTrainer, RecurrentNeatTrainingConfig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--settling-steps", type=int, default=3)
    parser.add_argument("--training-scenarios", type=int, default=24)
    parser.add_argument("--validation-scenarios", type=int, default=24)
    args = parser.parse_args(argv)
    trainer = RecurrentNeatTrainer(RecurrentNeatTrainingConfig(
        dataset_path=args.dataset, output_directory=args.output, seed=args.seed,
        population=args.population, generations=args.generations,
        settling_steps=args.settling_steps,
        training_scenarios_per_generation=args.training_scenarios,
        validation_scenarios=args.validation_scenarios,
    ))
    winner = trainer.train()
    print(f"Champion genome: {winner.key}; fitness={winner.fitness:.6f}")


if __name__ == "__main__":
    main()
