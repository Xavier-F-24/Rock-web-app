"""CLI for bounded full-farmer recurrent NEAT evolution."""

import argparse

from Rock_AI.training.action_curriculum import ActionCurriculumStage
from Rock_AI.training.full_farmer_neat_trainer import FullFarmerNeatTrainer
from Rock_AI.training.full_farmer_training_config import FullFarmerTrainingConfig


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--worlds-per-genome", type=int, default=3)
    parser.add_argument("--rounds-per-world", type=int, default=6)
    parser.add_argument("--curriculum-start", choices=[stage.name.lower() for stage in ActionCurriculumStage], default="imports")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--single-process", action="store_true")
    args = parser.parse_args(argv)
    config = FullFarmerTrainingConfig(
        args.output, args.seed, args.population, args.generations, args.worlds_per_genome,
        args.rounds_per_world, ActionCurriculumStage[args.curriculum_start.upper()],
        args.checkpoint_every, single_process=True,
    )
    _, metadata = FullFarmerNeatTrainer(config).train()
    print(f"Champion fitness: {metadata['fitness']:.6f}")
    print(f"Champion: {args.output}/champions/best_validation/network.json")


if __name__ == "__main__":
    main()
