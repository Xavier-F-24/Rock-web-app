"""CLI for reproducible pair-ranking data generation."""

from __future__ import annotations

import argparse
import json

from Rock_AI.datasets.pair_ranking_dataset_generator import PairRankingDatasetGenerator
from Rock_AI.datasets.pair_ranking_storage_helper import save_pair_ranking_dataset
from Rock_AI.training.training_config_helper import PairRankingDataConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--farms", type=int, default=100)
    parser.add_argument("--trials-per-pair", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--predictor-checkpoint")
    parser.add_argument("--output", default="training_data/pair_ranker_v1")
    args = parser.parse_args()
    config = PairRankingDataConfig(
        number_of_farms=args.farms,
        trials_per_pair=args.trials_per_pair,
        seed=args.seed,
        predictor_checkpoint=args.predictor_checkpoint,
        output_directory=args.output,
    )
    generator = PairRankingDatasetGenerator(config)
    groups = generator.generate()
    splits = generator.split_groups(groups)
    if any(not values for values in splits.values()):
        raise ValueError("At least three usable farms are required to populate every split")
    summary = save_pair_ranking_dataset(config.output_path, splits, generator.manifest())
    print(json.dumps(summary["splits"], indent=2))


if __name__ == "__main__":
    main()
