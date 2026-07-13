"""Generate a reproducible supervised breeding-predictor dataset."""

from __future__ import annotations

import argparse
import json

from Rock_AI.datasets.dataset_split_helper import split_predictor_examples
from Rock_AI.datasets.dataset_storage_helper import save_predictor_dataset
from Rock_AI.datasets.predictor_dataset_generator import PredictorDatasetGenerator
from Rock_AI.training.training_config_helper import TrainingDataConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--trials-per-pair", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", default="training_data/breeding_predictor_v1")
    parser.add_argument("--mutation-min", type=float, default=0.0)
    parser.add_argument("--mutation-max", type=float, default=0.12)
    parser.add_argument("--death-min", type=float, default=0.0)
    parser.add_argument("--death-max", type=float, default=0.15)
    parser.add_argument("--value-thresholds", type=float, nargs="+", default=[5.0, 10.0, 20.0])
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = TrainingDataConfig(
        number_of_parent_pairs=args.pairs,
        trials_per_pair=args.trials_per_pair,
        seed=args.seed,
        mutation_chance_range=(args.mutation_min, args.mutation_max),
        death_chance_range=(args.death_min, args.death_max),
        value_thresholds=tuple(args.value_thresholds),
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        output_directory=args.output,
    )
    generator = PredictorDatasetGenerator(config)
    examples = generator.generate_procedural_examples()
    splits = split_predictor_examples(examples, config)
    files = save_predictor_dataset(splits, generator.target_schema, config)
    manifest = json.loads(files["manifest"].read_text(encoding="utf-8"))
    report = manifest["validation_report"]
    console_report = {
        key: value
        for key, value in report.items()
        if key != "target_ranges" and not isinstance(value, list)
    }
    console_report["target_range_count"] = len(report["target_ranges"])
    print(json.dumps(console_report, indent=2, sort_keys=True))
    print(f"Wrote predictor dataset to {config.output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
