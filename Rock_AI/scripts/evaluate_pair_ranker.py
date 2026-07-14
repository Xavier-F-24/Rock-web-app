"""CLI for ranker metrics and baseline comparison."""

from __future__ import annotations

import argparse
import json

from Rock_AI.datasets.pair_ranking_storage_helper import load_pair_ranking_split
from Rock_AI.evaluation.pair_ranker_baselines import evaluate_pair_ranker_baselines
from Rock_AI.training.train_pair_ranker import evaluate_pair_ranker_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    arrays, _, manifest = load_pair_ranking_split(args.dataset, args.split)
    result = evaluate_pair_ranker_checkpoint(args.dataset, args.checkpoint, args.split, args.device)
    result["baselines"] = evaluate_pair_ranker_baselines(
        arrays,
        predictor_feature_names=manifest["feature_names"]["predictor"],
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
