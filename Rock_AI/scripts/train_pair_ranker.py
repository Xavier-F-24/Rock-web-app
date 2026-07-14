"""CLI for pair-ranker training."""

from __future__ import annotations

import argparse

from Rock_AI.training.train_pair_ranker import train_pair_ranker
from Rock_AI.training.training_config_helper import PairRankerTrainingConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume")
    parser.add_argument("--transfer-predictor")
    parser.add_argument("--freeze-transferred-encoder", action="store_true")
    args = parser.parse_args()
    train_pair_ranker(PairRankerTrainingConfig(
        dataset_path=args.dataset,
        output_directory=args.output,
        number_of_epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        resume_checkpoint=args.resume,
        transferred_predictor_checkpoint=args.transfer_predictor,
        freeze_transferred_parent_encoder=args.freeze_transferred_encoder,
    ))


if __name__ == "__main__":
    main()
