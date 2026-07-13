"""Train the first symmetric neural breeding predictor."""

from __future__ import annotations

import argparse

from Rock_AI.training.train_breeding_predictor import train_breeding_predictor
from Rock_AI.training.training_config_helper import PredictorTrainingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--parent-embedding", type=int, default=64)
    parser.add_argument("--rule-embedding", type=int, default=24)
    parser.add_argument("--context-embedding", type=int, default=16)
    parser.add_argument("--encoder-hidden", type=int, nargs="+", default=[128])
    parser.add_argument("--trunk-hidden", type=int, nargs="+", default=[128, 96])
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--resume")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = PredictorTrainingConfig(
        dataset_path=args.dataset,
        output_directory=args.output,
        seed=args.seed,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        number_of_epochs=args.epochs,
        encoder_hidden_dimensions=tuple(args.encoder_hidden),
        trunk_hidden_dimensions=tuple(args.trunk_hidden),
        parent_embedding_dimension=args.parent_embedding,
        rule_embedding_dimension=args.rule_embedding,
        context_embedding_dimension=args.context_embedding,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.patience,
        device=args.device,
        number_of_workers=args.workers,
        gradient_clip_norm=args.gradient_clip,
        resume_checkpoint=args.resume,
    )
    result = train_breeding_predictor(config)
    print(f"Best validation loss: {result['best_validation_loss']:.6f}")
    print(f"Best checkpoint: {result['best_checkpoint'].resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
