"""Evaluate a predictor checkpoint and compare mean and shallow baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from Rock_AI.evaluation.predictor_baselines import (
    ShallowLinearBaseline,
    TrainingTargetMeanBaseline,
    fit_shallow_linear_baseline,
)
from Rock_AI.evaluation.predictor_evaluator import BreedingPredictor, PredictorEvaluator
from Rock_AI.evaluation.predictor_metrics import calculate_predictor_metrics
from Rock_AI.models.loss_helper import PredictorLossConfig
from Rock_AI.models.model_output_helper import TargetLayout
from Rock_AI.training.predictor_data_helper import (
    NpzPredictorDataset,
    TargetNormalizer,
    make_data_loader,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--baseline-epochs", type=int, default=50)
    parser.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch.manual_seed(1234)
    predictor = BreedingPredictor.load(args.checkpoint, device=args.device)
    train_dataset = NpzPredictorDataset(args.dataset, "train")
    selected_dataset = NpzPredictorDataset(args.dataset, args.split)
    layout = TargetLayout.from_target_names(selected_dataset.target_names)
    normalizer = predictor.normalizer
    selected_loader = make_data_loader(
        selected_dataset, args.batch_size, shuffle=False, seed=1234
    )
    evaluator = PredictorEvaluator(
        predictor.model,
        layout,
        normalizer,
        PredictorLossConfig(**predictor.checkpoint["loss_configuration"]),
        predictor.device,
    )
    model_metrics, model_losses = evaluator.evaluate_loader(selected_loader)

    mean_baseline = TrainingTargetMeanBaseline.fit(
        train_dataset.arrays["targets"], train_dataset.arrays["target_mask"]
    )
    mean_predictions = mean_baseline.predict(len(selected_dataset))
    mean_metrics = calculate_predictor_metrics(
        mean_predictions,
        selected_dataset.arrays["targets"],
        selected_dataset.arrays["target_mask"],
        layout,
        normalizer,
    )

    shallow = ShallowLinearBaseline(
        train_dataset.arrays["parent_a_features"].shape[1],
        train_dataset.arrays["rule_features"].shape[1],
        train_dataset.arrays["context_features"].shape[1],
        layout,
    )
    train_loader = make_data_loader(train_dataset, args.batch_size, shuffle=True, seed=1234)
    fit_shallow_linear_baseline(
        shallow,
        train_loader,
        normalizer,
        layout,
        epochs=args.baseline_epochs,
        device=predictor.device,
    )
    shallow_evaluator = PredictorEvaluator(
        shallow,
        layout,
        normalizer,
        PredictorLossConfig(),
        predictor.device,
    )
    shallow_metrics, shallow_losses = shallow_evaluator.evaluate_loader(selected_loader)
    report = {
        "split": args.split,
        "model": {"metrics": model_metrics, "losses": model_losses},
        "training_mean_baseline": {"metrics": mean_metrics},
        "shallow_linear_baseline": {"metrics": shallow_metrics, "losses": shallow_losses},
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
