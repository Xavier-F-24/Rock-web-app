"""Discover complete, locally runnable neural pair-ranker bundles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class PairRankerCheckpointOption:
    label: str
    ranker_path: str
    predictor_path: str | None
    epoch: int | None
    predictor_feature_dimension: int


def _load_metadata(path: Path) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    architecture = checkpoint.get("model_architecture_config", {})
    return {
        "epoch": checkpoint.get("epoch"),
        "predictor_feature_dimension": int(architecture.get("predictor_feature_dimension", 0)),
        "target_dimension": len(checkpoint.get("target_names", ())),
    }


def _friendly_run_name(path: Path) -> str:
    return path.parent.name.replace("_", " ").title()


def discover_pair_ranker_checkpoints(
    repository_root: str | Path,
    *,
    include_latest: bool = False,
    metadata_loader: Callable[[Path], dict[str, Any]] = _load_metadata,
) -> tuple[PairRankerCheckpointOption, ...]:
    root = Path(repository_root).resolve()
    filenames = ("best.pt", "latest.pt") if include_latest else ("best.pt",)
    rankers = [path for filename in filenames for path in root.glob(f"training_runs/pair_ranker*/{filename}")]
    predictors = list(root.glob("training_runs/breeding_predictor*/best.pt"))
    predictor_metadata = []
    for path in predictors:
        try:
            predictor_metadata.append((path, metadata_loader(path)))
        except Exception:
            continue

    options = []
    for path in rankers:
        try:
            metadata = metadata_loader(path)
        except Exception:
            continue
        predictor_dimension = int(metadata.get("predictor_feature_dimension", 0))
        companion = None
        if predictor_dimension:
            companion = next(
                (candidate for candidate, values in predictor_metadata if int(values.get("target_dimension", -1)) == predictor_dimension),
                None,
            )
            if companion is None:
                continue
        relative_ranker = path.relative_to(root).as_posix()
        relative_predictor = companion.relative_to(root).as_posix() if companion else None
        version = "Best" if path.name == "best.pt" else "Latest"
        bundle = "standalone" if companion is None else f"with {_friendly_run_name(companion)}"
        epoch = metadata.get("epoch")
        epoch_text = f", epoch {epoch}" if epoch is not None else ""
        options.append(
            PairRankerCheckpointOption(
                label=f"{_friendly_run_name(path)} - {version} ({bundle}{epoch_text})",
                ranker_path=relative_ranker,
                predictor_path=relative_predictor,
                epoch=int(epoch) if epoch is not None else None,
                predictor_feature_dimension=predictor_dimension,
            )
        )
    options.sort(
        key=lambda option: (
            option.predictor_feature_dimension != 0,
            "smoke" in option.ranker_path.lower(),
            option.label,
        )
    )
    return tuple(options)
