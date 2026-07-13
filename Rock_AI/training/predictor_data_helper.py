"""Validated NPZ datasets, deterministic loaders, and scalar normalization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from Rock_AI.models.model_output_helper import TargetLayout


REQUIRED_ARRAYS = (
    "parent_a_features",
    "parent_b_features",
    "rule_features",
    "context_features",
    "context_mask",
    "targets",
    "target_mask",
    "schema_versions",
)


class NpzPredictorDataset(Dataset):
    """Load one split once, then serve named tensors without per-batch NPZ reads."""

    def __init__(self, dataset_directory: str | Path, split: str):
        self.dataset_directory = Path(dataset_directory)
        self.split = split
        self.manifest = json.loads(
            (self.dataset_directory / "manifest.json").read_text(encoding="utf-8")
        )
        path = self.dataset_directory / f"{split}.npz"
        with np.load(path, allow_pickle=False) as loaded:
            missing = set(REQUIRED_ARRAYS) - set(loaded.files)
            if missing:
                raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
            self.arrays = {name: loaded[name].copy() for name in REQUIRED_ARRAYS}
            self.parent_feature_names = tuple(str(value) for value in loaded["parent_feature_names"])
            self.rule_feature_names = tuple(str(value) for value in loaded["rule_feature_names"])
            self.context_feature_names = tuple(str(value) for value in loaded["context_feature_names"])
            self.target_names = tuple(str(value) for value in loaded["target_names"])
        self._validate()

    def _validate(self) -> None:
        count = self.arrays["targets"].shape[0]
        if any(array.shape[0] != count for array in self.arrays.values()):
            raise ValueError(f"Split {self.split!r} has inconsistent row counts")
        expected = self.manifest
        comparisons = (
            (self.parent_feature_names, tuple(expected["parent_feature_names"]), "parent features"),
            (self.rule_feature_names, tuple(expected["rule_feature_names"]), "rule features"),
            (self.context_feature_names, tuple(expected["context_feature_names"]), "context features"),
            (self.target_names, tuple(expected["target_names"]), "targets"),
        )
        for actual, declared, label in comparisons:
            if actual != declared:
                raise ValueError(f"Split {self.split!r} {label} do not match manifest order")
        if self.arrays["targets"].shape[1] != len(self.target_names):
            raise ValueError("Target matrix width does not match target names")
        if not all(np.isfinite(self.arrays[name]).all() for name in (
            "parent_a_features", "parent_b_features", "rule_features", "context_features", "targets"
        )):
            raise ValueError(f"Split {self.split!r} contains NaN or infinite values")

    def __len__(self) -> int:
        return self.arrays["targets"].shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "parent_a_features": torch.from_numpy(self.arrays["parent_a_features"][index]),
            "parent_b_features": torch.from_numpy(self.arrays["parent_b_features"][index]),
            "rule_features": torch.from_numpy(self.arrays["rule_features"][index]),
            "context_features": torch.from_numpy(self.arrays["context_features"][index]),
            "context_mask": torch.from_numpy(self.arrays["context_mask"][index]),
            "targets": torch.from_numpy(self.arrays["targets"][index]),
            "target_mask": torch.from_numpy(self.arrays["target_mask"][index]),
            "schema_version": torch.tensor(int(self.arrays["schema_versions"][index])),
        }


@dataclass(frozen=True)
class TargetNormalizer:
    scalar_indices: tuple[int, ...]
    means: tuple[float, ...]
    standard_deviations: tuple[float, ...]
    epsilon: float = 1e-6

    @classmethod
    def fit(cls, targets: np.ndarray, target_mask: np.ndarray, layout: TargetLayout) -> "TargetNormalizer":
        means: list[float] = []
        deviations: list[float] = []
        for index in layout.scalar_indices:
            values = targets[:, index][target_mask[:, index].astype(bool)]
            if not len(values):
                means.append(0.0)
                deviations.append(1.0)
                continue
            mean = float(np.mean(values))
            deviation = float(np.std(values))
            means.append(mean)
            deviations.append(deviation if deviation >= 1e-6 else 1.0)
        return cls(tuple(layout.scalar_indices), tuple(means), tuple(deviations))

    def normalize_scalar_tensor(self, targets: torch.Tensor) -> torch.Tensor:
        means = targets.new_tensor(self.means)
        deviations = targets.new_tensor(self.standard_deviations)
        return (targets[:, list(self.scalar_indices)] - means) / deviations

    def denormalize_scalar_tensor(self, normalized: torch.Tensor) -> torch.Tensor:
        means = normalized.new_tensor(self.means)
        deviations = normalized.new_tensor(self.standard_deviations)
        return normalized * deviations + means

    def normalize_array(self, values: np.ndarray) -> np.ndarray:
        return (values - np.asarray(self.means)) / np.asarray(self.standard_deviations)

    def denormalize_array(self, values: np.ndarray) -> np.ndarray:
        return values * np.asarray(self.standard_deviations) + np.asarray(self.means)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TargetNormalizer":
        return cls(
            scalar_indices=tuple(values["scalar_indices"]),
            means=tuple(values["means"]),
            standard_deviations=tuple(values["standard_deviations"]),
            epsilon=float(values.get("epsilon", 1e-6)),
        )


def make_data_loader(
    dataset: NpzPredictorDataset,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
    number_of_workers: int = 0,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=number_of_workers,
        generator=generator,
    )


def context_swap_pairs(context_feature_names: tuple[str, ...]) -> tuple[tuple[int, int], ...]:
    names = {name: index for index, name in enumerate(context_feature_names)}
    pairs: list[tuple[int, int]] = []
    for name, index in names.items():
        if not name.startswith("parent_a_"):
            continue
        counterpart = "parent_b_" + name.removeprefix("parent_a_")
        if counterpart in names:
            pairs.append((index, names[counterpart]))
    return tuple(sorted(pairs))
