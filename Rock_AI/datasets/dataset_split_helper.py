"""Leakage-resistant parent-pair and lineage-aware dataset splitting."""

from __future__ import annotations

import random
from dataclasses import dataclass

from Rock_AI.datasets.predictor_example_helper import PredictorExample
from Rock_AI.training.training_config_helper import TrainingDataConfig


@dataclass(frozen=True)
class PredictorDatasetSplits:
    train: tuple[PredictorExample, ...]
    validation: tuple[PredictorExample, ...]
    test: tuple[PredictorExample, ...]

    def as_dict(self) -> dict[str, tuple[PredictorExample, ...]]:
        return {"train": self.train, "validation": self.validation, "test": self.test}


class _DisjointSet:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def split_predictor_examples(
    examples: list[PredictorExample],
    config: TrainingDataConfig,
) -> PredictorDatasetSplits:
    """Keep any shared exact pair or lineage group entirely in one split."""

    if not examples:
        return PredictorDatasetSplits((), (), ())
    disjoint = _DisjointSet(len(examples))
    token_owner: dict[str, int] = {}
    for index, example in enumerate(examples):
        metadata = example.metadata
        tokens = (
            f"pair:{metadata['parent_pair_key']}",
            f"lineage:{metadata.get('lineage_group_id', metadata['parent_pair_key'])}",
        )
        for token in tokens:
            if token in token_owner:
                disjoint.union(index, token_owner[token])
            else:
                token_owner[token] = index

    grouped: dict[int, list[PredictorExample]] = {}
    for index, example in enumerate(examples):
        grouped.setdefault(disjoint.find(index), []).append(example)
    groups = list(grouped.values())
    random.Random(config.seed).shuffle(groups)

    names = ("train", "validation", "test")
    fractions = {
        "train": config.train_fraction,
        "validation": config.validation_fraction,
        "test": config.test_fraction,
    }
    targets = {name: fractions[name] * len(examples) for name in names}
    assigned: dict[str, list[PredictorExample]] = {name: [] for name in names}
    eligible = [name for name in names if fractions[name] > 0.0]
    if len(groups) >= len(eligible):
        for name in sorted(eligible, key=lambda item: (targets[item], item)):
            smallest_index = min(range(len(groups)), key=lambda index: len(groups[index]))
            assigned[name].extend(groups.pop(smallest_index))
    for group in groups:
        destination = max(
            eligible,
            key=lambda name: (targets[name] - len(assigned[name]), fractions[name], name),
        )
        assigned[destination].extend(group)

    return PredictorDatasetSplits(
        train=tuple(assigned["train"]),
        validation=tuple(assigned["validation"]),
        test=tuple(assigned["test"]),
    )


def find_split_leakage(splits: PredictorDatasetSplits) -> dict[str, list[str]]:
    pair_splits: dict[str, set[str]] = {}
    lineage_splits: dict[str, set[str]] = {}
    for split_name, examples in splits.as_dict().items():
        for example in examples:
            pair = str(example.metadata["parent_pair_key"])
            lineage = str(example.metadata.get("lineage_group_id", pair))
            pair_splits.setdefault(pair, set()).add(split_name)
            lineage_splits.setdefault(lineage, set()).add(split_name)
    return {
        "parent_pair_leakage": sorted(
            key for key, locations in pair_splits.items() if len(locations) > 1
        ),
        "lineage_leakage": sorted(
            key for key, locations in lineage_splits.items() if len(locations) > 1
        ),
    }
