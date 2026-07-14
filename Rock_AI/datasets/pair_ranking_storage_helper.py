"""NPZ plus JSON storage for variable-length farm ranking groups."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from Rock_AI.datasets.pair_ranking_record_helper import PairRankingGroup


DATASET_VERSION = 1


def _stack(groups: Iterable[PairRankingGroup]) -> tuple[dict[str, np.ndarray], list[dict]]:
    groups = list(groups)
    candidates = [candidate for group in groups for candidate in group.candidates]
    if not candidates:
        raise ValueError("Cannot serialize an empty ranking split")
    offsets = [0]
    metadata: list[dict] = []
    for group in groups:
        offsets.append(offsets[-1] + len(group.candidates))
        metadata.append(
            {
                "group_id": group.group_id,
                "lineage_group_id": group.lineage_group_id,
                "parent_ids": [list(candidate.parent_ids) for candidate in group.candidates],
                "rock_ids": list(group.rock_ids),
                "evaluation_seed": group.evaluation_seed,
                "monte_carlo_trial_count": group.monte_carlo_trial_count,
                "objective_profile": group.objective_profile.to_dict(),
                "breeding_rules": group.breeding_rules,
                **group.metadata,
            }
        )
    arrays = {
        "parent_a_features": np.stack([c.parent_a_features for c in candidates]).astype(np.float32),
        "parent_b_features": np.stack([c.parent_b_features for c in candidates]).astype(np.float32),
        "rule_features": np.stack([c.rule_features for c in candidates]).astype(np.float32),
        "farm_features": np.stack([c.farm_features for c in candidates]).astype(np.float32),
        "objective_features": np.stack([c.objective_features for c in candidates]).astype(np.float32),
        "metadata_features": np.stack([c.metadata_features for c in candidates]).astype(np.float32),
        "predictor_features": np.stack([c.predictor_features for c in candidates]).astype(np.float32),
        "utility_components": np.stack([c.utility_components for c in candidates]).astype(np.float32),
        "utility_scores": np.asarray([c.utility_score for c in candidates], dtype=np.float32),
        "uncertainties": np.asarray([c.uncertainty for c in candidates], dtype=np.float32),
        "ranks": np.asarray([c.rank for c in candidates], dtype=np.int64),
        "best_pair": np.asarray([c.best_pair for c in candidates], dtype=np.bool_),
        "candidate_mask": np.ones(len(candidates), dtype=np.bool_),
        "group_offsets": np.asarray(offsets, dtype=np.int64),
    }
    return arrays, metadata


def save_pair_ranking_dataset(
    output_directory: str | Path,
    splits: dict[str, list[PairRankingGroup]],
    manifest: dict,
) -> dict:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    summary = dict(manifest)
    summary["dataset_version"] = DATASET_VERSION
    summary["splits"] = {}
    for split, groups in splits.items():
        arrays, metadata = _stack(groups)
        np.savez_compressed(output / f"{split}.npz", **arrays)
        (output / f"{split}_groups.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        summary["splits"][split] = {
            "groups": len(groups),
            "candidates": int(arrays["utility_scores"].shape[0]),
        }
    (output / "manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def load_pair_ranking_split(directory: str | Path, split: str) -> tuple[dict[str, np.ndarray], list[dict], dict]:
    root = Path(directory)
    with np.load(root / f"{split}.npz", allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    metadata = json.loads((root / f"{split}_groups.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    offsets = arrays["group_offsets"]
    if len(offsets) != len(metadata) + 1 or offsets[-1] != len(arrays["utility_scores"]):
        raise ValueError("Ranking dataset group boundaries are inconsistent")
    return arrays, metadata, manifest
