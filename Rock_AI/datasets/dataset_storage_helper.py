"""NPZ numerical storage, JSONL metadata, loading, and validation reports."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from Rock_AI.datasets.dataset_split_helper import PredictorDatasetSplits, find_split_leakage
from Rock_AI.datasets.predictor_example_helper import (
    PredictorExample,
    PredictorTargetSchema,
)
from Rock_AI.representations.encoding_schema_helper import EncodingSchema, get_default_encoding_schema
from Rock_AI.training.training_config_helper import TrainingDataConfig


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def examples_to_arrays(
    examples: Iterable[PredictorExample],
    target_schema: PredictorTargetSchema,
    *,
    parent_feature_dimension: int,
    rule_feature_dimension: int,
    context_feature_dimension: int,
) -> dict[str, np.ndarray]:
    rows = list(examples)

    def stack_or_empty(attribute: str, width: int) -> np.ndarray:
        if not rows:
            return np.zeros((0, width), dtype=np.float32)
        return np.stack([getattr(example, attribute) for example in rows]).astype(np.float32)

    targets = (
        np.stack([example.target_vector(target_schema) for example in rows]).astype(np.float32)
        if rows
        else np.zeros((0, len(target_schema.target_names)), dtype=np.float32)
    )
    return {
        "parent_a_features": stack_or_empty("parent_a_features", parent_feature_dimension),
        "parent_b_features": stack_or_empty("parent_b_features", parent_feature_dimension),
        "rule_features": stack_or_empty("rule_features", rule_feature_dimension),
        "context_features": stack_or_empty("context_features", context_feature_dimension),
        "context_mask": np.ones((len(rows), context_feature_dimension), dtype=np.bool_),
        "targets": targets,
        "target_mask": np.ones_like(targets, dtype=np.bool_),
        "schema_versions": np.asarray([example.schema_version for example in rows], dtype=np.int32),
    }


def validate_predictor_dataset(
    splits: PredictorDatasetSplits,
    target_schema: PredictorTargetSchema,
) -> dict[str, Any]:
    all_examples = [example for values in splits.as_dict().values() for example in values]
    arrays = (
        [example.parent_a_features for example in all_examples]
        + [example.parent_b_features for example in all_examples]
        + [example.rule_features for example in all_examples]
        + [example.context_features for example in all_examples]
    )
    targets = [example.target_vector(target_schema) for example in all_examples]
    finite_inputs = all(np.isfinite(array).all() for array in arrays)
    finite_targets = all(np.isfinite(array).all() for array in targets)
    target_matrix = np.stack(targets) if targets else np.zeros((0, len(target_schema.target_names)))
    target_ranges = {
        name: {
            "minimum": float(target_matrix[:, index].min()),
            "maximum": float(target_matrix[:, index].max()),
        }
        for index, name in enumerate(target_schema.target_names)
    } if len(target_matrix) else {}
    uncertainty_values = [
        float(value)
        for example in all_examples
        for value in example.metadata.get("uncertainty_estimates", {}).values()
    ]
    mutation_settings = [
        float(example.metadata["rule_encoding"]["mutation_chance"])
        for example in all_examples
    ]
    leakage = find_split_leakage(splits)
    represented_phenotypes = [
        name
        for name, bounds in target_ranges.items()
        if name.startswith("phenotype.") and bounds["maximum"] > 0.0
    ]
    return {
        "number_of_examples": len(all_examples),
        "has_nan_or_infinite_inputs": not finite_inputs,
        "has_nan_or_infinite_targets": not finite_targets,
        "parent_feature_dimension": len(all_examples[0].parent_a_features) if all_examples else 0,
        "rule_feature_dimension": len(all_examples[0].rule_features) if all_examples else 0,
        "context_feature_dimension": len(all_examples[0].context_features) if all_examples else 0,
        "target_dimension": len(target_schema.target_names),
        "split_sizes": {
            name: len(examples) for name, examples in splits.as_dict().items()
        },
        "target_ranges": target_ranges,
        "average_uncertainty": float(np.mean(uncertainty_values)) if uncertainty_values else 0.0,
        "mutation_setting_distribution": {
            "minimum": min(mutation_settings) if mutation_settings else 0.0,
            "maximum": max(mutation_settings) if mutation_settings else 0.0,
            "mean": float(np.mean(mutation_settings)) if mutation_settings else 0.0,
        },
        "gene_distribution_target_count": len(target_schema.allele_distribution_target_names),
        "phenotype_target_count": len(target_schema.phenotype_target_names),
        "represented_gene_count": len(
            {name.split(".", maxsplit=2)[1] for name in target_schema.allele_distribution_target_names}
        ),
        "represented_gene_names": sorted(
            {name.split(".", maxsplit=2)[1] for name in target_schema.allele_distribution_target_names}
        ),
        "represented_phenotype_target_count": len(represented_phenotypes),
        "represented_phenotype_targets": represented_phenotypes,
        **leakage,
    }


def save_predictor_dataset(
    splits: PredictorDatasetSplits,
    target_schema: PredictorTargetSchema,
    config: TrainingDataConfig,
    *,
    schema: EncodingSchema | None = None,
) -> dict[str, Path]:
    schema = schema or get_default_encoding_schema()
    output = config.output_path
    output.mkdir(parents=True, exist_ok=True)
    all_examples = [example for values in splits.as_dict().values() for example in values]
    if not all_examples:
        raise ValueError("Cannot save an empty predictor dataset")
    first = all_examples[0]
    context_feature_names = (
        (
            "parent_a_generation_normalized",
            "parent_b_generation_normalized",
            "generation_difference_normalized",
            "parent_a_value_normalized",
            "parent_b_value_normalized",
        )
        if len(first.context_features)
        else ()
    )
    files: dict[str, Path] = {}
    for split_name, examples in splits.as_dict().items():
        arrays = examples_to_arrays(
            examples,
            target_schema,
            parent_feature_dimension=len(first.parent_a_features),
            rule_feature_dimension=len(first.rule_features),
            context_feature_dimension=len(first.context_features),
        )
        npz_path = output / f"{split_name}.npz"
        np.savez_compressed(
            npz_path,
            **arrays,
            parent_feature_names=np.asarray(schema.rock_matrix_feature_names),
            rule_feature_names=np.asarray(tuple(first.metadata["rule_encoding"])),
            context_feature_names=np.asarray(context_feature_names, dtype=str),
            target_names=np.asarray(target_schema.target_names),
        )
        metadata_path = output / f"{split_name}_metadata.jsonl"
        with metadata_path.open("w", encoding="utf-8", newline="\n") as handle:
            for example in examples:
                handle.write(json.dumps(_json_safe(example.metadata), sort_keys=True))
                handle.write("\n")
        files[f"{split_name}_npz"] = npz_path
        files[f"{split_name}_metadata"] = metadata_path

    report = validate_predictor_dataset(splits, target_schema)
    manifest = {
        "dataset_version": 1,
        "encoding_schema_version": schema.version,
        "game_rules_version": config.game_rules_version,
        "config": config.to_dict(),
        "encoding_schema": {
            "version": schema.version,
            "gene_names": list(schema.gene_names),
            "death_gene_names": list(schema.death_gene_names),
            "sex_values": list(schema.sex_values),
            "status_values": list(schema.status_values),
            "phenotype_values": {
                gene_name: list(values)
                for gene_name, values in schema.phenotype_values.items()
            },
        },
        "parent_feature_names": list(schema.rock_matrix_feature_names),
        "rule_feature_names": list(first.metadata["rule_encoding"]),
        "context_feature_names": list(context_feature_names),
        "target_names": list(target_schema.target_names),
        "validation_report": report,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    files["manifest"] = manifest_path
    return files


def load_predictor_split(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as loaded:
        return {name: loaded[name].copy() for name in loaded.files}


def load_metadata_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
