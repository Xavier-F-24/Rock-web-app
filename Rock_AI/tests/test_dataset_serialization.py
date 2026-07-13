from __future__ import annotations

import json

import numpy as np

from Rock_AI.datasets.dataset_split_helper import split_predictor_examples
from Rock_AI.datasets.dataset_storage_helper import (
    load_metadata_jsonl,
    load_predictor_split,
    save_predictor_dataset,
)
from Rock_AI.datasets.predictor_dataset_generator import PredictorDatasetGenerator
from Rock_AI.training.training_config_helper import TrainingDataConfig


def test_npz_jsonl_manifest_round_trip(tmp_path):
    config = TrainingDataConfig(
        number_of_parent_pairs=6,
        trials_per_pair=1,
        seed=7000,
        output_directory=str(tmp_path / "predictor"),
        value_thresholds=(5.0,),
    )
    generator = PredictorDatasetGenerator(config)
    examples = generator.generate_procedural_examples()
    splits = split_predictor_examples(examples, config)
    files = save_predictor_dataset(splits, generator.target_schema, config)

    train = load_predictor_split(files["train_npz"])
    metadata = load_metadata_jsonl(files["train_metadata"])
    manifest = json.loads(files["manifest"].read_text(encoding="utf-8"))

    assert train["parent_a_features"].shape[0] == len(splits.train)
    assert train["targets"].shape[1] == len(generator.target_schema.target_names)
    assert np.isfinite(train["targets"]).all()
    assert len(metadata) == len(splits.train)
    report = manifest["validation_report"]
    assert report["number_of_examples"] == 6
    assert report["has_nan_or_infinite_inputs"] is False
    assert report["has_nan_or_infinite_targets"] is False
    assert report["parent_pair_leakage"] == []
    assert report["lineage_leakage"] == []
