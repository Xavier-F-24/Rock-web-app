from __future__ import annotations

import numpy as np

from Rock_AI.datasets.dataset_split_helper import find_split_leakage, split_predictor_examples
from Rock_AI.datasets.predictor_example_helper import PredictorExample
from Rock_AI.training.training_config_helper import TrainingDataConfig


def _example(index, pair, lineage):
    return PredictorExample(
        parent_a_features=np.asarray([index], dtype=np.float32),
        parent_b_features=np.asarray([index + 1], dtype=np.float32),
        rule_features=np.asarray([0.0], dtype=np.float32),
        context_features=np.asarray([], dtype=np.float32),
        schema_version=1,
        expected_raw_clutch_size=1.0,
        expected_survivor_count=1.0,
        expected_average_surviving_child_value=1.0,
        expected_maximum_surviving_child_value=1.0,
        surviving_value_threshold_probabilities={},
        expected_mutation_count=0.0,
        probability_at_least_one_mutation=0.0,
        genotype_diversity_estimate=1.0,
        phenotype_diversity_estimate=1.0,
        per_gene_child_allele_pair_distributions={},
        phenotype_probability_vector={},
        metadata={
            "parent_pair_key": pair,
            "lineage_group_id": lineage,
            "example_id": str(index),
        },
    )


def test_split_keeps_duplicate_pairs_and_lineages_together():
    examples = [
        _example(0, "1|2", "lineage-a"),
        _example(1, "1|2", "lineage-a"),
        _example(2, "3|4", "lineage-a"),
        _example(3, "5|6", "lineage-b"),
        _example(4, "7|8", "lineage-c"),
        _example(5, "9|10", "lineage-d"),
    ]
    config = TrainingDataConfig(number_of_parent_pairs=6, trials_per_pair=1, seed=12)
    splits = split_predictor_examples(examples, config)

    assert sum(len(values) for values in splits.as_dict().values()) == len(examples)
    assert find_split_leakage(splits) == {
        "parent_pair_leakage": [],
        "lineage_leakage": [],
    }


def test_split_is_deterministic():
    examples = [_example(index, f"{index}|{index + 10}", f"lineage-{index}") for index in range(10)]
    config = TrainingDataConfig(number_of_parent_pairs=10, trials_per_pair=1, seed=88)

    first = split_predictor_examples(examples, config)
    second = split_predictor_examples(examples, config)

    for name in first.as_dict():
        assert [item.metadata["example_id"] for item in first.as_dict()[name]] == [
            item.metadata["example_id"] for item in second.as_dict()[name]
        ]
