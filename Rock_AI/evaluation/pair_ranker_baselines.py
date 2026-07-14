"""Transparent baselines for pair-ranking datasets."""

from __future__ import annotations

import numpy as np

from Rock_AI.evaluation.pair_ranker_metrics import calculate_pair_ranker_metrics


def evaluate_pair_ranker_baselines(
    arrays: dict[str, np.ndarray],
    seed: int = 0,
    predictor_feature_names: list[str] | tuple[str, ...] = (),
) -> dict[str, dict[str, float]]:
    utilities = arrays["utility_scores"]
    offsets = arrays["group_offsets"]
    group_count = len(offsets) - 1
    maximum = max(int(offsets[i + 1] - offsets[i]) for i in range(group_count))
    true = np.zeros((group_count, maximum), dtype=np.float32)
    mask = np.zeros((group_count, maximum), dtype=np.bool_)
    scores = {
        "random_legal_pair": np.zeros_like(true),
        "highest_combined_parent_value": np.zeros_like(true),
        "pair_evaluator_immediate_value": np.zeros_like(true),
        "manual_predictor_formula": np.zeros_like(true),
    }
    rng = np.random.default_rng(seed)
    for group in range(group_count):
        start, end = int(offsets[group]), int(offsets[group + 1])
        width = end - start
        mask[group, :width] = True
        true[group, :width] = utilities[start:end]
        scores["random_legal_pair"][group, :width] = rng.random(width)
        scores["highest_combined_parent_value"][group, :width] = arrays["metadata_features"][start:end, 0]
        scores["pair_evaluator_immediate_value"][group, :width] = arrays["utility_components"][start:end, 0]
        if arrays["predictor_features"].shape[1]:
            names = {name: index for index, name in enumerate(predictor_feature_names)}
            required = {
                "expected_survivor_count",
                "expected_average_surviving_child_value",
                "expected_maximum_surviving_child_value",
                "expected_mutation_count",
                "genotype_diversity_estimate",
                "phenotype_diversity_estimate",
            }
            if not required <= set(names):
                raise ValueError("Predictor feature names cannot support the manual baseline")
            values = arrays["predictor_features"][start:end]
            average = values[:, names["expected_average_surviving_child_value"]]
            scores["manual_predictor_formula"][group, :width] = (
                average
                + 0.25 * values[:, names["expected_maximum_surviving_child_value"]]
                + 0.5 * values[:, names["expected_survivor_count"]] * average
                + 2.0 * values[:, names["genotype_diversity_estimate"]]
                + 2.0 * values[:, names["phenotype_diversity_estimate"]]
                + 0.5 * values[:, names["expected_mutation_count"]]
            )
        else:
            scores["manual_predictor_formula"][group, :width] = arrays["utility_components"][start:end, 0]
    return {name: calculate_pair_ranker_metrics(values, true, mask) for name, values in scores.items()}
