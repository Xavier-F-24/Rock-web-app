"""Group-aware pair-ranking metrics, with utility regret as the primary measure."""

from __future__ import annotations

import math

import numpy as np


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def calculate_pair_ranker_metrics(
    predicted_scores: np.ndarray,
    true_utilities: np.ndarray,
    candidate_mask: np.ndarray,
) -> dict[str, float]:
    rows = {name: [] for name in ("top1", "top3", "mrr", "ndcg", "spearman", "kendall", "regret", "normalized_regret")}
    for scores, utilities, mask in zip(predicted_scores, true_utilities, candidate_mask):
        scores = scores[mask]
        utilities = utilities[mask]
        if not len(scores):
            continue
        predicted = np.argsort(-scores, kind="stable")
        true_order = np.argsort(-utilities, kind="stable")
        best = true_order[0]
        position = int(np.where(predicted == best)[0][0])
        rows["top1"].append(float(position == 0))
        rows["top3"].append(float(position < min(3, len(scores))))
        rows["mrr"].append(1.0 / (position + 1))
        gains = utilities - utilities.min()
        discounts = 1.0 / np.log2(np.arange(2, len(scores) + 2))
        dcg = float(np.sum(gains[predicted] * discounts))
        ideal = float(np.sum(gains[true_order] * discounts))
        rows["ndcg"].append(dcg / ideal if ideal > 1e-12 else 1.0)
        rows["spearman"].append(_correlation(_rankdata(scores), _rankdata(utilities)))
        concordant = discordant = 0
        for left in range(len(scores)):
            for right in range(left + 1, len(scores)):
                product = (scores[left] - scores[right]) * (utilities[left] - utilities[right])
                concordant += product > 0
                discordant += product < 0
        denominator = concordant + discordant
        rows["kendall"].append((concordant - discordant) / denominator if denominator else 0.0)
        regret = float(utilities[best] - utilities[predicted[0]])
        spread = float(utilities.max() - utilities.min())
        rows["regret"].append(regret)
        rows["normalized_regret"].append(regret / spread if spread > 1e-12 else 0.0)
    return {
        "top_1_accuracy": float(np.mean(rows["top1"])) if rows["top1"] else 0.0,
        "top_3_recall": float(np.mean(rows["top3"])) if rows["top3"] else 0.0,
        "mean_reciprocal_rank": float(np.mean(rows["mrr"])) if rows["mrr"] else 0.0,
        "ndcg": float(np.mean(rows["ndcg"])) if rows["ndcg"] else 0.0,
        "spearman_correlation": float(np.mean(rows["spearman"])) if rows["spearman"] else 0.0,
        "kendall_correlation": float(np.mean(rows["kendall"])) if rows["kendall"] else 0.0,
        "mean_utility_regret": float(np.mean(rows["regret"])) if rows["regret"] else 0.0,
        "mean_normalized_regret": float(np.mean(rows["normalized_regret"])) if rows["normalized_regret"] else 0.0,
    }
