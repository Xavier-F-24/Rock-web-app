from __future__ import annotations

import pytest

from Rock_AI.agents.breeding_agent_helper import BreedPairAction
from Rock_AI.datasets.pair_ranking_record_helper import FarmerObjectiveProfile
from Rock_AI.environments.breeding_campaign_environment import BreedingCampaignEnvironment
from Rock_AI.explanations.decision_explanation_helper import build_decision_explanation


def _prediction(value):
    return {
        "scalar_predictions": {
            "expected_survivor_count": 2.0,
            "expected_average_surviving_child_value": value,
            "expected_maximum_surviving_child_value": value + 3,
            "genotype_diversity_estimate": 0.8,
            "phenotype_diversity_estimate": 0.75,
        },
        "binary_probability_predictions": {
            "probability_at_least_one_mutation": 0.3,
        },
    }


def test_explanation_retains_top_five_and_matches_explicit_policy_scores():
    environment = BreedingCampaignEnvironment(seed=150)
    observation = environment.reset(150)
    selected = observation.legal_pair_ids[0]
    rows = [
        {
            "parent_ids": list(selected if index == 0 else observation.legal_pair_ids[index % len(observation.legal_pair_ids)]),
            "score": 10.0 - index * 0.01,
            "predicted_breeding_outcomes": _prediction(5.0 + index),
            "score_components": {"rare_trait": 0.4, "uncertainty_penalty": 0.1},
        }
        for index in range(6)
    ]
    explanation = build_decision_explanation(
        BreedPairAction(*selected),
        farm=environment.game,
        legal_pair_ids=observation.legal_pair_ids,
        objective_profile=FarmerObjectiveProfile(
            genotype_diversity_weight=5.0,
            phenotype_diversity_weight=5.0,
        ),
        decision_context={
            "ranked_candidate_pairs": rows,
            "selected_score": 10.0,
            "scores": {"confidence_proxy": 0.8},
        },
        mutation_chance=0.1,
        close_decision_threshold=0.05,
    )
    assert explanation.selected_candidate_score == rows[0]["score"]
    assert explanation.selected_pair_rank == 1
    assert explanation.first_second_score_difference == pytest.approx(0.01)
    assert len(explanation.top_candidates) == 5
    assert explanation.top_candidates[0].legality_confirmed
    assert explanation.top_candidates[0].predicted_expected_survivors == 2.0
    assert explanation.confidence_proxy == 0.8
    assert explanation.uncertainty_penalty == 0.1
    assert explanation.notable_genetics_observations
    assert explanation.warnings
