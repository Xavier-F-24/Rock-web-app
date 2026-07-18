from __future__ import annotations

import json
import math

import numpy as np
import pytest

from Rock_AI.models.neat_network_helper import (
    InstrumentedNeatNetwork,
    NeatConnectionDefinition,
    NeatNetworkArtifact,
    NeatNodeDefinition,
    load_neat_artifact,
    save_neat_artifact,
)
from Rock_AI.representations.information_provenance_helper import (
    FeatureDefinition,
    InformationAccess,
    InformationProvenance,
)
from Rock_AI.representations.player_observation_helper import PlayerDerivedEstimate
from Rock_AI.training.neat_training_helper import (
    NeatTrainingConfig,
    ScenarioSchedule,
    normalized_campaign_score,
    topology_complexity_penalty,
)


def _artifact() -> NeatNetworkArtifact:
    return NeatNetworkArtifact(
        artifact_version=1,
        topology_id="test-topology",
        observation_schema_version=2,
        normalizer_version=1,
        information_access="player",
        input_ids=(-1, -2),
        output_ids=(0,),
        input_feature_names=("visible.a", "visible.b"),
        nodes=(NeatNodeDefinition(0, 0.0, 1.0, "identity", "sum"),),
        connections=(
            NeatConnectionDefinition(-1, 0, 2.0, True),
            NeatConnectionDefinition(-2, 0, -1.0, True),
        ),
        metadata={},
    )


def test_safe_artifact_round_trip_and_trace_reproduces_output(tmp_path):
    path = tmp_path / "network.json"
    save_neat_artifact(_artifact(), path)
    loaded = load_neat_artifact(path)
    output, trace = InstrumentedNeatNetwork(loaded).activate((0.5, 0.25))
    assert output == pytest.approx((0.75,))
    assert sum(row["local_signal"] for row in trace["connection_signals"]) == pytest.approx(output[0])
    assert loaded.normalizer_version == 1
    assert loaded.input_feature_names == ("visible.a", "visible.b")


def test_scenario_sources_are_deterministic_and_separate():
    schedule = ScenarioSchedule(1234, 30, 10)
    generation_zero = schedule.rotating_training_indices(0, 8)
    assert generation_zero == schedule.rotating_training_indices(0, 8)
    assert generation_zero != schedule.rotating_training_indices(1, 8)
    assert schedule.validation_indices(5) == schedule.validation_indices(5)
    assert schedule.showcase_index == 0


def test_equal_campaign_baselines_are_finite_and_neutral():
    score = normalized_campaign_score(99.0, 5.0, 5.0)
    assert score == 0.0
    assert math.isfinite(score)


def test_default_complexity_penalty_is_bounded(tmp_path):
    config = NeatTrainingConfig(str(tmp_path), str(tmp_path / "run"))
    assert topology_complexity_penalty(config, 10_000_000) == pytest.approx(0.02)


def test_player_derived_estimate_rejects_oracle_checkpoint():
    feature = FeatureDefinition(
        "neutral_prediction",
        InformationProvenance.PLAYER_DERIVED_ESTIMATE,
    )
    estimate = PlayerDerivedEstimate(
        values=(0.5,),
        feature_definitions=(feature,),
        checkpoint_id="oracle-in-disguise",
        observation_schema_version=2,
        normalizer_version=1,
        information_access=InformationAccess.ORACLE,
        source_observation_hash="abc",
    )
    with pytest.raises(ValueError, match="Privileged predictor"):
        estimate.validate_for("abc", 2, 1)
