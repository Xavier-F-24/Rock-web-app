from __future__ import annotations

import copy

import pytest

import Rock_Genetics.rock_genetic_helper as genetics
from Rock_AI.datasets.pair_ranking_dataset_generator import (
    PairRankingDatasetGenerator,
)
from Rock_AI.datasets.pair_ranking_record_helper import FarmerObjectiveProfile
from Rock_AI.representations.information_provenance_helper import (
    FeatureDefinition,
    InformationAccess,
    InformationProvenance,
)
from Rock_AI.representations.player_candidate_helper import observation_batches
from Rock_AI.representations.player_feature_normalizer import (
    PlayerFeatureNormalizer,
)
from Rock_AI.representations.player_observation_adapter import (
    PlayerObservationAdapter,
)
from Rock_AI.representations.player_observation_helper import (
    OracleObservation,
    PlayerFeatureVector,
)
from Rock_AI.training.training_config_helper import PairRankingDataConfig


def _farm():
    config = PairRankingDataConfig(
        number_of_farms=1,
        trials_per_pair=1,
        minimum_rocks=4,
        maximum_rocks=4,
    )
    return PairRankingDatasetGenerator(config).create_procedural_farm(0)


def test_hidden_death_genes_do_not_change_player_observation():
    farm = _farm()
    changed = copy.deepcopy(farm)
    rock = next(iter(changed.rocks.values()))
    pair = next(iter(rock.death_genes.genes.values()))
    pair.allele_a = genetics.Allele(pair.allele_a.value + 100)
    pair.allele_b = genetics.Allele(pair.allele_b.value + 200)

    adapter = PlayerObservationAdapter()
    first = adapter.build(farm, None, FarmerObjectiveProfile())
    second = adapter.build(changed, None, FarmerObjectiveProfile())

    assert first.observation_hash == second.observation_hash
    assert [row.candidate_hash for row in first.candidates] == [
        row.candidate_hash for row in second.candidates
    ]
    names = {
        definition.name
        for candidate in first.candidates
        for vector in (candidate.parent_a, candidate.parent_b)
        for definition in vector.definitions
    }
    assert not any("allele" in name or "death_gene" in name for name in names)


def test_oracle_or_disguised_truth_cannot_enter_player_policy_boundary():
    oracle = OracleObservation("oracle", (("harmless_name", 1.0),))
    with pytest.raises(TypeError, match="PlayerObservation"):
        observation_batches(oracle)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="privileged"):
        PlayerFeatureVector(
            values=(1.0,),
            visibility_mask=(True,),
            definitions=(
                FeatureDefinition(
                    "neutral_score",
                    InformationProvenance.ORACLE_TRUTH,
                ),
            ),
        )


def test_normalizer_masks_round_trip_and_unknown_is_explicit():
    normalizer = PlayerFeatureNormalizer(
        ("known", "unknown"),
        (0.0, 0.0),
        (10.0, 10.0),
    )
    values, mask = normalizer.normalize((5.0, 9.0), (True, False))
    assert values == (0.5, 0.0)
    assert mask == (True, False)
    restored = PlayerFeatureNormalizer.from_dict(normalizer.to_dict())
    assert restored == normalizer


def test_candidate_identity_is_parent_order_invariant_and_r_f_are_public():
    observation = PlayerObservationAdapter().build(
        _farm(), None, FarmerObjectiveProfile()
    )
    candidate = observation.candidates[0]
    assert candidate.canonical_parent_ids == tuple(
        sorted(candidate.canonical_parent_ids)
    )
    metadata = dict(
        zip(
            candidate.visible_pair_metadata.feature_names,
            candidate.visible_pair_metadata.values,
        )
    )
    assert metadata["offspring_inbreeding_f"] == pytest.approx(
        metadata["relatedness_r"] / 2.0
    )
    assert observation.information_access is InformationAccess.PLAYER
