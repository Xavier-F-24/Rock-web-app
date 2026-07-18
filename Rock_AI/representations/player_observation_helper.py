"""Immutable records defining the player/oracle information boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import Rock_Genetics.rock_genetic_helper as genetics

from .information_provenance_helper import (
    FeatureDefinition,
    InformationAccess,
    validate_player_feature_definitions,
)


PLAYER_OBSERVATION_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class PlayerFeatureVector:
    values: tuple[float, ...]
    visibility_mask: tuple[bool, ...]
    definitions: tuple[FeatureDefinition, ...]

    def __post_init__(self) -> None:
        if not (
            len(self.values) == len(self.visibility_mask) == len(self.definitions)
        ):
            raise ValueError(
                "Player feature values, masks, and definitions must align"
            )
        validate_player_feature_definitions(self.definitions)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(feature.name for feature in self.definitions)

    def model_values(self) -> tuple[float, ...]:
        return self.values + tuple(float(value) for value in self.visibility_mask)


@dataclass(frozen=True)
class PlayerDerivedEstimate:
    values: tuple[float, ...]
    feature_definitions: tuple[FeatureDefinition, ...]
    checkpoint_id: str
    observation_schema_version: int
    normalizer_version: int
    information_access: InformationAccess
    source_observation_hash: str

    def validate_for(
        self,
        observation_hash: str,
        schema_version: int,
        normalizer_version: int,
    ) -> None:
        if self.information_access is not InformationAccess.PLAYER:
            raise ValueError(
                "Privileged predictor output cannot be used by a player policy"
            )
        if self.observation_schema_version != schema_version:
            raise ValueError("Predictor observation schema is incompatible")
        if self.normalizer_version != normalizer_version:
            raise ValueError("Predictor normalizer is incompatible")
        if self.source_observation_hash != observation_hash:
            raise ValueError(
                "Predictor output was not reproduced from this player observation"
            )
        validate_player_feature_definitions(self.feature_definitions)


@dataclass(frozen=True)
class PlayerCandidateObservation:
    canonical_parent_ids: tuple[int | str, int | str]
    parent_a: PlayerFeatureVector
    parent_b: PlayerFeatureVector
    public_rules: PlayerFeatureVector
    visible_farm: PlayerFeatureVector
    objective: PlayerFeatureVector
    visible_pair_metadata: PlayerFeatureVector
    candidate_hash: str
    derived_estimate: PlayerDerivedEstimate | None = None


@dataclass(frozen=True)
class PlayerObservation:
    schema_version: int
    normalizer_version: int
    generation: int
    remaining_breeding_actions: int
    candidates: tuple[PlayerCandidateObservation, ...]
    observation_hash: str
    information_access: InformationAccess = InformationAccess.PLAYER

    def __post_init__(self) -> None:
        if self.information_access is not InformationAccess.PLAYER:
            raise ValueError("PlayerObservation must have PLAYER access")


@dataclass(frozen=True)
class OracleObservation:
    trace_id: str
    payload: tuple[tuple[str, Any], ...]
    information_access: InformationAccess = InformationAccess.ORACLE


@dataclass(frozen=True)
class OracleLabelRecord:
    candidate_hash: str
    utility: float
    components: tuple[tuple[str, float], ...]
    information_access: InformationAccess = InformationAccess.ORACLE


@dataclass(frozen=True)
class TruthDisplayRecord:
    rock_id: int | str
    ordinary_genotype: tuple[tuple[str, int, int], ...]
    death_genotype: tuple[tuple[str, int, int], ...]
    information_access: InformationAccess = InformationAccess.ORACLE


ROCK_BASE_FEATURES = (
    "generation",
    "value",
    "sell_value",
    "score_value",
    "parent_count",
    "is_market",
    "has_split",
    "sex_male",
    "sex_female",
    "status_active",
    "status_sold",
    "status_dead",
    "status_craisened",
    "status_bred",
    "observed_child_count",
    "observed_active_child_fraction",
    "observed_dead_child_fraction",
    "observed_craisened_child_fraction",
    "observed_child_value_mean",
)

FARM_FEATURES = (
    "money",
    "generation",
    "rock_count",
    "active_rock_fraction",
    "active_male_fraction",
    "active_female_fraction",
    "visible_value_mean",
    "visible_value_max",
    "legal_pair_count",
    "visible_phenotype_coverage",
)

PAIR_METADATA_FEATURES = (
    "parent_value_sum",
    "parent_value_difference",
    "parent_generation_sum",
    "parent_generation_difference",
    "visible_phenotype_difference_fraction",
    "relatedness_r",
    "offspring_inbreeding_f",
)


def phenotype_categories(gene_name: str) -> tuple[str, ...]:
    spec = genetics.GENE_SPECS[gene_name]
    values = {"n/a"}
    values.update(option.name for option in spec.options.values())
    values.update(state.name for state in spec.states.values())
    values.update(state.name for state in spec.special_states.values())
    if spec.required_gender_states is not None:
        values.add(str(spec.required_gender_states))
    return tuple(sorted(values))


@dataclass(frozen=True)
class PlayerObservationSchema:
    version: int = PLAYER_OBSERVATION_SCHEMA_VERSION

    @property
    def phenotype_feature_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for gene_name in sorted(genetics.GENE_SPECS):
            names.extend(
                f"phenotype.{gene_name}={value}"
                for value in phenotype_categories(gene_name)
            )
            names.append(f"phenotype.{gene_name}.visible")
        return tuple(names)

    @property
    def rock_feature_names(self) -> tuple[str, ...]:
        return ROCK_BASE_FEATURES + self.phenotype_feature_names

    @property
    def farm_feature_names(self) -> tuple[str, ...]:
        return FARM_FEATURES

    @property
    def pair_metadata_feature_names(self) -> tuple[str, ...]:
        return PAIR_METADATA_FEATURES


@dataclass(frozen=True)
class OracleObservationSchema:
    version: int = 1
