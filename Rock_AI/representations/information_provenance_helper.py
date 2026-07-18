"""Explicit provenance rules for every value exposed to an AI policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class InformationProvenance(str, Enum):
    ORACLE_TRUTH = "oracle_truth"
    PLAYER_OBSERVATION = "player_observation"
    PLAYER_DERIVED_ESTIMATE = "player_derived_estimate"


class InformationAccess(str, Enum):
    PLAYER = "player"
    ORACLE = "oracle"


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    provenance: InformationProvenance
    transform: str = "identity"
    unknown_representation: float = 0.0
    mask_semantics: str = "true_when_observed"

    @property
    def policy_safe(self) -> bool:
        return self.provenance in {
            InformationProvenance.PLAYER_OBSERVATION,
            InformationProvenance.PLAYER_DERIVED_ESTIMATE,
        }


def validate_player_feature_definitions(
    definitions: tuple[FeatureDefinition, ...],
) -> None:
    unsafe = [feature.name for feature in definitions if not feature.policy_safe]
    if unsafe:
        raise ValueError(
            f"Player policy features contain privileged provenance: {unsafe}"
        )
