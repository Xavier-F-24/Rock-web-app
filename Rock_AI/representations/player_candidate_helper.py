"""Shared model-ready representations for player-safe candidate observations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from .player_observation_helper import (
    PlayerCandidateObservation,
    PlayerObservation,
)


@dataclass(frozen=True)
class PlayerCandidateArrays:
    parent_a_features: np.ndarray
    parent_b_features: np.ndarray
    rule_features: np.ndarray
    farm_features: np.ndarray
    objective_features: np.ndarray
    metadata_features: np.ndarray
    predictor_features: np.ndarray


def candidate_arrays(
    candidate: PlayerCandidateObservation,
) -> PlayerCandidateArrays:
    if not isinstance(candidate, PlayerCandidateObservation):
        raise TypeError("Gameplay candidate scoring requires PlayerCandidateObservation")
    predictor = candidate.derived_estimate
    predictor_values = () if predictor is None else predictor.values
    return PlayerCandidateArrays(
        parent_a_features=np.asarray(
            candidate.parent_a.model_values(), dtype=np.float32
        ),
        parent_b_features=np.asarray(
            candidate.parent_b.model_values(), dtype=np.float32
        ),
        rule_features=np.asarray(
            candidate.public_rules.model_values(), dtype=np.float32
        ),
        farm_features=np.asarray(
            candidate.visible_farm.model_values(), dtype=np.float32
        ),
        objective_features=np.asarray(
            candidate.objective.model_values(), dtype=np.float32
        ),
        metadata_features=np.asarray(
            candidate.visible_pair_metadata.model_values(), dtype=np.float32
        ),
        predictor_features=np.asarray(predictor_values, dtype=np.float32),
    )


def neat_symmetric_candidate_vector(
    candidate: PlayerCandidateObservation,
) -> np.ndarray:
    arrays = candidate_arrays(candidate)
    parent_sum = arrays.parent_a_features + arrays.parent_b_features
    parent_difference = np.abs(
        arrays.parent_a_features - arrays.parent_b_features
    )
    parent_product = arrays.parent_a_features * arrays.parent_b_features
    return np.concatenate(
        (
            parent_sum,
            parent_difference,
            parent_product,
            arrays.rule_features,
            arrays.farm_features,
            arrays.objective_features,
            arrays.metadata_features,
            arrays.predictor_features,
        )
    ).astype(np.float64)


def candidate_model_input_hash(
    candidate: PlayerCandidateObservation,
    *,
    predictor_checkpoint_id: str | None = None,
    predictor_feature_names: tuple[str, ...] = (),
    predictor_values: np.ndarray | tuple[float, ...] = (),
) -> str:
    payload = {
        "base_candidate_hash": candidate.candidate_hash,
        "canonical_parent_ids": list(candidate.canonical_parent_ids),
        "predictor_checkpoint_id": predictor_checkpoint_id,
        "predictor_feature_names": list(predictor_feature_names),
        "predictor_values": [float(value) for value in predictor_values],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def observation_batches(
    observation: PlayerObservation,
) -> tuple[dict[str, np.ndarray], tuple[PlayerCandidateObservation, ...]]:
    if not isinstance(observation, PlayerObservation):
        raise TypeError("Gameplay policies require PlayerObservation")
    if not observation.candidates:
        return {}, ()
    rows = [candidate_arrays(candidate) for candidate in observation.candidates]
    keys = tuple(PlayerCandidateArrays.__dataclass_fields__)
    batches = {
        key: np.stack([getattr(row, key) for row in rows]).astype(np.float32)
        for key in keys
    }
    return batches, observation.candidates
