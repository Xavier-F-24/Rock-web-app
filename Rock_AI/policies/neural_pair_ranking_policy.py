"""Checkpoint-backed policy that scores only authoritative legal pairs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from Rock_AI.datasets.pair_ranking_record_helper import FarmerObjectiveProfile
from Rock_AI.models.pair_ranker_model import PairRankerModel, PairRankerModelConfig
from Rock_AI.models.model_trace_helper import ModelTrace
from Rock_AI.models.pytorch_trace_helper import trace_dense_model
from Rock_AI.representations.player_candidate_helper import candidate_arrays, candidate_model_input_hash
from Rock_AI.representations.player_observation_helper import (
    PLAYER_OBSERVATION_SCHEMA_VERSION,
    PlayerObservation,
)
from Rock_AI.training.train_pair_ranker import UtilityNormalizer, load_pair_ranker_checkpoint


@dataclass(frozen=True)
class RankedPairDecision:
    parent_ids: tuple[int | str, int | str]
    neural_score: float
    predicted_breeding_outcomes: dict[str, Any] | None = None
    score_components: dict[str, float] | None = None


@dataclass(frozen=True)
class PairRankingDecision:
    ranked_pairs: tuple[RankedPairDecision, ...]
    selected_best_pair: tuple[int | str, int | str] | None
    confidence_proxy: float
    nearly_tied_warning: str | None = None
    no_action_reason: str | None = None


def _extract_rocks(farm: object) -> list[genetics.Rock]:
    source = getattr(farm, "rocks", farm)
    values = source.values() if isinstance(source, Mapping) else source
    return sorted(values, key=lambda rock: (str(type(rock.id)), str(rock.id)))


class _RockLookup:
    def __init__(self, rocks: Iterable[genetics.Rock]):
        self.rocks = {int(rock.id): rock for rock in rocks}

    def get_rock(self, rock_id: int) -> genetics.Rock | None:
        return self.rocks.get(int(rock_id))


class NeuralPairRankingPolicy:
    def __init__(self, model, checkpoint, normalizer, device, predictor=None, tie_warning_threshold=0.05):
        self.model = model.eval()
        self.checkpoint = checkpoint
        self.normalizer = normalizer
        self.device = device
        self.predictor = predictor
        self.tie_warning_threshold = tie_warning_threshold
        self.latest_model_trace: ModelTrace | None = None

    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path,
        *,
        predictor_checkpoint: str | Path | None = None,
        device: str = "cpu",
    ) -> "NeuralPairRankingPolicy":
        selected = torch.device(device)
        checkpoint = load_pair_ranker_checkpoint(checkpoint_path, selected)
        if checkpoint.get("information_access") != "player":
            raise ValueError(
                "Privileged pair-ranker checkpoints cannot run as player agents"
            )
        if int(checkpoint["observation_schema_version"]) != PLAYER_OBSERVATION_SCHEMA_VERSION:
            raise ValueError("Pair-ranker player-observation schema is incompatible")
        values = dict(checkpoint["model_architecture_config"])
        values["encoder_hidden_dimensions"] = tuple(values["encoder_hidden_dimensions"])
        values["trunk_hidden_dimensions"] = tuple(values["trunk_hidden_dimensions"])
        model = PairRankerModel(PairRankerModelConfig(**values)).to(selected)
        model.load_state_dict(checkpoint["model_state_dict"])
        norm = checkpoint["normalization_statistics"]
        predictor = None
        if values["predictor_feature_dimension"]:
            if predictor_checkpoint is None:
                raise ValueError("This ranker requires a breeding-predictor checkpoint")
            from Rock_AI.evaluation.predictor_evaluator import BreedingPredictor

            predictor = BreedingPredictor.load(predictor_checkpoint, device=device)
            if len(predictor.layout.target_names) != values["predictor_feature_dimension"]:
                raise ValueError("Breeding predictor output schema is incompatible with ranker")
        instance = cls(
            model,
            checkpoint,
            UtilityNormalizer(norm["mean"], norm["standard_deviation"]),
            selected,
            predictor,
        )
        instance.checkpoint_path = str(Path(checkpoint_path))
        instance.predictor_checkpoint_path = (
            str(Path(predictor_checkpoint)) if predictor_checkpoint is not None else None
        )
        return instance

    def _predictor_features(self, candidate):
        if self.predictor is None:
            return np.zeros(0, dtype=np.float32), None
        result = self.predictor.predict_candidate(candidate)
        values = {}
        values.update(result["scalar_predictions"])
        values.update(result["binary_probability_predictions"])
        for group in result["genotype_distributions"].values():
            values.update(group)
        for group in result["phenotype_distributions"].values():
            values.update(group)
        vector = np.asarray([values[name] for name in self.predictor.layout.target_names], dtype=np.float32)
        return vector, result

    def rank_observation(
        self,
        observation: PlayerObservation,
    ) -> PairRankingDecision:
        if not isinstance(observation, PlayerObservation):
            raise TypeError("NeuralPairRankingPolicy requires PlayerObservation")
        if observation.schema_version != int(
            self.checkpoint["observation_schema_version"]
        ):
            raise ValueError("Observation schema does not match pair-ranker checkpoint")
        if not observation.candidates:
            return PairRankingDecision((), None, 0.0, no_action_reason="No legal breeding pairs")
        feature_names = self.checkpoint["feature_names"]
        first = observation.candidates[0]
        expected = {
            "parent": first.parent_a.feature_names
            + tuple(
                f"{name}.observed_mask" for name in first.parent_a.feature_names
            ),
            "rules": first.public_rules.feature_names
            + tuple(
                f"{name}.observed_mask"
                for name in first.public_rules.feature_names
            ),
            "farm": first.visible_farm.feature_names
            + tuple(
                f"{name}.observed_mask" for name in first.visible_farm.feature_names
            ),
            "objective": first.objective.feature_names
            + tuple(
                f"{name}.observed_mask" for name in first.objective.feature_names
            ),
            "metadata": first.visible_pair_metadata.feature_names
            + tuple(
                f"{name}.observed_mask"
                for name in first.visible_pair_metadata.feature_names
            ),
        }
        for group, names in expected.items():
            if tuple(feature_names[group]) != tuple(names):
                raise ValueError(
                    f"Pair-ranker {group} feature order is incompatible"
                )
        batches = {name: [] for name in (
            "parent_a_features", "parent_b_features", "rule_features", "farm_features",
            "objective_features", "metadata_features", "predictor_features"
        )}
        outcomes = []
        input_hashes = []
        for candidate in observation.candidates:
            arrays = candidate_arrays(candidate)
            batches["parent_a_features"].append(arrays.parent_a_features)
            batches["parent_b_features"].append(arrays.parent_b_features)
            batches["rule_features"].append(arrays.rule_features)
            batches["farm_features"].append(arrays.farm_features)
            batches["objective_features"].append(arrays.objective_features)
            batches["metadata_features"].append(arrays.metadata_features)
            predictor_features, result = self._predictor_features(candidate)
            batches["predictor_features"].append(predictor_features)
            outcomes.append(result)
            input_hashes.append(candidate_model_input_hash(
                candidate,
                predictor_checkpoint_id=(
                    getattr(self.predictor, "checkpoint_path", None)
                    if self.predictor is not None else None
                ),
                predictor_feature_names=(
                    tuple(self.predictor.layout.target_names)
                    if self.predictor is not None else ()
                ),
                predictor_values=predictor_features,
            ))
        tensors = {
            key: torch.tensor(np.asarray(values, dtype=np.float32), device=self.device)
            for key, values in batches.items()
        }
        with torch.no_grad():
            normalized_scores = self.model(*(tensors[key] for key in batches))
            scores = self.normalizer.denormalize(normalized_scores).cpu().numpy()
        order = sorted(
            range(len(observation.candidates)),
            key=lambda index: (
                -float(scores[index]),
                tuple(map(str, observation.candidates[index].canonical_parent_ids)),
            ),
        )
        ranked = tuple(
            RankedPairDecision(
                observation.candidates[index].canonical_parent_ids,
                float(scores[index]),
                outcomes[index],
                {
                    "candidate_hash": input_hashes[index],
                },
            )
            for index in order
        )
        selected_index = order[0]
        selected_inputs = tuple(
            tensors[key][selected_index:selected_index + 1]
            for key in batches
        )
        raw_trace = trace_dense_model(self.model, selected_inputs)
        flat_feature_names = (
            tuple(f"parent_a.{name}" for name in feature_names.get("parent", ()))
            + tuple(f"parent_b.{name}" for name in feature_names.get("parent", ()))
            + tuple(f"rules.{name}" for name in feature_names.get("rules", ()))
            + tuple(f"farm.{name}" for name in feature_names.get("farm", ()))
            + tuple(f"objective.{name}" for name in feature_names.get("objective", ()))
            + tuple(f"metadata.{name}" for name in feature_names.get("metadata", ()))
            + tuple(f"predictor.{name}" for name in feature_names.get("predictor", ()))
        )
        flat_values = tuple(
            float(value)
            for key in batches
            for value in tensors[key][selected_index].detach().cpu().numpy()
        )
        checkpoint_id = getattr(self, "checkpoint_path", "in-memory-pytorch-ranker")
        self.latest_model_trace = ModelTrace(
            model_type="pytorch_pair_ranker",
            checkpoint_id=checkpoint_id,
            topology_id=f"pair-ranker-{self.checkpoint.get('epoch', 'unknown')}",
            observation_schema_version=observation.schema_version,
            normalizer_version=observation.normalizer_version,
            observation_hash=observation.observation_hash,
            feature_names=flat_feature_names,
            input_values=flat_values,
            node_activations=raw_trace["node_activations"],
            connection_signals=tuple(raw_trace["connection_signals"]),
            output_scores={
                "|".join(map(str, observation.candidates[index].canonical_parent_ids)): float(scores[index])
                for index in range(len(scores))
            },
            selected_candidate_ids=ranked[0].parent_ids,
            candidate_input_hashes=tuple(input_hashes),
            metadata={
                "raw_normalized_output": raw_trace["output"],
                "trace_connection_limit_per_layer": 30,
            },
        )
        gap = ranked[0].neural_score - ranked[1].neural_score if len(ranked) > 1 else float("inf")
        confidence = 1.0 if len(ranked) == 1 else float(1.0 / (1.0 + np.exp(-gap)))
        warning = "Top candidate pairs are nearly tied" if len(ranked) > 1 and gap <= self.tie_warning_threshold else None
        return PairRankingDecision(ranked, ranked[0].parent_ids, confidence, warning)

    def rank_legal_pairs(self, *args, **kwargs):
        raise TypeError(
            "Raw farms are not accepted; use rank_observation(PlayerObservation)"
        )
