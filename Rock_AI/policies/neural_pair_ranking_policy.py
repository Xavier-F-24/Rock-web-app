"""Checkpoint-backed policy that scores only authoritative legal pairs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

import Rock_Breeding.rock_breeding_helper as breeding
import Rock_Genetics.rock_genetic_helper as genetics
from Rock_AI.datasets.breeding_record_helper import EncodedBreedingRules
from Rock_AI.datasets.pair_ranking_record_helper import FarmerObjectiveProfile
from Rock_AI.models.pair_ranker_model import PairRankerModel, PairRankerModelConfig
from Rock_AI.models.pair_scoring_helper import pair_diversity_features
from Rock_AI.representations.encoding_schema_helper import get_default_encoding_schema
from Rock_AI.representations.farm_encoder_helper import encode_farm
from Rock_AI.representations.rock_encoder_helper import encode_rock
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
        schema = get_default_encoding_schema()
        if int(checkpoint["encoding_schema_version"]) != schema.version:
            raise ValueError("Pair-ranker checkpoint encoding schema is incompatible")
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

    def _predictor_features(self, parent_a, parent_b, rules):
        if self.predictor is None:
            return np.zeros(0, dtype=np.float32), None
        result = self.predictor.predict(parent_a, parent_b, rules)
        values = {}
        values.update(result["scalar_predictions"])
        values.update(result["binary_probability_predictions"])
        for group in result["genotype_distributions"].values():
            values.update(group)
        for group in result["phenotype_distributions"].values():
            values.update(group)
        vector = np.asarray([values[name] for name in self.predictor.layout.target_names], dtype=np.float32)
        return vector, result

    def rank_legal_pairs(
        self,
        farm: object,
        rules: EncodedBreedingRules | Mapping[str, Any] | None,
        objective_profile: FarmerObjectiveProfile | None = None,
    ) -> PairRankingDecision:
        schema = get_default_encoding_schema()
        rocks = _extract_rocks(farm)
        lookup = farm if hasattr(farm, "get_rock") else _RockLookup(rocks)
        rules = EncodedBreedingRules.from_config(rules)
        objective = objective_profile or FarmerObjectiveProfile()
        validator = breeding.BreedingMaster()
        legal = []
        for left in range(len(rocks)):
            for right in range(left + 1, len(rocks)):
                if validator.validate_breeding_pair(rocks[left], rocks[right], game=lookup, warn_relatedness=False)["valid"]:
                    legal.append((rocks[left], rocks[right]))
        if not legal:
            return PairRankingDecision((), None, 0.0, no_action_reason="No legal breeding pairs")
        farm_features = encode_farm(farm, max_rocks=max(1, len(rocks)), game=lookup).global_farm_features.astype(np.float32)
        feature_names = self.checkpoint["feature_names"]
        if tuple(feature_names["parent"]) != tuple(schema.rock_matrix_feature_names):
            raise ValueError("Pair-ranker parent feature order is incompatible with current encoder")
        batches = {name: [] for name in (
            "parent_a_features", "parent_b_features", "rule_features", "farm_features",
            "objective_features", "metadata_features", "predictor_features"
        )}
        outcomes = []
        for parent_a, parent_b in legal:
            allele_difference, phenotype_difference = pair_diversity_features(parent_a, parent_b)
            relatedness, _ = validator.calculate_relatedness(lookup, parent_a, parent_b)
            batches["parent_a_features"].append(encode_rock(parent_a, schema).as_feature_vector())
            batches["parent_b_features"].append(encode_rock(parent_b, schema).as_feature_vector())
            batches["rule_features"].append(rules.feature_values)
            batches["farm_features"].append(farm_features)
            batches["objective_features"].append(objective.feature_values)
            batches["metadata_features"].append((
                (parent_a.value + parent_b.value) / schema.value_scale,
                abs(parent_a.value - parent_b.value) / schema.value_scale,
                (parent_a.generation + parent_b.generation) / schema.generation_scale,
                abs(parent_a.generation - parent_b.generation) / schema.generation_scale,
                allele_difference, phenotype_difference, relatedness,
            ))
            predictor_features, result = self._predictor_features(parent_a, parent_b, rules)
            batches["predictor_features"].append(predictor_features)
            outcomes.append(result)
        tensors = {
            key: torch.tensor(np.asarray(values, dtype=np.float32), device=self.device)
            for key, values in batches.items()
        }
        with torch.no_grad():
            normalized_scores = self.model(*(tensors[key] for key in batches))
            scores = self.normalizer.denormalize(normalized_scores).cpu().numpy()
        order = sorted(range(len(legal)), key=lambda index: (-float(scores[index]), tuple(map(str, (legal[index][0].id, legal[index][1].id)))))
        ranked = tuple(
            RankedPairDecision(
                (legal[index][0].id, legal[index][1].id),
                float(scores[index]),
                outcomes[index],
                {"parent_value_sum": float(legal[index][0].value + legal[index][1].value)},
            )
            for index in order
        )
        gap = ranked[0].neural_score - ranked[1].neural_score if len(ranked) > 1 else float("inf")
        confidence = 1.0 if len(ranked) == 1 else float(1.0 / (1.0 + np.exp(-gap)))
        warning = "Top candidate pairs are nearly tied" if len(ranked) > 1 and gap <= self.tie_warning_threshold else None
        return PairRankingDecision(ranked, ranked[0].parent_ids, confidence, warning)
