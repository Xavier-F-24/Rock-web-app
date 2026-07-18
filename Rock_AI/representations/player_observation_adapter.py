"""Sole conversion boundary from authoritative state to player policy inputs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any, Iterable, Mapping

import Rock_Breeding.rock_breeding_helper as breeding
import Rock_Genetics.rock_genetic_helper as genetics
from Rock_AI.datasets.breeding_record_helper import EncodedBreedingRules
from Rock_AI.datasets.pair_ranking_record_helper import (
    OBJECTIVE_FEATURE_NAMES,
    FarmerObjectiveProfile,
)

from .information_provenance_helper import (
    FeatureDefinition,
    InformationProvenance,
)
from .player_feature_normalizer import PlayerFeatureNormalizer
from .player_observation_helper import (
    PlayerCandidateObservation,
    PlayerFeatureVector,
    PlayerObservation,
    PlayerObservationSchema,
    TruthDisplayRecord,
    phenotype_categories,
)


def _definitions(names: Iterable[str]) -> tuple[FeatureDefinition, ...]:
    return tuple(
        FeatureDefinition(name, InformationProvenance.PLAYER_OBSERVATION)
        for name in names
    )


def _id_key(value: int | str) -> tuple[int, int | str]:
    return (0, value) if isinstance(value, int) else (1, str(value))


def _canonical_ids(
    left: int | str, right: int | str
) -> tuple[int | str, int | str]:
    ordered = sorted((left, right), key=_id_key)
    return ordered[0], ordered[1]


def _stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _extract_rocks(farm: object) -> list[genetics.Rock]:
    source = getattr(farm, "rocks", None)
    if source is None:
        source = getattr(farm, "rock_list", None)
    if source is None and isinstance(farm, (Mapping, list, tuple)):
        source = farm
    if source is None:
        raise TypeError(
            "farm must expose rocks or rock_list, or be a rock collection"
        )
    values = source.values() if isinstance(source, Mapping) else source
    return sorted(values, key=lambda rock: _id_key(rock.id))


class _RockLookup:
    def __init__(self, rocks: list[genetics.Rock]):
        self.rocks = {int(rock.id): rock for rock in rocks}

    def get_rock(self, rock_id: int) -> genetics.Rock | None:
        return self.rocks.get(int(rock_id))


class PlayerObservationAdapter:
    """Only supported authoritative-state to player-observation conversion."""

    def __init__(
        self,
        schema: PlayerObservationSchema | None = None,
        normalizer: PlayerFeatureNormalizer | None = None,
    ):
        self.schema = schema or PlayerObservationSchema()
        self.normalizer = normalizer or self.default_normalizer(self.schema)

    @staticmethod
    def default_normalizer(
        schema: PlayerObservationSchema,
    ) -> PlayerFeatureNormalizer:
        names = (
            schema.rock_feature_names
            + tuple(EncodedBreedingRules().feature_names)
            + schema.farm_feature_names
            + tuple(OBJECTIVE_FEATURE_NAMES)
            + schema.pair_metadata_feature_names
        )
        explicit_bounds = {
            "generation": (0.0, 20.0),
            "value": (0.0, 100.0),
            "sell_value": (0.0, 100.0),
            "score_value": (0.0, 100.0),
            "parent_count": (0.0, 2.0),
            "money": (0.0, 1000.0),
            "rock_count": (0.0, 128.0),
            "legal_pair_count": (0.0, 256.0),
            "parent_value_sum": (0.0, 200.0),
            "parent_value_difference": (0.0, 100.0),
            "parent_generation_sum": (0.0, 40.0),
            "parent_generation_difference": (0.0, 20.0),
            "observed_child_count": (0.0, 32.0),
            "observed_child_value_mean": (0.0, 100.0),
            "clutch_mean": (0.0, 20.0),
            "clutch_std": (0.0, 20.0),
            "max_clutch_size": (0.0, 20.0),
            "spore_clone_count": (0.0, 20.0),
        }
        lower: list[float] = []
        upper: list[float] = []
        for name in names:
            default = (0.0, 10.0) if name.endswith("_weight") else (0.0, 1.0)
            low, high = explicit_bounds.get(name, default)
            lower.append(low)
            upper.append(high)
        return PlayerFeatureNormalizer(names, tuple(lower), tuple(upper))

    def _vector(
        self,
        names: tuple[str, ...],
        values: tuple[float, ...],
        masks: tuple[bool, ...],
    ) -> PlayerFeatureVector:
        indices = [self.normalizer.feature_names.index(name) for name in names]
        local = PlayerFeatureNormalizer(
            names,
            tuple(self.normalizer.lower_bounds[index] for index in indices),
            tuple(self.normalizer.upper_bounds[index] for index in indices),
            version=self.normalizer.version,
            unknown_value=self.normalizer.unknown_value,
            mask_semantics=self.normalizer.mask_semantics,
        )
        normalized, visible = local.normalize(values, masks)
        return PlayerFeatureVector(normalized, visible, _definitions(names))

    @staticmethod
    def _child_history(
        rock: genetics.Rock, rocks: list[genetics.Rock]
    ) -> tuple[float, ...]:
        children = [
            child
            for child in rocks
            if rock.id in (getattr(child, "parent_ids", None) or [])
        ]
        count = len(children)
        if not count:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        statuses = [child.status for child in children]
        return (
            float(count),
            statuses.count(genetics.RockStatus.ACTIVE) / count,
            statuses.count(genetics.RockStatus.DEAD) / count,
            statuses.count(genetics.RockStatus.CRAISENED) / count,
            sum(float(child.value) for child in children) / count,
        )

    def _rock_vector(
        self, rock: genetics.Rock, rocks: list[genetics.Rock]
    ) -> PlayerFeatureVector:
        status = rock.status
        base = (
            float(rock.generation),
            float(rock.value),
            float(rock.sell_value),
            float(rock.score_value),
            float(len(getattr(rock, "parent_ids", None) or [])),
            float(bool(rock.is_market)),
            float(bool(rock.has_split)),
            float(rock.sex == genetics.Sex.MALE),
            float(rock.sex == genetics.Sex.FEMALE),
            float(status == genetics.RockStatus.ACTIVE),
            float(status == genetics.RockStatus.SOLD),
            float(status == genetics.RockStatus.DEAD),
            float(status == genetics.RockStatus.CRAISENED),
            float(status == genetics.RockStatus.BRED),
            *self._child_history(rock, rocks),
        )
        values = list(base)
        masks = [True] * len(base)
        for gene_name in sorted(genetics.GENE_SPECS):
            pair = rock.genotype.genes.get(gene_name)
            phenotype = None if pair is None else pair.phenotype
            visible = phenotype is not None
            value = str(phenotype) if visible else None
            categories = phenotype_categories(gene_name)
            values.extend(
                float(visible and value == category) for category in categories
            )
            masks.extend([visible] * len(categories))
            values.append(float(visible))
            masks.append(True)
        return self._vector(
            self.schema.rock_feature_names, tuple(values), tuple(masks)
        )

    def _farm_vector(
        self, farm: object, rocks: list[genetics.Rock], legal_count: int
    ) -> PlayerFeatureVector:
        active = [
            rock for rock in rocks if rock.status == genetics.RockStatus.ACTIVE
        ]
        visible_phenotypes = {
            (name, str(pair.phenotype))
            for rock in active
            for name, pair in rock.genotype.genes.items()
            if pair.phenotype is not None
        }
        values = [float(rock.value) for rock in rocks]
        inventory = getattr(farm, "inventory", None)
        money = float(
            getattr(farm, "money", getattr(inventory, "money", 0)) or 0
        )
        payload = (
            money,
            float(getattr(farm, "generation", 0)),
            float(len(rocks)),
            len(active) / len(rocks) if rocks else 0.0,
            (
                sum(rock.sex == genetics.Sex.MALE for rock in active)
                / len(active)
                if active
                else 0.0
            ),
            (
                sum(rock.sex == genetics.Sex.FEMALE for rock in active)
                / len(active)
                if active
                else 0.0
            ),
            sum(values) / len(values) if values else 0.0,
            max(values, default=0.0),
            float(legal_count),
            len(visible_phenotypes)
            / max(1, len(self.schema.phenotype_feature_names)),
        )
        return self._vector(
            self.schema.farm_feature_names,
            payload,
            (True,) * len(payload),
        )

    def build(
        self,
        farm: object,
        rules: EncodedBreedingRules | Mapping[str, Any] | None,
        objective: FarmerObjectiveProfile,
        *,
        remaining_breeding_actions: int | None = None,
    ) -> PlayerObservation:
        rocks = _extract_rocks(farm)
        lookup = farm if hasattr(farm, "get_rock") else _RockLookup(rocks)
        validator = breeding.BreedingMaster()
        legal: list[tuple[genetics.Rock, genetics.Rock]] = []
        for left in range(len(rocks)):
            for right in range(left + 1, len(rocks)):
                result = validator.validate_breeding_pair(
                    rocks[left],
                    rocks[right],
                    game=lookup,
                    warn_relatedness=False,
                )
                if result["valid"]:
                    legal.append((rocks[left], rocks[right]))

        encoded_rules = EncodedBreedingRules.from_config(rules)
        rule_payload = asdict(encoded_rules)
        rule_values = tuple(encoded_rules.feature_values)
        rule_masks = tuple(value is not None for value in rule_payload.values())
        rule_vector = self._vector(
            tuple(encoded_rules.feature_names), rule_values, rule_masks
        )
        objective_vector = self._vector(
            tuple(OBJECTIVE_FEATURE_NAMES),
            tuple(objective.feature_values),
            (True,) * len(OBJECTIVE_FEATURE_NAMES),
        )
        farm_vector = self._farm_vector(farm, rocks, len(legal))
        rock_vectors = {
            rock.id: self._rock_vector(rock, rocks) for rock in rocks
        }
        candidates: list[PlayerCandidateObservation] = []
        for parent_a, parent_b in legal:
            canonical = _canonical_ids(parent_a.id, parent_b.id)
            first, second = (
                (parent_a, parent_b)
                if canonical[0] == parent_a.id
                else (parent_b, parent_a)
            )
            visible_a = tuple(
                str(first.genotype.genes[name].phenotype)
                for name in sorted(genetics.GENE_SPECS)
            )
            visible_b = tuple(
                str(second.genotype.genes[name].phenotype)
                for name in sorted(genetics.GENE_SPECS)
            )
            relatedness, _ = validator.calculate_relatedness(
                lookup, first, second
            )
            metadata_values = (
                float(first.value + second.value),
                float(abs(first.value - second.value)),
                float(first.generation + second.generation),
                float(abs(first.generation - second.generation)),
                sum(left != right for left, right in zip(visible_a, visible_b))
                / max(1, len(visible_a)),
                float(relatedness),
                float(relatedness / 2.0),
            )
            metadata = self._vector(
                self.schema.pair_metadata_feature_names,
                metadata_values,
                (True,) * len(metadata_values),
            )
            groups = (
                rock_vectors[first.id],
                rock_vectors[second.id],
                rule_vector,
                farm_vector,
                objective_vector,
                metadata,
            )
            candidate_payload = {
                "schema_version": self.schema.version,
                "normalizer_version": self.normalizer.version,
                "feature_names": [
                    name for group in groups for name in group.feature_names
                ],
                "values_and_masks": [
                    value for group in groups for value in group.model_values()
                ],
                "parent_ids": list(canonical),
                "public_rules": rule_payload,
                "objective": objective.to_dict(),
                "predictor_checkpoint_id": None,
            }
            candidates.append(
                PlayerCandidateObservation(
                    canonical_parent_ids=canonical,
                    parent_a=rock_vectors[first.id],
                    parent_b=rock_vectors[second.id],
                    public_rules=rule_vector,
                    visible_farm=farm_vector,
                    objective=objective_vector,
                    visible_pair_metadata=metadata,
                    candidate_hash=_stable_hash(candidate_payload),
                )
            )

        observation_payload = {
            "schema_version": self.schema.version,
            "normalizer_version": self.normalizer.version,
            "generation": int(getattr(farm, "generation", 0)),
            "candidate_hashes": [
                candidate.candidate_hash for candidate in candidates
            ],
        }
        remaining = remaining_breeding_actions
        if remaining is None:
            maximum = int(
                getattr(farm, "max_pairs_per_generation", len(candidates))
            )
            remaining = max(
                0, maximum - len(getattr(farm, "breeding_queue", ()))
            )
        return PlayerObservation(
            schema_version=self.schema.version,
            normalizer_version=self.normalizer.version,
            generation=int(getattr(farm, "generation", 0)),
            remaining_breeding_actions=int(remaining),
            candidates=tuple(candidates),
            observation_hash=_stable_hash(observation_payload),
        )

    @staticmethod
    def truth_display(rock: genetics.Rock) -> TruthDisplayRecord:
        ordinary = tuple(
            (name, int(pair.allele_a.value), int(pair.allele_b.value))
            for name, pair in sorted(rock.genotype.genes.items())
        )
        death = tuple(
            (name, int(pair.allele_a.value), int(pair.allele_b.value))
            for name, pair in sorted(rock.death_genes.genes.items())
        )
        return TruthDisplayRecord(rock.id, ordinary, death)
