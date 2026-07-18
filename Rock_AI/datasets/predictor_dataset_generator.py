"""Diverse, reproducible examples for a future breeding predictor."""

from __future__ import annotations

import itertools
import random
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

import Rock_Breeding.rock_breeding_helper as breeding
import Rock_Genetics.rock_genetic_helper as genetics
from Rock_AI.datasets.breeding_record_helper import EncodedBreedingRules
from Rock_AI.datasets.predictor_example_helper import PredictorExample, PredictorTargetSchema
from Rock_AI.datasets.pair_ranking_record_helper import FarmerObjectiveProfile
from Rock_AI.evaluation.breeding_expectation_helper import BreedingExpectationEvaluator
from Rock_AI.representations.encoding_schema_helper import EncodingSchema, get_default_encoding_schema
from Rock_AI.representations.player_observation_adapter import PlayerObservationAdapter
from Rock_AI.training.training_config_helper import TrainingDataConfig


CONTEXT_FEATURE_NAMES = (
    "parent_a_generation_normalized",
    "parent_b_generation_normalized",
    "generation_difference_normalized",
    "parent_a_value_normalized",
    "parent_b_value_normalized",
)
PROCEDURAL_PROFILES = (
    "random",
    "homozygous",
    "heterozygous",
    "low_value",
    "high_value",
    "rare_traits",
)


class _RockLookup:
    def __init__(self, rocks: Iterable[genetics.Rock]):
        self.rocks = {int(rock.id): rock for rock in rocks}

    def get_rock(self, rock_id: int) -> genetics.Rock | None:
        return self.rocks.get(int(rock_id))


def _extract_rocks(source: object) -> list[genetics.Rock]:
    rocks = getattr(source, "rocks", None)
    if rocks is None:
        rocks = getattr(source, "rock_list", None)
    if rocks is None and isinstance(source, (Mapping, list, tuple)):
        rocks = source
    if rocks is None:
        raise TypeError("source must expose rocks or rock_list, or be a rock collection")
    values = rocks.values() if isinstance(rocks, Mapping) else rocks
    return sorted(values, key=lambda rock: int(rock.id))


class PredictorDatasetGenerator:
    def __init__(
        self,
        config: TrainingDataConfig,
        *,
        schema: EncodingSchema | None = None,
        expectation_evaluator: BreedingExpectationEvaluator | None = None,
    ):
        self.config = config
        self.schema = schema or get_default_encoding_schema()
        self.expectation_evaluator = expectation_evaluator or BreedingExpectationEvaluator()
        self.target_schema = PredictorTargetSchema.build(config.value_thresholds, self.schema)
        self.rng = random.Random(config.seed)
        self.player_adapter = PlayerObservationAdapter()

    def _sample_rules(self) -> EncodedBreedingRules:
        uniform = self.rng.uniform
        return EncodedBreedingRules.from_config(
            {
                "mutation_chance": uniform(*self.config.mutation_chance_range),
                "child_death_chance": uniform(*self.config.death_chance_range),
                "craisen_chance": uniform(*self.config.craisen_chance_range),
                "clutch_mean": uniform(*self.config.clutch_mean_range),
                "clutch_std": uniform(*self.config.clutch_std_range),
            }
        )

    def _context(self, parent_a: genetics.Rock, parent_b: genetics.Rock) -> np.ndarray:
        if not self.config.include_context_features:
            return np.zeros(0, dtype=np.float32)
        return np.asarray(
            (
                parent_a.generation / self.schema.generation_scale,
                parent_b.generation / self.schema.generation_scale,
                abs(parent_a.generation - parent_b.generation) / self.schema.generation_scale,
                parent_a.value / self.schema.value_scale,
                parent_b.value / self.schema.value_scale,
            ),
            dtype=np.float32,
        )

    @staticmethod
    def _pair_key(parent_a: genetics.Rock, parent_b: genetics.Rock) -> str:
        ids = sorted((str(parent_a.id), str(parent_b.id)))
        return f"{ids[0]}|{ids[1]}"

    def _build_example(
        self,
        parent_a: genetics.Rock,
        parent_b: genetics.Rock,
        *,
        pair_index: int,
        parent_source_type: str,
        lineage_group_id: str,
        profile: str | None = None,
    ) -> PredictorExample:
        rules = self._sample_rules()
        evaluation_seed = self.config.seed + 10_000 + pair_index * self.config.trials_per_pair
        expectation = self.expectation_evaluator.evaluate(
            parent_a,
            parent_b,
            rules=rules,
            trial_count=self.config.trials_per_pair,
            seed=evaluation_seed,
            value_thresholds=self.config.value_thresholds,
        )
        player_observation = self.player_adapter.build(
            [parent_a, parent_b],
            rules,
            FarmerObjectiveProfile(),
            remaining_breeding_actions=1,
        )
        if len(player_observation.candidates) != 1:
            raise ValueError("Predictor parent pair did not produce one legal player candidate")
        candidate = player_observation.candidates[0]
        uncertainty = {
            "expected_raw_clutch_size_standard_error": expectation.expected_raw_clutch_size.standard_error,
            "expected_survivor_count_standard_error": expectation.expected_survivor_count.standard_error,
            "expected_average_surviving_child_value_standard_error": expectation.expected_average_surviving_child_value.standard_error,
            "expected_maximum_surviving_child_value_standard_error": expectation.expected_maximum_surviving_child_value.standard_error,
        }
        metadata = {
            "example_id": f"predictor-{self.config.seed}-{pair_index:08d}",
            "parent_ids": [parent_a.id, parent_b.id],
            "parent_pair_key": self._pair_key(parent_a, parent_b),
            "lineage_group_id": lineage_group_id,
            "evaluation_seed": evaluation_seed,
            "monte_carlo_trial_count": self.config.trials_per_pair,
            "uncertainty_estimates": uncertainty,
            "source_type": "hybrid",
            "parent_source_type": parent_source_type,
            "procedural_profile": profile,
            "game_rules_version": self.config.game_rules_version,
            "encoding_schema_version": self.schema.version,
            "observation_schema_version": player_observation.schema_version,
            "information_access": "player",
            "player_feature_normalizer": self.player_adapter.normalizer.to_dict(),
            "parent_feature_names": list(candidate.parent_a.feature_names)
            + [f"{name}.observed_mask" for name in candidate.parent_a.feature_names],
            "rule_encoding": rules.to_dict(),
            "field_methods": dict(expectation.field_methods),
        }
        return PredictorExample.from_expectation(
            np.asarray(candidate.parent_a.model_values(), dtype=np.float32),
            np.asarray(candidate.parent_b.model_values(), dtype=np.float32),
            np.asarray(candidate.public_rules.model_values(), dtype=np.float32),
            self._context(parent_a, parent_b),
            expectation,
            metadata,
            player_observation.schema_version,
        )

    @staticmethod
    def _allele_frequency(gene_name: str) -> dict[int, float]:
        spec = genetics.GENE_SPECS[gene_name]
        counts = {allele: 0 for allele in spec.options}
        for roll in range(1, 21):
            allele = genetics.GenomeFactory.get_allele_from_roll(roll, spec).value
            counts[allele] += 1
        return {allele: count / 20 for allele, count in counts.items()}

    def _profile_traits(self, profile: str, rng: random.Random) -> dict[str, tuple[int, int]]:
        selected: dict[str, tuple[int, int]] = {}
        for gene_name in self.schema.gene_names:
            spec = genetics.GENE_SPECS[gene_name]
            alleles = tuple(sorted(spec.options))
            if profile == "homozygous":
                allele = rng.choice(alleles)
                selected[gene_name] = (allele, allele)
            elif profile == "heterozygous":
                selected[gene_name] = (
                    tuple(rng.sample(alleles, 2)) if len(alleles) > 1 else (alleles[0], alleles[0])
                )
            elif profile == "low_value":
                allele = min(alleles, key=lambda value: (spec.options[value].cost, value))
                selected[gene_name] = (allele, allele)
            elif profile == "high_value":
                allele = max(alleles, key=lambda value: (spec.options[value].cost, -value))
                selected[gene_name] = (allele, allele)
            elif profile == "rare_traits":
                frequencies = self._allele_frequency(gene_name)
                allele = min(alleles, key=lambda value: (frequencies[value], -spec.options[value].cost))
                selected[gene_name] = (allele, allele)
        return selected

    def _procedural_rock(
        self,
        rock_id: int,
        sex: genetics.Sex,
        profile: str,
        seed: int,
    ) -> genetics.Rock:
        rng = random.Random(seed)
        factory = genetics.GenomeFactory(rng=random.Random(seed + 1))
        if profile == "random":
            genome = factory.make_random_rock_genome()
        else:
            genome = factory.make_selected_rock_genome(
                self._profile_traits(profile, rng), random_fill=False
            )
        rock = genetics.Rock(
            id=rock_id,
            sex=sex,
            name=genetics.RockName(given=f"Synthetic{rock_id}"),
            genotype=genome,
            death_genes=factory.make_death_genes(),
            generation=0,
        )
        genetics.ExpressionEngine().instantiate_phenotype(rock)
        genetics.ValueCalculator().set_rock_value(rock)
        return rock

    def generate_procedural_examples(self) -> list[PredictorExample]:
        examples: list[PredictorExample] = []
        for index in range(self.config.number_of_parent_pairs):
            profile = PROCEDURAL_PROFILES[index % len(PROCEDURAL_PROFILES)]
            parent_a = self._procedural_rock(
                1_000_000 + index * 2,
                genetics.Sex.MALE,
                profile,
                self.config.seed + index * 10,
            )
            parent_b = self._procedural_rock(
                1_000_001 + index * 2,
                genetics.Sex.FEMALE,
                profile,
                self.config.seed + index * 10 + 5,
            )
            examples.append(
                self._build_example(
                    parent_a,
                    parent_b,
                    pair_index=index,
                    parent_source_type="procedural",
                    lineage_group_id=f"synthetic-lineage-{index:08d}",
                    profile=profile,
                )
            )
        return examples

    def _diverse_legal_pairs(
        self,
        source: object,
        pair_count: int,
        game: object | None = None,
    ) -> list[tuple[genetics.Rock, genetics.Rock]]:
        rocks = _extract_rocks(source)[: self.config.maximum_rocks]
        lookup = game or (source if hasattr(source, "get_rock") else _RockLookup(rocks))
        validator = breeding.BreedingMaster()
        legal = [
            pair
            for pair in itertools.combinations(rocks, 2)
            if validator.validate_breeding_pair(
                pair[0], pair[1], game=lookup, warn_relatedness=False
            )["valid"]
        ]
        legal.sort(key=lambda pair: (pair[0].value + pair[1].value, pair[0].id, pair[1].id))
        if len(legal) <= pair_count:
            return legal
        positions = np.linspace(0, len(legal) - 1, pair_count, dtype=int)
        return [legal[int(position)] for position in positions]

    def generate_from_farm(
        self,
        farm: object,
        *,
        pair_count: int | None = None,
        game: object | None = None,
    ) -> list[PredictorExample]:
        count = pair_count or self.config.number_of_parent_pairs
        pairs = self._diverse_legal_pairs(farm, count, game=game)
        return [
            self._build_example(
                parent_a,
                parent_b,
                pair_index=index,
                parent_source_type="farm",
                lineage_group_id=f"farm-pair-{self._pair_key(parent_a, parent_b)}",
            )
            for index, (parent_a, parent_b) in enumerate(pairs)
        ]

    def generate_from_historical_rocks(
        self,
        rocks: Sequence[genetics.Rock],
        *,
        pair_count: int | None = None,
    ) -> list[PredictorExample]:
        count = pair_count or self.config.number_of_parent_pairs
        pairs = self._diverse_legal_pairs(list(rocks), count)
        return [
            self._build_example(
                parent_a,
                parent_b,
                pair_index=index,
                parent_source_type="historical",
                lineage_group_id=f"historical-pair-{self._pair_key(parent_a, parent_b)}",
            )
            for index, (parent_a, parent_b) in enumerate(pairs)
        ]
