"""Reproducible farm-group labels built with the authoritative PairEvaluator."""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Iterable, Sequence

import numpy as np

import Rock_Breeding.rock_breeding_helper as breeding
import Rock_Genetics.rock_genetic_helper as genetics
from Rock_AI.datasets.breeding_record_helper import EncodedBreedingRules
from Rock_AI.datasets.pair_ranking_record_helper import (
    OBJECTIVE_FEATURE_NAMES,
    PAIR_METADATA_FEATURE_NAMES,
    UTILITY_COMPONENT_NAMES,
    FarmerObjectiveProfile,
    PairRankingCandidate,
    PairRankingGroup,
)
from Rock_AI.evaluation.pair_evaluator import PairEvaluator
from Rock_AI.models.pair_scoring_helper import score_pair_evaluation
from Rock_AI.representations.encoding_schema_helper import EncodingSchema, get_default_encoding_schema
from Rock_AI.representations.player_candidate_helper import candidate_arrays, candidate_model_input_hash
from Rock_AI.representations.player_observation_adapter import PlayerObservationAdapter
from Rock_AI.training.training_config_helper import PairRankingDataConfig


class SyntheticFarm:
    def __init__(self, rocks: Iterable[genetics.Rock], farm_id: str, money: float = 100.0):
        self.rocks = {int(rock.id): rock for rock in rocks}
        self.farm_id = farm_id
        self.money = money
        self.generation = max((rock.generation for rock in self.rocks.values()), default=0)

    def get_rock(self, rock_id: int) -> genetics.Rock | None:
        return self.rocks.get(int(rock_id))


def _extract_rocks(farm: object) -> list[genetics.Rock]:
    source = getattr(farm, "rocks", farm)
    values = source.values() if isinstance(source, dict) else source
    return sorted(values, key=lambda rock: (str(type(rock.id)), str(rock.id)))


class PairRankingDatasetGenerator:
    def __init__(
        self,
        config: PairRankingDataConfig,
        *,
        schema: EncodingSchema | None = None,
        pair_evaluator: PairEvaluator | None = None,
    ):
        self.config = config
        self.schema = schema or get_default_encoding_schema()
        self.pair_evaluator = pair_evaluator or PairEvaluator()
        self.rng = random.Random(config.seed)
        self.player_adapter = PlayerObservationAdapter()
        self.predictor = None
        if config.predictor_checkpoint:
            from Rock_AI.evaluation.predictor_evaluator import BreedingPredictor

            self.predictor = BreedingPredictor.load(config.predictor_checkpoint)

    def _make_rock(self, rock_id: int, sex: genetics.Sex, seed: int, generation: int) -> genetics.Rock:
        factory = genetics.GenomeFactory(rng=random.Random(seed))
        rock = genetics.Rock(
            id=rock_id,
            sex=sex,
            name=genetics.RockName(given=f"Rank{rock_id}"),
            genotype=factory.make_random_rock_genome(),
            death_genes=factory.make_death_genes(),
            generation=generation,
        )
        genetics.ExpressionEngine().instantiate_phenotype(rock)
        genetics.ValueCalculator().set_rock_value(rock)
        return rock

    def create_procedural_farm(self, farm_index: int) -> SyntheticFarm:
        count = self.rng.randint(self.config.minimum_rocks, self.config.maximum_rocks)
        base = 10_000_000 + farm_index * 100
        rocks = [
            self._make_rock(
                base + index,
                genetics.Sex.MALE if index % 2 == 0 else genetics.Sex.FEMALE,
                self.config.seed + farm_index * 1000 + index,
                self.rng.randint(0, 3),
            )
            for index in range(count)
        ]
        return SyntheticFarm(rocks, f"synthetic-farm-{farm_index:08d}", self.rng.uniform(25, 250))

    def _sample_rules(self) -> EncodedBreedingRules:
        return EncodedBreedingRules.from_config(
            {
                "mutation_chance": self.rng.uniform(*self.config.mutation_chance_range),
                "child_death_chance": self.rng.uniform(*self.config.death_chance_range),
                "craisen_chance": self.rng.uniform(0.0, 0.5),
                "clutch_mean": self.rng.uniform(1.0, 2.5),
                "clutch_std": self.rng.uniform(1.0, 2.5),
            }
        )

    def _sample_objective(self, farm_index: int) -> FarmerObjectiveProfile:
        values = [self.rng.uniform(0.0, 3.0) for _ in range(10)]
        gene = self.schema.gene_names[farm_index % len(self.schema.gene_names)]
        allele = self.rng.choice(tuple(sorted(genetics.GENE_SPECS[gene].options)))
        return FarmerObjectiveProfile(*values, preserved_gene=gene, preserved_allele=allele)

    def _predictor_vector(self, candidate) -> np.ndarray:
        if self.predictor is None:
            return np.zeros(0, dtype=np.float32)
        result = self.predictor.predict_candidate(candidate)
        values = {}
        values.update(result["scalar_predictions"])
        values.update(result["binary_probability_predictions"])
        for distribution in result["genotype_distributions"].values():
            values.update(distribution)
        for distribution in result["phenotype_distributions"].values():
            values.update(distribution)
        return np.asarray([values[name] for name in self.predictor.layout.target_names], dtype=np.float32)

    def build_group(
        self,
        farm: object,
        farm_index: int,
        *,
        rules: EncodedBreedingRules | None = None,
        objective: FarmerObjectiveProfile | None = None,
    ) -> PairRankingGroup | None:
        rocks = _extract_rocks(farm)
        lookup = farm if hasattr(farm, "get_rock") else SyntheticFarm(rocks, f"lookup-{farm_index}")
        rules = rules or self._sample_rules()
        objective = objective or self._sample_objective(farm_index)
        player_observation = self.player_adapter.build(
            farm,
            rules,
            objective,
        )
        player_candidates = {
            frozenset(map(str, candidate.canonical_parent_ids)): candidate
            for candidate in player_observation.candidates
        }
        legal = []
        validator = breeding.BreedingMaster()
        for left in range(len(rocks)):
            for right in range(left + 1, len(rocks)):
                if validator.validate_breeding_pair(
                    rocks[left], rocks[right], game=lookup, warn_relatedness=False
                )["valid"]:
                    legal.append((rocks[left], rocks[right]))
        minimum = 1 if self.config.retain_single_candidate_farms else 2
        if len(legal) < minimum:
            return None
        rows = []
        group_seed = self.config.seed + 100_000 + farm_index * 10_000
        for index, (parent_a, parent_b) in enumerate(legal):
            evaluation = self.pair_evaluator.evaluate_pair(
                parent_a,
                parent_b,
                rules=rules,
                trial_count=self.config.trials_per_pair,
                seed=group_seed + index,
                game=lookup,
            )
            scored = score_pair_evaluation(evaluation, objective)
            key = frozenset(map(str, (parent_a.id, parent_b.id)))
            player_candidate = player_candidates[key]
            arrays = candidate_arrays(player_candidate)
            predictor_features = self._predictor_vector(player_candidate)
            model_input_hash = candidate_model_input_hash(
                player_candidate,
                predictor_checkpoint_id=(
                    getattr(self.predictor, "checkpoint_path", None)
                    if self.predictor is not None else None
                ),
                predictor_feature_names=(
                    tuple(self.predictor.layout.target_names)
                    if self.predictor is not None else ()
                ),
                predictor_values=predictor_features,
            )
            rows.append(
                PairRankingCandidate(
                    parent_ids=(parent_a.id, parent_b.id),
                    parent_a_features=arrays.parent_a_features,
                    parent_b_features=arrays.parent_b_features,
                    rule_features=arrays.rule_features,
                    farm_features=arrays.farm_features,
                    objective_features=arrays.objective_features,
                    metadata_features=arrays.metadata_features,
                    predictor_features=predictor_features,
                    utility_components=np.asarray(
                        [scored.raw_components[name] for name in UTILITY_COMPONENT_NAMES], dtype=np.float32
                    ),
                    utility_score=scored.score,
                    uncertainty=scored.uncertainty,
                    metadata={
                        "evaluation_seed": group_seed + index,
                        "candidate_hash": model_input_hash,
                    },
                )
            )
        order = sorted(range(len(rows)), key=lambda i: (-rows[i].utility_score, tuple(map(str, rows[i].parent_ids))))
        ranked = []
        for candidate_index, candidate in enumerate(rows):
            rank = order.index(candidate_index) + 1
            ranked.append(replace(candidate, rank=rank, best_pair=rank == 1))
        farm_id = str(getattr(farm, "farm_id", f"farm-{farm_index:08d}"))
        return PairRankingGroup(
            group_id=f"rank-group-{self.config.seed}-{farm_index:08d}",
            lineage_group_id=farm_id,
            candidates=tuple(ranked),
            evaluation_seed=group_seed,
            monte_carlo_trial_count=self.config.trials_per_pair,
            objective_profile=objective,
            breeding_rules=rules.to_dict(),
            rock_ids=tuple(rock.id for rock in rocks),
            metadata={"farm_id": farm_id},
        )

    def generate(self, farms: Sequence[object] | None = None) -> list[PairRankingGroup]:
        sources = list(farms) if farms is not None else [
            self.create_procedural_farm(index) for index in range(self.config.number_of_farms)
        ]
        groups = [self.build_group(farm, index) for index, farm in enumerate(sources)]
        return [group for group in groups if group is not None]

    def split_groups(self, groups: Sequence[PairRankingGroup]) -> dict[str, list[PairRankingGroup]]:
        ordered = sorted(groups, key=lambda group: group.lineage_group_id)
        random.Random(self.config.seed + 77).shuffle(ordered)
        count = len(ordered)
        train_end = max(1, round(count * self.config.train_fraction))
        validation_end = train_end + max(1, round(count * self.config.validation_fraction))
        if count >= 3:
            validation_end = min(validation_end, count - 1)
            train_end = min(train_end, validation_end - 1)
        return {
            "train": ordered[:train_end],
            "validation": ordered[train_end:validation_end],
            "test": ordered[validation_end:],
        }

    def manifest(self) -> dict:
        predictor_width = 0 if self.predictor is None else len(self.predictor.layout.target_names)
        rng_state = self.rng.getstate()
        try:
            sample_farm = self.create_procedural_farm(999_999)
        finally:
            self.rng.setstate(rng_state)
        sample_rules = EncodedBreedingRules()
        sample_objective = FarmerObjectiveProfile()
        sample = self.player_adapter.build(
            sample_farm, sample_rules, sample_objective
        ).candidates[0]
        arrays = candidate_arrays(sample)
        with_masks = lambda vector: list(vector.feature_names) + [
            f"{name}.observed_mask" for name in vector.feature_names
        ]
        return {
            "encoding_schema_version": player_observation_version(),
            "observation_schema_version": player_observation_version(),
            "information_access": "player",
            "player_feature_normalizer": self.player_adapter.normalizer.to_dict(),
            "game_rules_version": self.config.game_rules_version,
            "config": self.config.to_dict(),
            "feature_names": {
                "parent": with_masks(sample.parent_a),
                "rules": with_masks(sample.public_rules),
                "farm": with_masks(sample.visible_farm),
                "objective": with_masks(sample.objective),
                "metadata": with_masks(sample.visible_pair_metadata),
                "predictor": (
                    [] if self.predictor is None else list(self.predictor.layout.target_names)
                ),
            },
            "utility_component_names": list(UTILITY_COMPONENT_NAMES),
            "dimensions": {
                "parent": len(arrays.parent_a_features),
                "rules": len(arrays.rule_features),
                "farm": len(arrays.farm_features),
                "objective": len(arrays.objective_features),
                "metadata": len(arrays.metadata_features),
                "predictor": predictor_width,
            },
        }


def player_observation_version() -> int:
    return PlayerObservationAdapter().schema.version
