"""One-step and Monte Carlo breeding simulations using BreedingMaster."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from statistics import mean
from typing import Any, Mapping

import Rock_Breeding.rock_breeding_helper as breeding
import Rock_Genetics.rock_genetic_helper as genetics
from Rock_GameState.rock_game_state_helper import GameMaster
from Rock_AI.datasets.breeding_record_helper import BreedingRecord, EncodedBreedingRules
from Rock_AI.environments.rock_training_environment import RockTrainingEnvironment
from Rock_AI.representations.encoding_schema_helper import EncodingSchema, get_default_encoding_schema
from Rock_AI.representations.rock_encoder_helper import encode_rock


@dataclass
class _BreedingEpisodeState:
    parent_a: genetics.Rock | None = None
    parent_b: genetics.Rock | None = None
    rules: EncodedBreedingRules | None = None
    last_record: BreedingRecord | None = None


class _RecordingBreedingMaster(breeding.BreedingMaster):
    """Observe the authoritative clutch roll without replacing its logic."""

    last_base_clutch_size: int | None = None

    def roll_clutch_size(self, *args, **kwargs):
        result = super().roll_clutch_size(*args, **kwargs)
        self.last_base_clutch_size = int(result)
        return result


class BreedingTrainingEnvironment(RockTrainingEnvironment):
    def __init__(
        self,
        seed: int = 0,
        *,
        schema: EncodingSchema | None = None,
        rules: EncodedBreedingRules | Mapping[str, Any] | None = None,
    ):
        super().__init__(seed=seed)
        self.schema = schema or get_default_encoding_schema()
        self.default_rules = EncodedBreedingRules.from_config(rules)
        self.state = _BreedingEpisodeState(rules=self.default_rules)
        self.last_children: list[genetics.Rock] = []

    def reset(self, seed: int | None = None) -> _BreedingEpisodeState:
        super().reset(seed)
        self.state = _BreedingEpisodeState(rules=self.default_rules)
        self.last_children = []
        return self.state

    def load_parents(self, parent_a: genetics.Rock, parent_b: genetics.Rock) -> None:
        if parent_a is None or parent_b is None:
            raise ValueError("Both parents are required")
        self.state.parent_a = copy.deepcopy(parent_a)
        self.state.parent_b = copy.deepcopy(parent_b)

    def generate_parents(self, seed: int | None = None) -> tuple[genetics.Rock, genetics.Rock]:
        parent_seed = self.seed if seed is None else int(seed)
        game = GameMaster(seed=parent_seed)
        males = sorted(
            (rock for rock in game.rocks.values() if rock.sex == genetics.Sex.MALE),
            key=lambda rock: rock.id,
        )
        females = sorted(
            (rock for rock in game.rocks.values() if rock.sex == genetics.Sex.FEMALE),
            key=lambda rock: rock.id,
        )
        if not males or not females:
            raise RuntimeError("Game startup did not create an opposite-sex parent pair")
        self.load_parents(males[0], females[0])
        return copy.deepcopy(self.state.parent_a), copy.deepcopy(self.state.parent_b)

    @staticmethod
    def _seeded_master(seed: int, rules: EncodedBreedingRules) -> _RecordingBreedingMaster:
        master = _RecordingBreedingMaster()
        master.rng = random.Random(seed)
        master.GenomeFactory.rng = random.Random(seed + 1)
        master.NameGenerator.rng = random.Random(seed + 2)
        master.clutch_mean = rules.clutch_mean
        master.clutch_std = rules.clutch_std
        master.max_clutch_size = rules.max_clutch_size
        return master

    @staticmethod
    def _genotype(rock: genetics.Rock, attribute: str = "genotype") -> dict[str, list[int]]:
        genome = getattr(rock, attribute)
        return {
            gene_name: [int(pair.allele_a.value), int(pair.allele_b.value)]
            for gene_name, pair in sorted(genome.genes.items())
        }

    @staticmethod
    def _phenotypes(rock: genetics.Rock) -> dict[str, str | None]:
        return {
            gene_name: pair.phenotype
            for gene_name, pair in sorted(rock.genotype.genes.items())
        }

    @classmethod
    def _mutation_details(
        cls,
        child: genetics.Rock,
        parent_a: genetics.Rock,
        parent_b: genetics.Rock,
    ) -> dict[str, Any]:
        standard: dict[str, list[bool]] = {}
        for gene_name, pair in sorted(child.genotype.genes.items()):
            parent_a_values = {allele.value for allele in parent_a.genotype.genes[gene_name].alleles}
            parent_b_values = {allele.value for allele in parent_b.genotype.genes[gene_name].alleles}
            standard[gene_name] = [
                pair.allele_a.value not in parent_a_values,
                pair.allele_b.value not in parent_b_values,
            ]
        death: dict[str, list[bool]] = {}
        for gene_name, pair in sorted(child.death_genes.genes.items()):
            parent_a_values = {allele.value for allele in parent_a.death_genes.genes[gene_name].alleles}
            parent_b_values = {allele.value for allele in parent_b.death_genes.genes[gene_name].alleles}
            death[gene_name] = [
                pair.allele_a.value not in parent_a_values,
                pair.allele_b.value not in parent_b_values,
            ]
        mutation_count = sum(sum(flags) for flags in standard.values()) + sum(
            sum(flags) for flags in death.values()
        )
        return {
            "child_id": child.id,
            "detection_method": "offspring_allele_not_present_in_source_parent",
            "gene_allele_mutations": standard,
            "death_gene_allele_mutations": death,
            "mutation_count": mutation_count,
        }

    def execute_breeding(
        self,
        parent_a: genetics.Rock | None = None,
        parent_b: genetics.Rock | None = None,
        *,
        rules: EncodedBreedingRules | Mapping[str, Any] | None = None,
        seed: int | None = None,
        next_id: int | None = None,
        child_generation: int | None = None,
    ) -> BreedingRecord:
        trial_seed = self.seed if seed is None else int(seed)
        if parent_a is not None or parent_b is not None:
            if parent_a is None or parent_b is None:
                raise ValueError("Both parents must be supplied together")
            self.load_parents(parent_a, parent_b)
        if self.state.parent_a is None or self.state.parent_b is None:
            self.generate_parents(seed=trial_seed)

        source_a = self.state.parent_a
        source_b = self.state.parent_b
        assert source_a is not None and source_b is not None
        trial_a, trial_b = copy.deepcopy(source_a), copy.deepcopy(source_b)
        trial_rules = EncodedBreedingRules.from_config(rules or self.default_rules)
        if not trial_rules.require_opposite_gender:
            raise ValueError(
                "The authoritative breed_parent_set currently requires opposite-sex parents; "
                "require_opposite_gender=False is not supported by this environment."
            )
        if trial_rules.spore_chance is not None or trial_rules.mitosion_chance is not None:
            raise ValueError(
                "Spore and mitosion activation are genotype-controlled in the authoritative engine; "
                "their chance fields must remain None."
            )
        master = self._seeded_master(trial_seed, trial_rules)
        validation = master.validate_breeding_pair(
            trial_a,
            trial_b,
            require_opposite_gender=trial_rules.require_opposite_gender,
            warn_relatedness=False,
        )
        if not validation["valid"]:
            raise ValueError("Invalid breeding pair: " + "; ".join(validation["errors"]))

        parent_encodings = (
            encode_rock(trial_a, self.schema).to_dict(),
            encode_rock(trial_b, self.schema).to_dict(),
        )
        first_child_id = next_id if next_id is not None else max(int(trial_a.id), int(trial_b.id)) + 1
        generation = child_generation if child_generation is not None else max(trial_a.generation, trial_b.generation) + 1
        children = master.breed_parent_set(
            parent_a=trial_a,
            parent_b=trial_b,
            next_id=first_child_id,
            child_generation=generation,
            mutation_chance=trial_rules.mutation_chance,
            death_chance=trial_rules.child_death_chance,
            craisen_chance=trial_rules.craisen_chance,
            spore_death_chance=trial_rules.spore_death_chance,
            spore_clone_count=trial_rules.spore_clone_count,
            clutch_reroll=trial_rules.clutch_reroll,
            clutch_plus_one=trial_rules.clutch_plus_one,
        )
        self.last_children = copy.deepcopy(children)

        statuses = tuple(child.status.value for child in children)
        values = tuple(int(child.value) for child in children)
        survivors = sum(child.status == genetics.RockStatus.ACTIVE for child in children)
        mutations = tuple(self._mutation_details(child, source_a, source_b) for child in children)
        mutation_total = sum(item["mutation_count"] for item in mutations)
        summary: dict[str, float | int] = {
            "returned_child_count": len(children),
            "survivor_fraction": survivors / len(children) if children else 0.0,
            "mean_child_value": mean(values) if values else 0.0,
            "min_child_value": min(values) if values else 0,
            "max_child_value": max(values) if values else 0,
            "mutation_count": mutation_total,
        }
        record = BreedingRecord(
            random_seed=trial_seed,
            parent_ids=(source_a.id, source_b.id),
            encoded_parent_data=parent_encodings,
            encoded_rule_data=trial_rules.to_dict(),
            clutch_size=master.last_base_clutch_size or 0,
            child_ids=tuple(child.id for child in children),
            child_genotypes=tuple(self._genotype(child) for child in children),
            child_death_genotypes=tuple(self._genotype(child, "death_genes") for child in children),
            child_phenotypes=tuple(self._phenotypes(child) for child in children),
            mutation_information=mutations,
            child_statuses=statuses,
            survivor_count=survivors,
            child_values=values,
            summary_statistics=summary,
            schema_version=self.schema.version,
        )
        self.state.rules = trial_rules
        self.state.last_record = record
        return record

    def repeat_pair(
        self,
        parent_a: genetics.Rock,
        parent_b: genetics.Rock,
        trials: int,
        *,
        start_seed: int | None = None,
        rules: EncodedBreedingRules | Mapping[str, Any] | None = None,
    ) -> list[BreedingRecord]:
        if trials < 0:
            raise ValueError("trials cannot be negative")
        first_seed = self.seed if start_seed is None else int(start_seed)
        return [
            self.execute_breeding(parent_a, parent_b, rules=rules, seed=first_seed + offset)
            for offset in range(trials)
        ]

    @staticmethod
    def aggregate_records(records: list[BreedingRecord]) -> dict[str, float | int]:
        if not records:
            return {"trial_count": 0, "mean_clutch_size": 0.0, "mean_survivors": 0.0, "mean_child_value": 0.0}
        all_values = [value for record in records for value in record.child_values]
        return {
            "trial_count": len(records),
            "mean_clutch_size": mean(record.clutch_size for record in records),
            "mean_survivors": mean(record.survivor_count for record in records),
            "mean_child_value": mean(all_values) if all_values else 0.0,
        }
