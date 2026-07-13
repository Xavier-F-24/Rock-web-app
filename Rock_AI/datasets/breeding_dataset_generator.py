"""Reproducible generation and export of headless breeding records."""

from __future__ import annotations

import itertools
import json
import random
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import Rock_Breeding.rock_breeding_helper as breeding
import Rock_Genetics.rock_genetic_helper as genetics
from Rock_AI.datasets.breeding_record_helper import BreedingRecord, EncodedBreedingRules
from Rock_AI.environments.breeding_training_environment import BreedingTrainingEnvironment


def _rocks_from_farm(farm: object) -> list[genetics.Rock]:
    source = getattr(farm, "rocks", None)
    if source is None:
        source = getattr(farm, "rock_list", None)
    if source is None and isinstance(farm, (dict, list, tuple)):
        source = farm
    if source is None:
        raise TypeError("farm must expose rocks or rock_list, or be a rock collection")
    values = source.values() if isinstance(source, Mapping) else source
    return sorted(values, key=lambda rock: (0, int(rock.id)) if isinstance(rock.id, int) else (1, str(rock.id)))


class _FarmLookup:
    def __init__(self, rocks: Iterable[genetics.Rock]):
        self.rocks = {int(rock.id): rock for rock in rocks}

    def get_rock(self, rock_id: int) -> genetics.Rock | None:
        return self.rocks.get(int(rock_id))


class BreedingDatasetGenerator:
    def __init__(
        self,
        seed: int = 0,
        *,
        rules: EncodedBreedingRules | Mapping[str, object] | None = None,
    ):
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.rules = EncodedBreedingRules.from_config(rules)

    def generate_from_parent_pairs(
        self,
        parent_pairs: Iterable[tuple[genetics.Rock, genetics.Rock]],
        *,
        trials_per_pair: int = 1,
        start_seed: int | None = None,
        rules: EncodedBreedingRules | Mapping[str, object] | None = None,
    ) -> list[BreedingRecord]:
        if trials_per_pair < 0:
            raise ValueError("trials_per_pair cannot be negative")
        next_seed = self.seed if start_seed is None else int(start_seed)
        active_rules = EncodedBreedingRules.from_config(rules or self.rules)
        records: list[BreedingRecord] = []
        environment = BreedingTrainingEnvironment(seed=next_seed, rules=active_rules)
        for parent_a, parent_b in parent_pairs:
            for _ in range(trials_per_pair):
                records.append(
                    environment.execute_breeding(
                        parent_a,
                        parent_b,
                        rules=active_rules,
                        seed=next_seed,
                    )
                )
                next_seed += 1
        return records

    def sample_valid_parent_pairs(
        self,
        farm: object,
        *,
        count: int | None = None,
        seed: int | None = None,
        game: object | None = None,
    ) -> list[tuple[genetics.Rock, genetics.Rock]]:
        if count is not None and count < 0:
            raise ValueError("count cannot be negative")
        rocks = _rocks_from_farm(farm)
        validator = breeding.BreedingMaster()
        lookup = game or _FarmLookup(rocks)
        pairs = [
            pair
            for pair in itertools.combinations(rocks, 2)
            if validator.validate_breeding_pair(
                pair[0], pair[1], game=lookup, warn_relatedness=False
            )["valid"]
        ]
        sampler = random.Random(self.seed if seed is None else int(seed))
        sampler.shuffle(pairs)
        return pairs if count is None else pairs[:count]

    def generate_from_farm(
        self,
        farm: object,
        *,
        pair_count: int | None = None,
        trials_per_pair: int = 1,
        start_seed: int | None = None,
        sample_seed: int | None = None,
        game: object | None = None,
    ) -> list[BreedingRecord]:
        pairs = self.sample_valid_parent_pairs(
            farm,
            count=pair_count,
            seed=sample_seed,
            game=game,
        )
        return self.generate_from_parent_pairs(
            pairs,
            trials_per_pair=trials_per_pair,
            start_seed=start_seed,
        )

    @staticmethod
    def write_jsonl(records: Sequence[BreedingRecord], path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":")))
                handle.write("\n")
        return output_path
