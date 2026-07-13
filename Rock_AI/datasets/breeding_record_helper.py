"""Typed, JSON-safe outputs from one authoritative breeding event."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import Rock_Breeding.rock_breeding_helper as breeding


@dataclass(frozen=True)
class EncodedBreedingRules:
    mutation_chance: float = breeding.MUTATION_CHANCE
    child_death_chance: float = breeding.CHILD_DEATH_CHANCE
    craisen_chance: float = breeding.CRAISEN_DEATH_CHANCE
    clutch_mean: float = breeding.CLUTCH_MEAN
    clutch_std: float = breeding.CLUTCH_STD
    max_clutch_size: int | None = breeding.MAX_CLUTCH_SIZE
    spore_chance: float | None = None
    spore_death_chance: float = breeding.SPORE_DEATH_CHANCE
    spore_clone_count: int = breeding.SPORE_CLONE_COUNT
    mitosion_chance: float | None = None
    clutch_reroll: bool = False
    clutch_plus_one: bool = False
    require_opposite_gender: bool = True

    @classmethod
    def from_config(
        cls,
        config: EncodedBreedingRules | Mapping[str, Any] | None = None,
        *,
        master: breeding.BreedingMaster | None = None,
    ) -> EncodedBreedingRules:
        if isinstance(config, cls):
            return config
        defaults = cls()
        values = asdict(defaults)
        if master is not None:
            values.update(
                mutation_chance=float(master.child_gene_mutation_chance),
                child_death_chance=float(master.child_death_chance),
                craisen_chance=float(master.craisen_death_chance),
                clutch_mean=float(master.clutch_mean),
                clutch_std=float(master.clutch_std),
                max_clutch_size=master.max_clutch_size,
                spore_death_chance=float(master.spore_death_chance),
                spore_clone_count=int(master.spore_clone_count),
            )
        if config:
            unknown = set(config) - set(values)
            if unknown:
                raise ValueError(f"Unknown breeding rule(s): {', '.join(sorted(unknown))}")
            values.update(config)
        result = cls(**values)
        result.validate()
        return result

    def validate(self) -> None:
        probabilities = (
            "mutation_chance",
            "child_death_chance",
            "craisen_chance",
            "spore_death_chance",
        )
        for name in probabilities:
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        for name in ("spore_chance", "mitosion_chance"):
            value = getattr(self, name)
            if value is not None and not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be None or between 0 and 1")
        if self.clutch_std < 0:
            raise ValueError("clutch_std cannot be negative")
        if self.max_clutch_size is not None and self.max_clutch_size < 1:
            raise ValueError("max_clutch_size must be positive or None")
        if self.spore_clone_count < 0:
            raise ValueError("spore_clone_count cannot be negative")

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(asdict(self))

    @property
    def feature_values(self) -> tuple[float, ...]:
        values: list[float] = []
        for value in asdict(self).values():
            values.append(-1.0 if value is None else float(value))
        return tuple(values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BreedingRecord:
    random_seed: int
    parent_ids: tuple[int | str, int | str]
    encoded_parent_data: tuple[dict[str, Any], dict[str, Any]]
    encoded_rule_data: dict[str, Any]
    clutch_size: int
    child_ids: tuple[int | str, ...]
    child_genotypes: tuple[dict[str, list[int]], ...]
    child_death_genotypes: tuple[dict[str, list[int]], ...]
    child_phenotypes: tuple[dict[str, str | None], ...]
    mutation_information: tuple[dict[str, Any], ...]
    child_statuses: tuple[str, ...]
    survivor_count: int
    child_values: tuple[int, ...]
    summary_statistics: dict[str, float | int] = field(default_factory=dict)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
