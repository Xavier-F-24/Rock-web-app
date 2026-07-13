"""Explicit, versioned feature ordering for machine-learning encodings."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Mapping

import Rock_Genetics.rock_genetic_helper as genetics


SCHEMA_VERSION = 1
SEX_VALUES = ("<missing>", "male", "female")
STATUS_VALUES = ("<missing>", "active", "sold", "dead", "craisened", "bred")
ROCK_CONTINUOUS_FEATURES = (
    "generation_normalized",
    "value_normalized",
    "sell_value_normalized",
    "score_value_normalized",
    "parent_count_normalized",
    "is_market",
    "has_split",
    "checked_craisen",
)
ROCK_CATEGORICAL_FEATURES = ("sex_index", "status_index")
FARM_GLOBAL_FEATURES = (
    "money_normalized",
    "generation_normalized",
    "rock_count_normalized",
    "active_rock_fraction",
)


@dataclass(frozen=True)
class EncodingSchema:
    """All ordered mappings required to reproduce an encoded feature vector."""

    version: int
    gene_names: tuple[str, ...]
    death_gene_names: tuple[str, ...]
    sex_values: tuple[str, ...]
    status_values: tuple[str, ...]
    phenotype_values: Mapping[str, tuple[str, ...]]
    continuous_feature_names: tuple[str, ...]
    categorical_feature_names: tuple[str, ...]
    farm_global_feature_names: tuple[str, ...]
    generation_scale: float = 20.0
    value_scale: float = 100.0
    parent_count_scale: float = 2.0
    money_scale: float = 1000.0

    @property
    def genotype_feature_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for gene_name in self.gene_names + self.death_gene_names:
            names.extend(
                (
                    f"{gene_name}.allele_a",
                    f"{gene_name}.allele_b",
                    f"{gene_name}.homozygous",
                    f"{gene_name}.heterozygous",
                )
            )
        return tuple(names)

    @property
    def phenotype_feature_names(self) -> tuple[str, ...]:
        return tuple(f"{gene_name}.phenotype_index" for gene_name in self.gene_names)

    @property
    def rock_matrix_feature_names(self) -> tuple[str, ...]:
        return (
            self.continuous_feature_names
            + self.categorical_feature_names
            + self.genotype_feature_names
            + self.phenotype_feature_names
        )

    def categorical_index(self, category: str, value: object) -> int:
        values = self.sex_values if category == "sex" else self.status_values
        normalized = getattr(value, "value", value)
        try:
            return values.index(str(normalized))
        except (ValueError, TypeError):
            return 0

    def phenotype_index(self, gene_name: str, value: object) -> int:
        values = self.phenotype_values[gene_name]
        normalized = "<missing>" if value is None else str(value)
        try:
            return values.index(normalized)
        except ValueError:
            return 0


def _phenotype_values_for_gene(gene_name: str) -> tuple[str, ...]:
    spec = genetics.GENE_SPECS[gene_name]
    values = {"<missing>", "n/a"}
    values.update(option.name for option in spec.options.values())
    values.update(state.name for state in spec.states.values())
    values.update(state.name for state in spec.special_states.values())
    if spec.required_gender_states is not None:
        values.add(str(spec.required_gender_states))
    return ("<missing>",) + tuple(sorted(values - {"<missing>"}))


@lru_cache(maxsize=1)
def get_default_encoding_schema() -> EncodingSchema:
    """Return the immutable schema used by version-one training records."""

    gene_names = tuple(sorted(genetics.GENE_SPECS))
    phenotype_values = MappingProxyType(
        {gene_name: _phenotype_values_for_gene(gene_name) for gene_name in gene_names}
    )
    return EncodingSchema(
        version=SCHEMA_VERSION,
        gene_names=gene_names,
        death_gene_names=tuple(sorted(genetics.GenomeFactory.death_gene_list)),
        sex_values=SEX_VALUES,
        status_values=STATUS_VALUES,
        phenotype_values=phenotype_values,
        continuous_feature_names=ROCK_CONTINUOUS_FEATURES,
        categorical_feature_names=ROCK_CATEGORICAL_FEATURES,
        farm_global_feature_names=FARM_GLOBAL_FEATURES,
    )
