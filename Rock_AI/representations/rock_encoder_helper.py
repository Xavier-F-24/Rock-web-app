"""Numerical rock encoding that retains the complete diploid genotype."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import Rock_Genetics.rock_genetic_helper as genetics
from Rock_AI.representations.encoding_schema_helper import EncodingSchema, get_default_encoding_schema


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


@dataclass(frozen=True)
class EncodedRock:
    rock_id: int | str
    continuous_features: np.ndarray
    categorical_features: np.ndarray
    genotype_features: np.ndarray
    phenotype_features: np.ndarray
    continuous_feature_names: tuple[str, ...]
    categorical_feature_names: tuple[str, ...]
    genotype_feature_names: tuple[str, ...]
    phenotype_feature_names: tuple[str, ...]
    schema_version: int

    def as_feature_vector(self) -> np.ndarray:
        return np.concatenate(
            (
                self.continuous_features,
                self.categorical_features.astype(np.float64),
                self.genotype_features,
                self.phenotype_features.astype(np.float64),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rock_id": self.rock_id,
            "continuous_features": self.continuous_features.tolist(),
            "categorical_features": self.categorical_features.tolist(),
            "genotype_features": self.genotype_features.tolist(),
            "phenotype_features": self.phenotype_features.tolist(),
            "continuous_feature_names": list(self.continuous_feature_names),
            "categorical_feature_names": list(self.categorical_feature_names),
            "genotype_feature_names": list(self.genotype_feature_names),
            "phenotype_feature_names": list(self.phenotype_feature_names),
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True)
class EncodedParentPair:
    parent_ids: tuple[int | str, int | str]
    parent_feature_matrix: np.ndarray
    feature_names: tuple[str, ...]
    schema_version: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_ids": list(self.parent_ids),
            "parent_feature_matrix": self.parent_feature_matrix.tolist(),
            "feature_names": list(self.feature_names),
            "schema_version": self.schema_version,
        }


def encode_rock(
    rock: genetics.Rock,
    schema: EncodingSchema | None = None,
    *,
    trace_id: int | str | None = None,
) -> EncodedRock:
    """Encode a rock without dropping either allele from any known gene."""

    if rock is None:
        raise ValueError("rock cannot be None")
    schema = schema or get_default_encoding_schema()

    parent_ids = list(getattr(rock, "parent_ids", None) or [])
    continuous = np.asarray(
        (
            _safe_float(getattr(rock, "generation", 0)) / schema.generation_scale,
            _safe_float(getattr(rock, "value", 0)) / schema.value_scale,
            _safe_float(getattr(rock, "sell_value", 0)) / schema.value_scale,
            _safe_float(getattr(rock, "score_value", 0)) / schema.value_scale,
            min(len(parent_ids), schema.parent_count_scale) / schema.parent_count_scale,
            float(bool(getattr(rock, "is_market", False))),
            float(bool(getattr(rock, "has_split", False))),
            float(bool(getattr(rock, "checked_craisen", False))),
        ),
        dtype=np.float64,
    )
    categorical = np.asarray(
        (
            schema.categorical_index("sex", getattr(rock, "sex", None)),
            schema.categorical_index("status", getattr(rock, "status", None)),
        ),
        dtype=np.int64,
    )

    genotype: list[float] = []
    phenotypes: list[int] = []
    genes = getattr(getattr(rock, "genotype", None), "genes", {})
    for gene_name in schema.gene_names:
        if gene_name not in genes:
            raise ValueError(f"Rock {getattr(rock, 'id', '<unknown>')} is missing gene {gene_name!r}")
        pair = genes[gene_name]
        allele_a = int(pair.allele_a.value)
        allele_b = int(pair.allele_b.value)
        homozygous = allele_a == allele_b
        genotype.extend((allele_a, allele_b, float(homozygous), float(not homozygous)))
        phenotypes.append(schema.phenotype_index(gene_name, getattr(pair, "phenotype", None)))

    death_genes = getattr(getattr(rock, "death_genes", None), "genes", {})
    for gene_name in schema.death_gene_names:
        if gene_name not in death_genes:
            raise ValueError(f"Rock {getattr(rock, 'id', '<unknown>')} is missing death gene {gene_name!r}")
        pair = death_genes[gene_name]
        allele_a = int(pair.allele_a.value)
        allele_b = int(pair.allele_b.value)
        homozygous = allele_a == allele_b
        genotype.extend((allele_a, allele_b, float(homozygous), float(not homozygous)))

    rock_id = trace_id if trace_id is not None else getattr(rock, "id", "<unknown>")
    return EncodedRock(
        rock_id=rock_id,
        continuous_features=continuous,
        categorical_features=categorical,
        genotype_features=np.asarray(genotype, dtype=np.float64),
        phenotype_features=np.asarray(phenotypes, dtype=np.int64),
        continuous_feature_names=schema.continuous_feature_names,
        categorical_feature_names=schema.categorical_feature_names,
        genotype_feature_names=schema.genotype_feature_names,
        phenotype_feature_names=schema.phenotype_feature_names,
        schema_version=schema.version,
    )


def encode_parent_pair(
    parent_a: genetics.Rock,
    parent_b: genetics.Rock,
    schema: EncodingSchema | None = None,
) -> EncodedParentPair:
    """Encode two ordered parents while retaining their source IDs."""

    schema = schema or get_default_encoding_schema()
    encoded_a = encode_rock(parent_a, schema)
    encoded_b = encode_rock(parent_b, schema)
    return EncodedParentPair(
        parent_ids=(encoded_a.rock_id, encoded_b.rock_id),
        parent_feature_matrix=np.vstack(
            (encoded_a.as_feature_vector(), encoded_b.as_feature_vector())
        ),
        feature_names=schema.rock_matrix_feature_names,
        schema_version=schema.version,
    )
