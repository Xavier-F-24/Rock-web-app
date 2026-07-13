"""Dataset-derived target grouping and structured model outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class DistributionGroup:
    name: str
    target_indices: tuple[int, ...]
    target_names: tuple[str, ...]


@dataclass(frozen=True)
class TargetLayout:
    target_names: tuple[str, ...]
    scalar_indices: tuple[int, ...]
    binary_probability_indices: tuple[int, ...]
    genotype_groups: tuple[DistributionGroup, ...]
    phenotype_groups: tuple[DistributionGroup, ...]

    @classmethod
    def from_target_names(cls, target_names: list[str] | tuple[str, ...]) -> "TargetLayout":
        names = tuple(str(name) for name in target_names)
        genotype: dict[str, list[int]] = {}
        phenotype: dict[str, list[int]] = {}
        binary: list[int] = []
        scalar: list[int] = []
        for index, name in enumerate(names):
            if name.startswith("gene.") and ".allele_pair." in name:
                gene_name = name.split(".", maxsplit=2)[1]
                genotype.setdefault(gene_name, []).append(index)
            elif name.startswith("phenotype.") and "=" in name:
                gene_name = name.removeprefix("phenotype.").split("=", maxsplit=1)[0]
                phenotype.setdefault(gene_name, []).append(index)
            elif name.startswith("probability_"):
                binary.append(index)
            else:
                scalar.append(index)

        def groups(values: dict[str, list[int]]) -> tuple[DistributionGroup, ...]:
            return tuple(
                DistributionGroup(
                    name=group_name,
                    target_indices=tuple(indices),
                    target_names=tuple(names[index] for index in indices),
                )
                for group_name, indices in sorted(values.items())
            )

        return cls(
            target_names=names,
            scalar_indices=tuple(scalar),
            binary_probability_indices=tuple(binary),
            genotype_groups=groups(genotype),
            phenotype_groups=groups(phenotype),
        )

    @property
    def scalar_names(self) -> tuple[str, ...]:
        return tuple(self.target_names[index] for index in self.scalar_indices)

    @property
    def binary_probability_names(self) -> tuple[str, ...]:
        return tuple(self.target_names[index] for index in self.binary_probability_indices)

    @property
    def genotype_output_dimension(self) -> int:
        return sum(len(group.target_indices) for group in self.genotype_groups)

    @property
    def phenotype_output_dimension(self) -> int:
        return sum(len(group.target_indices) for group in self.phenotype_groups)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BreedingPredictorOutput:
    scalar_normalized: torch.Tensor
    binary_logits: torch.Tensor
    binary_probabilities: torch.Tensor
    genotype_logits: tuple[torch.Tensor, ...]
    genotype_probabilities: tuple[torch.Tensor, ...]
    phenotype_logits: tuple[torch.Tensor, ...]
    phenotype_probabilities: tuple[torch.Tensor, ...]

    def probability_vector(self) -> torch.Tensor:
        pieces = [self.binary_probabilities]
        pieces.extend(self.genotype_probabilities)
        pieces.extend(self.phenotype_probabilities)
        return torch.cat(pieces, dim=-1) if pieces else self.scalar_normalized.new_zeros(
            (self.scalar_normalized.shape[0], 0)
        )
