"""Farm-level and paired metrics for complete breeding campaigns."""

from __future__ import annotations

from collections import Counter
from statistics import mean

import Rock_Genetics.rock_genetic_helper as genetics
from Rock_AI.datasets.pair_ranking_record_helper import FarmerObjectiveProfile


def _genotype_signature(rock) -> tuple:
    return tuple(
        (name, int(pair.allele_a.value), int(pair.allele_b.value))
        for name, pair in sorted(rock.genotype.genes.items())
    )


def _phenotype_signature(rock) -> tuple:
    return tuple((name, pair.phenotype) for name, pair in sorted(rock.genotype.genes.items()))


def _rare_alleles() -> set[tuple[str, int]]:
    rare = set()
    for gene_name, spec in genetics.GENE_SPECS.items():
        counts = Counter(
            genetics.GenomeFactory.get_allele_from_roll(roll, spec).value
            for roll in range(1, 21)
        )
        rare.update((gene_name, allele) for allele, count in counts.items() if count / 20 < 0.10)
    return rare


RARE_ALLELES = _rare_alleles()


def calculate_farm_metrics(game: object) -> dict[str, float | int]:
    source = getattr(game, "rocks", {})
    rocks = list(source.values() if isinstance(source, dict) else source)
    active = [rock for rock in rocks if rock.status == genetics.RockStatus.ACTIVE]
    offspring = [rock for rock in rocks if getattr(rock, "parent_ids", None)]
    surviving_offspring = [rock for rock in offspring if rock.status == genetics.RockStatus.ACTIVE]
    values = [float(rock.value) for rock in rocks]
    active_values = [float(rock.value) for rock in active]
    genotype_diversity = len({_genotype_signature(rock) for rock in active}) / len(active) if active else 0.0
    phenotype_diversity = len({_phenotype_signature(rock) for rock in active}) / len(active) if active else 0.0
    rare_trait_count = sum(
        (gene_name, int(allele.value)) in RARE_ALLELES
        for rock in active
        for gene_name, pair in rock.genotype.genes.items()
        for allele in pair.alleles
    )
    return {
        "rock_count": len(rocks),
        "active_rock_count": len(active),
        "final_total_farm_value": float(sum(values)),
        "final_active_rock_value": float(sum(active_values)),
        "final_maximum_rock_value": float(max(values, default=0.0)),
        "average_rock_value": float(mean(values)) if values else 0.0,
        "surviving_offspring": len(surviving_offspring),
        "genotype_diversity": float(genotype_diversity),
        "phenotype_diversity": float(phenotype_diversity),
        "rare_trait_count": int(rare_trait_count),
        "generation": int(getattr(game, "generation", 0)),
    }


def calculate_final_objective_utility(
    metrics: dict[str, float | int],
    objective: FarmerObjectiveProfile,
    *,
    mutation_count: int = 0,
) -> float:
    return float(
        metrics["final_active_rock_value"] * objective.immediate_expected_value_weight
        + metrics["final_maximum_rock_value"] * objective.maximum_offspring_value_weight
        + metrics["surviving_offspring"] * objective.survivor_count_weight
        + metrics["genotype_diversity"] * objective.genotype_diversity_weight
        + metrics["phenotype_diversity"] * objective.phenotype_diversity_weight
        + metrics["rare_trait_count"] * objective.rare_trait_weight
        + mutation_count * objective.mutation_opportunity_weight
    )
