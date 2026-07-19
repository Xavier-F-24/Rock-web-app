"""Build independent NEAT populations from safe compatible champion artifacts."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import neat

from Rock_AI.neat.neat_export_helper import load_recurrent_artifact
from .training_job_config import BranchInitializationStrategy, TrainingJobConfig


@dataclass(frozen=True)
class ChampionBranchManifest:
    new_run_id: str
    parent_run_id: str
    parent_generation: int
    parent_champion_genome_id: int | str
    parent_topology_id: str
    parent_validation_metrics: dict[str, Any]
    branch_seed: int
    initialization_strategy: str
    exact_elite_count: int
    champion_descendant_count: int
    fresh_genome_count: int
    imported_historical_genome_count: int
    structural_mutation_settings: dict[str, Any]
    parametric_mutation_settings: dict[str, Any]
    creation_time: str

    def to_dict(self): return asdict(self)


def _genome_from_artifact(artifact, neat_config, genome_id: int):
    genome = neat_config.genome_type(genome_id)
    node_type = neat_config.genome_config.node_gene_type
    connection_type = neat_config.genome_config.connection_gene_type
    for row in artifact.nodes:
        gene = node_type(row.node_id)
        gene.init_attributes(neat_config.genome_config)
        gene.bias = row.bias; gene.response = row.response
        gene.activation = row.activation; gene.aggregation = row.aggregation
        genome.nodes[row.node_id] = gene
    for row in artifact.connections:
        innovation = row.innovation_id
        if innovation is None:
            innovation = neat_config.genome_config.innovation_tracker.get_innovation_number(row.source_id, row.target_id, "branch_import")
        gene = connection_type((row.source_id, row.target_id), innovation=innovation)
        gene.init_attributes(neat_config.genome_config)
        gene.weight = row.weight; gene.enabled = row.enabled
        genome.connections[(row.source_id, row.target_id)] = gene
    genome.fitness = None
    return genome


def validate_champion_compatibility(artifact, neat_config, expected_schema: int, expected_features: Sequence[str]) -> None:
    if artifact.observation_schema_version != expected_schema:
        raise ValueError("Champion observation schema is incompatible")
    if tuple(artifact.input_feature_names) != tuple(expected_features):
        raise ValueError("Champion feature schema is incompatible")
    if len(artifact.input_ids) != neat_config.genome_config.num_inputs or len(artifact.output_ids) != neat_config.genome_config.num_outputs:
        raise ValueError("Champion topology dimensions are incompatible")


def build_champion_branch_population(config: TrainingJobConfig, neat_config, *, expected_schema: int, expected_features: Sequence[str], historical_champions: Sequence[str | Path] = ()):
    artifact = load_recurrent_artifact(config.source_champion)
    validate_champion_compatibility(artifact, neat_config, expected_schema, expected_features)
    population = neat.Population(neat_config, seed=config.seed)
    fresh_pool = list(population.population.values())
    random.Random(config.seed).shuffle(fresh_pool)
    elite_count = min(config.exact_elite_count, config.population_size)
    if config.initialization_strategy in {BranchInitializationStrategy.CHAMPION_PLUS_MUTATIONS, BranchInitializationStrategy.CHAMPION_CLONES_WITH_PERTURBATION}:
        descendants = config.population_size - elite_count
        historical_target = 0
    else:
        descendants = min(config.population_size - elite_count, round(config.population_size * config.champion_descendant_fraction))
        historical_target = min(config.population_size - elite_count - descendants, round(config.population_size * config.historical_diversity_fraction))
    genomes = {}; next_key = 1
    champion = _genome_from_artifact(artifact, neat_config, next_key)
    for _ in range(elite_count):
        clone = copy.deepcopy(champion); clone.key = next_key; genomes[next_key] = clone; next_key += 1
    for _ in range(descendants):
        descendant = copy.deepcopy(champion); descendant.key = next_key
        protected_nodes = copy.deepcopy(descendant.nodes)
        protected_connections = copy.deepcopy(descendant.connections)
        for _ in range(max(1, round(config.structural_mutation_scale))):
            descendant.mutate(neat_config.genome_config)
        for _ in range(max(0, round(config.weight_mutation_scale) - 1)):
            for gene in descendant.connections.values(): gene.mutate(neat_config.genome_config)
        if config.preserve_recurrent_structure:
            for row in artifact.connections:
                key = (row.source_id, row.target_id)
                if row.recurrent and key in protected_connections:
                    descendant.connections[key] = copy.deepcopy(protected_connections[key])
        if not config.permit_simplification_mutations:
            for key, gene in protected_nodes.items(): descendant.nodes.setdefault(key, copy.deepcopy(gene))
            for key, gene in protected_connections.items(): descendant.connections.setdefault(key, copy.deepcopy(gene))
        genomes[next_key] = descendant; next_key += 1
    imported = 0
    for historical_path in historical_champions:
        if imported >= historical_target: break
        historical = load_recurrent_artifact(historical_path)
        try: validate_champion_compatibility(historical, neat_config, expected_schema, expected_features)
        except ValueError: continue
        genomes[next_key] = _genome_from_artifact(historical, neat_config, next_key); next_key += 1; imported += 1
    while len(genomes) < config.population_size:
        fresh = copy.deepcopy(fresh_pool[(len(genomes) - elite_count - descendants) % len(fresh_pool)])
        fresh.key = next_key; fresh.fitness = None; genomes[next_key] = fresh; next_key += 1
    # A branch is a new evolutionary history. Rebase innovations by structural
    # edge so champion and fresh genes cannot reuse one innovation for different edges.
    innovation_by_edge = {
        edge: index + 1
        for index, edge in enumerate(sorted({edge for genome in genomes.values() for edge in genome.connections}))
    }
    for genome in genomes.values():
        for edge, gene in genome.connections.items():
            gene.innovation = innovation_by_edge[edge]
    population.reproduction.innovation_tracker.global_counter = len(innovation_by_edge)
    population.reproduction.innovation_tracker.generation_innovations.clear()
    neat_config.genome_config.innovation_tracker = population.reproduction.innovation_tracker
    population.population = genomes
    population.species = neat_config.species_set_type(neat_config.species_set_config, population.reporters)
    population.species.speciate(neat_config, genomes, 0)
    population.generation = 0
    validation_path = Path(config.source_champion).with_name("validation_metrics.json")
    branch_manifest = ChampionBranchManifest(
        Path(config.output_run).name, Path(config.source_run).name, int(config.source_generation or 0),
        artifact.genome_id, artifact.topology_id,
        json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.exists() else {},
        config.seed, config.initialization_strategy.value, elite_count, descendants,
        config.population_size - elite_count - descendants - imported, imported,
        {"scale": config.structural_mutation_scale, "preserve_recurrent": config.preserve_recurrent_structure, "permit_simplification": config.permit_simplification_mutations},
        {"weight_mutation_scale": config.weight_mutation_scale}, datetime.now(timezone.utc).isoformat(),
    )
    return population, branch_manifest
