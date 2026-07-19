"""Project-owned structured progress events for NEAT workers."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import neat
import numpy as np


class NeatProgressReporter(neat.reporting.BaseReporter):
    def __init__(self, progress_path: str | Path, event_callback=None):
        self.path = Path(progress_path); self.event_callback = event_callback
        self.run_started = time.monotonic(); self.generation_started = self.run_started; self.generation = 0

    def _write(self, event_type: str, **payload):
        row = {"event_type": event_type, "timestamp": datetime.now(timezone.utc).isoformat(), **payload}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n"); stream.flush()
        if self.event_callback: self.event_callback(row)

    def start_generation(self, generation):
        self.generation = int(generation); self.generation_started = time.monotonic(); self._write("generation_started", generation=self.generation)

    def post_evaluate(self, config, population, species, best_genome):
        fitness = [float(genome.fitness) for genome in population.values() if genome.fitness is not None]
        node_counts = [len(genome.nodes) for genome in population.values()]
        connection_counts = [sum(gene.enabled for gene in genome.connections.values()) for genome in population.values()]
        self._write(
            "generation_evaluated", generation=self.generation, genomes_evaluated=len(population),
            species_count=len(species.species), best_fitness=max(fitness, default=None),
            mean_fitness=float(np.mean(fitness)) if fitness else None, median_fitness=float(np.median(fitness)) if fitness else None,
            mean_node_count=float(np.mean(node_counts)), mean_connection_count=float(np.mean(connection_counts)),
            best_genome_node_count=len(best_genome.nodes), best_genome_connection_count=sum(gene.enabled for gene in best_genome.connections.values()),
            elapsed_generation_seconds=time.monotonic() - self.generation_started,
            total_elapsed_seconds=time.monotonic() - self.run_started,
        )
