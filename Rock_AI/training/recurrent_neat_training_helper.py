"""Deterministic open-topology recurrent NEAT training for player-visible pair data."""

from __future__ import annotations

import gzip
import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import neat
import numpy as np

from Rock_AI.datasets.pair_ranking_storage_helper import load_pair_ranking_split
from Rock_AI.datasets.pair_ranking_dataset_generator import PairRankingDataConfig, PairRankingDatasetGenerator
from Rock_AI.agents.oracle_breeding_agent import OracleBreedingAgent
from Rock_AI.agents.random_breeding_agent import RandomBreedingAgent
from Rock_AI.agents.recurrent_neat_breeding_agent import RecurrentNeatBreedingAgent
from Rock_AI.environments.breeding_campaign_environment import BreedingCampaignConfig, BreedingCampaignEnvironment
from Rock_AI.evaluation.breeding_agent_evaluator import BreedingAgentEvaluator
from Rock_AI.neat.neat_export_helper import export_recurrent_genome, save_recurrent_artifact
from Rock_AI.neat.neat_genome_helper import BoundedRecurrentGenome
from Rock_AI.neat.neat_recurrent_network import RecurrentEvaluationConfig, RecurrentNeatNetwork
from Rock_AI.neat.neat_state_helper import TEMPORAL_FEATURE_NAMES
from Rock_AI.neat.neat_topology_helper import TopologyResourceLimits
from Rock_AI.policies.recurrent_neat_pair_ranking_policy import RecurrentNeatPairRankingPolicy
from Rock_Serialization.rock_serialization_helper import game_to_dict
from Rock_AI.training.neat_training_helper import _candidate_feature_names, _candidate_matrix
from Rock_AI.training_jobs.worker_heartbeat import BackgroundHeartbeat, HeartbeatPhase, TimedHeartbeat


@dataclass(frozen=True)
class RecurrentNeatTrainingConfig:
    dataset_path: str
    output_directory: str
    seed: int = 1234
    population: int = 100
    generations: int = 50
    training_scenarios_per_generation: int = 24
    validation_scenarios: int = 24
    checkpoint_frequency: int = 5
    settling_steps: int = 3
    complexity_penalty: float = 0.00001
    minimum_campaign_generation: int = 5
    validation_quality_threshold: float = 0.55
    stability_window: int = 3
    campaign_scenarios_per_generation: int = 1
    campaign_generations: int = 3
    oracle_trial_count: int = 25
    supervised_weight: float = 0.60
    campaign_weight: float = 0.40
    max_hidden_nodes: int = 128
    max_enabled_connections: int = 4096
    max_total_genes: int = 8192
    heartbeat_interval_seconds: float = 5.0

    def __post_init__(self) -> None:
        if min(self.population, self.generations, self.training_scenarios_per_generation) <= 0:
            raise ValueError("Population, generations, and scenario count must be positive")
        if self.settling_steps <= 0:
            raise ValueError("settling_steps must be positive")
        if self.heartbeat_interval_seconds <= 0:
            raise ValueError("heartbeat_interval_seconds must be positive")
        if abs(self.supervised_weight + self.campaign_weight - 1.0) > 1e-9:
            raise ValueError("Supervised and campaign weights must sum to one")


def _write_config(path: Path, inputs: int, population: int) -> None:
    path.write_text(f"""[NEAT]
fitness_criterion = max
fitness_threshold = 1.0
pop_size = {population}
reset_on_extinction = False
no_fitness_termination = True

[BoundedRecurrentGenome]
activation_default = tanh
activation_mutate_rate = 0.05
activation_options = tanh sigmoid relu identity sin gauss
aggregation_default = sum
aggregation_mutate_rate = 0.03
aggregation_options = sum mean max min product
bias_init_mean = 0.0
bias_init_stdev = 1.0
bias_max_value = 30.0
bias_min_value = -30.0
bias_mutate_power = 0.5
bias_mutate_rate = 0.7
bias_replace_rate = 0.1
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient = 0.5
conn_add_prob = 0.55
conn_delete_prob = 0.25
enabled_default = True
enabled_mutate_rate = 0.03
feed_forward = False
initial_connection = partial_direct 0.10
node_add_prob = 0.25
node_delete_prob = 0.10
num_hidden = 0
num_inputs = {inputs}
num_outputs = 3
response_init_mean = 1.0
response_init_stdev = 0.0
response_max_value = 30.0
response_min_value = -30.0
response_mutate_power = 0.1
response_mutate_rate = 0.05
response_replace_rate = 0.0
weight_init_mean = 0.0
weight_init_stdev = 1.0
weight_max_value = 30
weight_min_value = -30
weight_mutate_power = 0.5
weight_mutate_rate = 0.8
weight_replace_rate = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation = 15
species_elitism = 2

[DefaultReproduction]
elitism = 2
survival_threshold = 0.2
min_species_size = 2
""", encoding="utf-8")


class _SpeciesReporter(neat.reporting.BaseReporter):
    def __init__(self, path: Path):
        self.path = path
        self.generation = 0

    def post_evaluate(self, config, population, species, best_genome):
        sizes = {str(key): len(value.members) for key, value in species.species.items()}
        row = {"generation": self.generation, "species_count": len(sizes), "species_sizes": sizes}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        self.generation += 1


class _HeartbeatCheckpointer(neat.Checkpointer):
    def __init__(self, *args, heartbeat=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.heartbeat = heartbeat

    def save_checkpoint(self, config, population, species_set, generation):
        if self.heartbeat:
            self.heartbeat(
                HeartbeatPhase.CHECKPOINT_WRITING, force=True,
                operation="checkpoint_started", progress_current=0,
                progress_total=1, progress_label="Checkpoint writing",
            )
        result = super().save_checkpoint(config, population, species_set, generation)
        if self.heartbeat:
            self.heartbeat(
                HeartbeatPhase.CHECKPOINT_WRITING, force=True,
                operation="checkpoint_completed", progress_current=1,
                progress_total=1, progress_label="Checkpoint writing",
            )
        return result

    def __getstate__(self):
        state = dict(self.__dict__)
        state["heartbeat"] = None
        return state


class TrainingCancelled(RuntimeError):
    pass


class RecurrentNeatTrainer:
    def __init__(self, training_config: RecurrentNeatTrainingConfig, *, allow_existing: bool = False, starting_generation: int = 0, progress_callback=None, cancel_path: str | Path | None = None):
        self.training_config = training_config
        self.output = Path(training_config.output_directory)
        if self.output.exists() and any(self.output.iterdir()) and not allow_existing:
            raise FileExistsError(f"Training run already exists: {self.output}")
        self.output.mkdir(parents=True, exist_ok=True)
        self.train_arrays, self.train_groups, self.manifest = load_pair_ranking_split(training_config.dataset_path, "train")
        self.validation_arrays, self.validation_groups, validation_manifest = load_pair_ranking_split(training_config.dataset_path, "validation")
        if self.manifest.get("information_access") != "player":
            raise ValueError("Recurrent gameplay training requires player-visible data")
        if validation_manifest.get("observation_schema_version") != self.manifest.get("observation_schema_version"):
            raise ValueError("Training and validation observation schemas differ")
        temporal_width = len(TEMPORAL_FEATURE_NAMES) * 2
        self.train_matrix = np.pad(_candidate_matrix(self.train_arrays), ((0, 0), (0, temporal_width)))
        self.validation_matrix = np.pad(_candidate_matrix(self.validation_arrays), ((0, 0), (0, temporal_width)))
        base_names = _candidate_feature_names(self.manifest)
        self.feature_names = base_names + TEMPORAL_FEATURE_NAMES + tuple(f"{name}.visible" for name in TEMPORAL_FEATURE_NAMES)
        self.limits = TopologyResourceLimits(
            max_hidden_nodes=training_config.max_hidden_nodes,
            max_enabled_connections=training_config.max_enabled_connections,
            max_total_genes=training_config.max_total_genes,
        )
        BoundedRecurrentGenome.resource_limits = self.limits
        self.neat_config_path = self.output / "neat_config.ini"
        if not self.neat_config_path.exists():
            _write_config(self.neat_config_path, self.train_matrix.shape[1], training_config.population)
        self.neat_config = neat.Config(
            BoundedRecurrentGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
            neat.DefaultStagnation, str(self.neat_config_path),
        )
        self.generation = int(starting_generation)
        self.validation_history: list[float] = []
        self.progress_callback = progress_callback
        self.cancel_path = Path(cancel_path) if cancel_path else None
        self.heartbeat = TimedHeartbeat(
            lambda event: self._emit(event), training_config.heartbeat_interval_seconds,
        )
        self.background_heartbeat = BackgroundHeartbeat(
            lambda event: self._emit(event), training_config.heartbeat_interval_seconds,
        )

    def _emit(self, event) -> None:
        if self.progress_callback:
            self.progress_callback(event)

    def _cancel_if_requested(self, operation: str) -> None:
        if self.cancel_path is not None and self.cancel_path.exists():
            raise TrainingCancelled(f"Cooperative cancellation requested during {operation}")

    def _pulse(self, phase, *, force=False, **payload) -> None:
        context = {"generation": self.generation, **payload}
        self.background_heartbeat.update(phase, **context)
        self.heartbeat.pulse(phase, force=force, **context)

    def _indices(self, count: int, total: int, salt: int) -> tuple[int, ...]:
        rng = random.Random(self.training_config.seed + salt + self.generation * 104729)
        rows = list(range(total)); rng.shuffle(rows)
        return tuple(rows[:min(count, total)])

    @staticmethod
    def _quality(network, matrix, utilities, offsets, indices) -> float:
        values = []
        for group_index in indices:
            start, end = map(int, offsets[group_index:group_index + 2])
            state = network.initial_state(f"group-{group_index}")
            scores = [network.activate(row, state, commit=False).outputs[0] for row in matrix[start:end]]
            truth = utilities[start:end]
            spread = float(np.max(truth) - np.min(truth))
            regret = float(np.max(truth) - truth[int(np.argmax(scores))])
            values.append(1.0 if spread <= 1e-9 else 1.0 - min(1.0, regret / spread))
        return float(np.mean(values)) if values else 0.0

    def _network(self, genome, config):
        artifact = export_recurrent_genome(
            genome, config, self.feature_names,
            observation_schema_version=int(self.manifest["observation_schema_version"]),
            normalizer_version=int(self.manifest["player_feature_normalizer"]["version"]),
            evaluation_config=RecurrentEvaluationConfig(self.training_config.settling_steps),
            resource_limits=self.limits,
        )
        return artifact, RecurrentNeatNetwork(artifact)

    def _curriculum_stage(self) -> str:
        stable = (
            len(self.validation_history) >= self.training_config.stability_window
            and min(self.validation_history[-self.training_config.stability_window:])
            >= self.training_config.validation_quality_threshold
        )
        return "paired_campaign" if self.generation >= self.training_config.minimum_campaign_generation and stable else "supervised_ranking"

    def _campaign_inputs(self):
        scenarios = []
        total = self.training_config.campaign_scenarios_per_generation
        for index in range(total):
            self._cancel_if_requested("campaign baseline generation")
            self._pulse(
                HeartbeatPhase.SCENARIO_EVALUATION, force=True,
                operation="campaign_farm_generation", scenario_id=index,
                progress_current=index, progress_total=total,
                progress_label="Campaign baseline scenarios",
            )
            seed = self.training_config.seed + 2_000_003 + self.generation * 100_003 + index * 997
            generator = PairRankingDatasetGenerator(PairRankingDataConfig(
                number_of_farms=1, trials_per_pair=max(1, self.training_config.oracle_trial_count),
                seed=seed, minimum_rocks=6, maximum_rocks=6,
            ))
            farm = generator.create_procedural_farm(index)
            for rock in farm.rocks.values():
                rock.generation = 0
            campaign_config = BreedingCampaignConfig(
                max_generations=self.training_config.campaign_generations,
                max_pairs_per_generation=3,
                max_decisions=max(12, self.training_config.campaign_generations * 6),
            )
            evaluator = BreedingAgentEvaluator(campaign_config)
            self._cancel_if_requested("random campaign baseline")
            self._pulse(
                HeartbeatPhase.SCENARIO_EVALUATION, force=True,
                operation="random_campaign_baseline", scenario_id=index,
                progress_current=index, progress_total=total,
                progress_label="Campaign baseline scenarios",
            )
            random_episode = evaluator.run_episode(RandomBreedingAgent(), seed=seed, initial_farm=copy.deepcopy(farm))
            self._cancel_if_requested("oracle campaign baseline")
            self._pulse(
                HeartbeatPhase.SCENARIO_EVALUATION, force=True,
                operation="oracle_campaign_baseline", scenario_id=index,
                progress_current=index, progress_total=total,
                progress_label="Campaign baseline scenarios",
            )
            oracle_episode = evaluator.run_episode(
                OracleBreedingAgent(trial_count=self.training_config.oracle_trial_count),
                seed=seed, initial_farm=copy.deepcopy(farm),
            )
            scenarios.append((seed, farm, evaluator,
                float(random_episode.final_farm_summary["objective_utility"]),
                float(oracle_episode.final_farm_summary["objective_utility"])))
            self._pulse(
                HeartbeatPhase.SCENARIO_EVALUATION, force=True,
                operation="campaign_baseline_completed", scenario_id=index,
                progress_current=index + 1, progress_total=total,
                progress_label="Campaign baseline scenarios",
            )
        return scenarios

    @staticmethod
    def _normalized_campaign_score(agent_value: float, random_value: float, oracle_value: float) -> float:
        denominator = oracle_value - random_value
        return 0.0 if abs(denominator) <= 1e-9 else float(np.clip((agent_value - random_value) / denominator, -1.0, 1.0))

    def _campaign_quality(self, artifact, scenarios, *, genome_id=None, genome_index=0, genome_total=0) -> float:
        scores = []
        for scenario_index, (seed, farm, evaluator, random_value, oracle_value) in enumerate(scenarios):
            self._cancel_if_requested("genome campaign episode")
            self._pulse(
                HeartbeatPhase.WORLD_EPISODE,
                genome_id=genome_id, scenario_id=scenario_index,
                operation="genome_campaign_episode",
                progress_current=genome_index, progress_total=genome_total,
                progress_label="Genome campaign evaluation",
            )
            policy = RecurrentNeatPairRankingPolicy(artifact, checkpoint_id="in-memory-recurrent-champion")
            episode = evaluator.run_episode(
                RecurrentNeatBreedingAgent(policy), seed=seed, initial_farm=copy.deepcopy(farm)
            )
            scores.append(self._normalized_campaign_score(
                float(episode.final_farm_summary["objective_utility"]), random_value, oracle_value
            ))
        return float(np.mean(scores)) if scores else 0.0

    def _fitness(self, genomes, config):
        if self.cancel_path is not None and self.cancel_path.exists():
            raise TrainingCancelled("Cooperative cancellation requested")
        train_indices = self._indices(self.training_config.training_scenarios_per_generation, len(self.train_groups), 0)
        stage = self._curriculum_stage()
        self._pulse(
            HeartbeatPhase.GENERATION_FINALIZATION, force=True,
            operation="generation_preparation", progress_current=0,
            progress_total=len(genomes), progress_label="Genome evaluation",
        )
        campaign_scenarios = self._campaign_inputs() if stage == "paired_campaign" else ()
        scored = []
        progress_interval = max(1, len(genomes) // 20)
        for genome_index, (_, genome) in enumerate(genomes, start=1):
            if self.cancel_path is not None and self.cancel_path.exists():
                raise TrainingCancelled("Cooperative cancellation requested during genome evaluation")
            try:
                self._pulse(
                    HeartbeatPhase.GENOME_EVALUATION, force=True,
                    genome_id=genome.key, operation="supervised_pair_ranking",
                    progress_current=genome_index - 1, progress_total=len(genomes),
                    progress_label="Genome evaluation",
                )
                artifact, network = self._network(genome, config)
                quality = self._quality(network, self.train_matrix, self.train_arrays["utility_scores"], self.train_arrays["group_offsets"], train_indices)
                campaign_quality = self._campaign_quality(
                    artifact, campaign_scenarios, genome_id=genome.key,
                    genome_index=genome_index - 1, genome_total=len(genomes),
                )
                combined = quality if stage == "supervised_ranking" else (
                    self.training_config.supervised_weight * quality
                    + self.training_config.campaign_weight * campaign_quality
                )
                complexity = len(artifact.nodes) + artifact.enabled_connection_count
                genome.fitness = float(combined - min(0.02, complexity * self.training_config.complexity_penalty))
            except (ValueError, ArithmeticError, RuntimeError):
                genome.fitness = -1.0
                quality, campaign_quality, complexity = -1.0, -1.0, 0
            scored.append((genome.fitness, quality, campaign_quality, complexity, genome))
            self._pulse(
                HeartbeatPhase.GENOME_EVALUATION, force=True,
                genome_id=genome.key, operation="genome_completed",
                progress_current=genome_index, progress_total=len(genomes),
                progress_label="Genome evaluation",
            )
            if self.progress_callback and (genome_index % progress_interval == 0 or genome_index == len(genomes)):
                self._emit({
                    "event_type": "genome_evaluation_progress",
                    "generation": self.generation,
                    "genomes_evaluated": genome_index,
                    "genomes_total": len(genomes),
                })
        scored.sort(key=lambda row: row[0], reverse=True)
        champion = scored[0][4]
        artifact, network = self._network(champion, config)
        validation_indices = self._indices(self.training_config.validation_scenarios, len(self.validation_groups), 9_999_991)
        self._pulse(
            HeartbeatPhase.GENERATION_FINALIZATION, force=True,
            operation="validation", progress_current=0,
            progress_total=len(validation_indices), progress_label="Validation scenarios",
        )
        validation = self._quality(network, self.validation_matrix, self.validation_arrays["utility_scores"], self.validation_arrays["group_offsets"], validation_indices)
        self._pulse(
            HeartbeatPhase.GENERATION_FINALIZATION, force=True,
            operation="validation_completed", progress_current=len(validation_indices),
            progress_total=len(validation_indices), progress_label="Validation scenarios",
        )
        self.validation_history.append(validation)
        row = {
            "generation": self.generation, "curriculum_stage": stage,
            "best_fitness": scored[0][0], "mean_fitness": float(np.mean([row[0] for row in scored])),
            "training_quality": scored[0][1], "campaign_quality": scored[0][2], "validation_quality": validation,
            "topology_complexity": scored[0][3], "resource_limit_hits": BoundedRecurrentGenome.resource_limit_hits,
            "training_scenario_indices": list(train_indices), "validation_scenario_indices": list(validation_indices),
        }
        with (self.output / "generation_metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        if self.progress_callback:
            self._emit({"event_type": "generation_completed", **row})
        self._export_champion(champion, config, artifact, row)
        self.generation += 1

    def _export_champion(self, genome, config, artifact, metrics):
        self._pulse(
            HeartbeatPhase.CHECKPOINT_WRITING, force=True,
            operation="champion_export", progress_current=0,
            progress_total=3, progress_label="Champion artifacts",
        )
        directory = self.output / "champions" / f"generation_{self.generation:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        save_recurrent_artifact(artifact, directory / "network.json")
        save_recurrent_artifact(artifact, directory / "topology.json")
        safe_genome = {"genome_id": genome.key, "fitness": genome.fitness, "topology_id": artifact.topology_id}
        (directory / "genome.json").write_text(json.dumps(safe_genome, indent=2), encoding="utf-8")
        (directory / "metadata.json").write_text(json.dumps(artifact.metadata | safe_genome, indent=2), encoding="utf-8")
        (directory / "normalizer.json").write_text(json.dumps(self.manifest["player_feature_normalizer"], indent=2), encoding="utf-8")
        (directory / "observation_schema.json").write_text(json.dumps({"version": self.manifest["observation_schema_version"], "feature_names": list(self.feature_names)}, indent=2), encoding="utf-8")
        (directory / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (directory / "validation_metrics.json").write_text(json.dumps({"quality": metrics["validation_quality"]}, indent=2), encoding="utf-8")
        showcase_seed = self.training_config.seed + 7_777_777
        generator = PairRankingDatasetGenerator(PairRankingDataConfig(
            number_of_farms=1, trials_per_pair=1, seed=showcase_seed,
            minimum_rocks=6, maximum_rocks=6,
        ))
        showcase_farm = generator.create_procedural_farm(0)
        for rock in showcase_farm.rocks.values():
            rock.generation = 0
        campaign_config = BreedingCampaignConfig(max_generations=3, max_pairs_per_generation=3, max_decisions=24)
        initial_environment = BreedingCampaignEnvironment(seed=showcase_seed, config=campaign_config)
        initial_environment.reset(showcase_seed, initial_farm=copy.deepcopy(showcase_farm))
        self._pulse(
            HeartbeatPhase.CHECKPOINT_WRITING, force=True,
            operation="showcase_episode", progress_current=1,
            progress_total=3, progress_label="Champion artifacts",
        )
        episode = BreedingAgentEvaluator(campaign_config).run_episode(
            RecurrentNeatBreedingAgent(RecurrentNeatPairRankingPolicy(artifact, checkpoint_id="showcase")),
            seed=showcase_seed, initial_farm=copy.deepcopy(showcase_farm),
        )
        with gzip.open(directory / "showcase_episode.jsonl.gz", "wt", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "generation": self.generation, "showcase_seed": showcase_seed,
                "initial_game": game_to_dict(initial_environment.game), "episode": episode.to_dict(),
                "note": "Fixed showcase episode is replay-only and never contributes to reproduction fitness.",
            }, sort_keys=True) + "\n")
        network = RecurrentNeatNetwork(artifact)
        sample = self.validation_matrix[:1] if len(self.validation_matrix) else self.train_matrix[:1]
        result = network.activate(sample[0], network.initial_state("showcase"), commit=True) if len(sample) else None
        np.savez_compressed(directory / "activation_trace.npz", inputs=sample, outputs=np.asarray(result.outputs if result else ()), settling_steps=np.asarray(self.training_config.settling_steps))
        before = result.trace["state_before"]["node_activations"] if result else ()
        after = result.trace["state_after"]["node_activations"] if result else ()
        np.savez_compressed(
            directory / "memory_trace.npz",
            before_node_ids=np.asarray([int(row[0]) for row in before], dtype=np.int64),
            before_values=np.asarray([float(row[1]) for row in before], dtype=np.float64),
            after_node_ids=np.asarray([int(row[0]) for row in after], dtype=np.int64),
            after_values=np.asarray([float(row[1]) for row in after], dtype=np.float64),
        )
        self._pulse(
            HeartbeatPhase.CHECKPOINT_WRITING, force=True,
            operation="champion_export_completed", progress_current=3,
            progress_total=3, progress_label="Champion artifacts",
        )

    def train(self):
        random.seed(self.training_config.seed); np.random.seed(self.training_config.seed)
        self.prepare_run_metadata()
        population = neat.Population(self.neat_config, seed=self.training_config.seed)
        return self.train_population(population, self.training_config.generations)

    def prepare_run_metadata(self, extra_manifest: dict[str, Any] | None = None) -> None:
        training_path = self.output / "training_config.json"
        manifest_path = self.output / "run_manifest.json"
        if not training_path.exists():
            training_path.write_text(json.dumps(asdict(self.training_config), indent=2), encoding="utf-8")
        manifest = {
            "run_type": "recurrent_neat_player_farmer", "information_access": "player",
            "topology_implementation_version": 1,
            "fitness_implementation_version": 1,
            "observation_schema_version": self.manifest["observation_schema_version"],
            "feature_names": list(self.feature_names), "safe_artifacts_only": True,
            "candidate_memory_semantics": "same_snapshot_commit_selected_only",
        }
        manifest.update(extra_manifest or {})
        if not manifest_path.exists():
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def train_population(self, population, generations: int, *, extra_reporters=()):
        population.add_reporter(neat.StdOutReporter(True))
        species_reporter = _SpeciesReporter(self.output / "species_metrics.jsonl")
        species_reporter.generation = self.generation
        population.add_reporter(species_reporter)
        for reporter in extra_reporters:
            population.add_reporter(reporter)
        (self.output / "checkpoints").mkdir(parents=True, exist_ok=True)
        population.add_reporter(_HeartbeatCheckpointer(
            self.training_config.checkpoint_frequency,
            filename_prefix=str(self.output / "checkpoints" / "neat-checkpoint-"),
            heartbeat=self._pulse,
        ))
        self.background_heartbeat.start()
        try:
            return population.run(self._fitness, int(generations))
        finally:
            self.background_heartbeat.stop()
