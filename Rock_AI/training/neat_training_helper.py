"""Deterministic staged NEAT training for player-visible pair ranking."""

from __future__ import annotations

import gzip
import json
import math
import random
import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import neat
import numpy as np

from Rock_AI.agents.neat_breeding_agent import NeatBreedingAgent
from Rock_AI.agents.oracle_breeding_agent import OracleBreedingAgent
from Rock_AI.agents.random_breeding_agent import RandomBreedingAgent
from Rock_AI.datasets.pair_ranking_dataset_generator import PairRankingDatasetGenerator
from Rock_AI.datasets.pair_ranking_storage_helper import load_pair_ranking_split
from Rock_AI.evaluation.breeding_agent_evaluator import BreedingAgentEvaluator
from Rock_AI.environments.breeding_campaign_environment import BreedingCampaignConfig
from Rock_AI.models.neat_network_helper import export_neat_genome, save_neat_artifact
from Rock_AI.policies.neat_pair_ranking_policy import NeatPairRankingPolicy
from Rock_AI.representations.player_observation_helper import PLAYER_OBSERVATION_SCHEMA_VERSION
from Rock_AI.training.training_config_helper import PairRankingDataConfig
from Rock_Serialization.rock_serialization_helper import game_to_dict


@dataclass(frozen=True)
class NeatTrainingConfig:
    dataset_path: str
    output_directory: str
    population: int = 100
    generations: int = 50
    seed: int = 1234
    training_scenarios_per_generation: int = 12
    validation_scenarios: int = 12
    campaign_scenarios: int = 1
    oracle_trial_count: int = 25
    checkpoint_frequency: int = 5
    complexity_penalty: float = 0.00001
    supervised_weight: float = 0.60
    campaign_weight: float = 0.40
    short_campaign_start_fraction: float = 0.35
    long_campaign_start_fraction: float = 0.70
    short_campaign_generations: int = 2
    long_campaign_generations: int = 4

    def __post_init__(self) -> None:
        if min(self.population, self.generations, self.training_scenarios_per_generation) <= 0:
            raise ValueError("population, generations, and training scenarios must be positive")
        if self.population > 10000 or self.generations > 10000:
            raise ValueError("NEAT configuration exceeds supported safety bounds")
        if self.complexity_penalty < 0:
            raise ValueError("complexity_penalty cannot be negative")


@dataclass(frozen=True)
class ScenarioSchedule:
    run_seed: int
    training_group_count: int
    validation_group_count: int

    def rotating_training_indices(self, generation: int, count: int) -> tuple[int, ...]:
        rng = random.Random(self.run_seed + 104729 * int(generation))
        population = list(range(self.training_group_count))
        rng.shuffle(population)
        return tuple(population[: min(count, len(population))])

    def validation_indices(self, count: int) -> tuple[int, ...]:
        rng = random.Random(self.run_seed + 9_999_991)
        population = list(range(self.validation_group_count))
        rng.shuffle(population)
        return tuple(population[: min(count, len(population))])

    @property
    def showcase_index(self) -> int:
        return 0


def normalized_campaign_score(agent: float, random_value: float, oracle_value: float) -> float:
    denominator = oracle_value - random_value
    if abs(denominator) <= 1e-9:
        return 0.0
    return float(np.clip((agent - random_value) / denominator, -1.0, 1.0))


def topology_complexity_penalty(config: NeatTrainingConfig, complexity: int) -> float:
    return min(float(config.complexity_penalty) * max(0, int(complexity)), 0.02)


def _candidate_matrix(arrays: dict[str, np.ndarray]) -> np.ndarray:
    left = arrays["parent_a_features"]
    right = arrays["parent_b_features"]
    return np.concatenate(
        (
            left + right,
            np.abs(left - right),
            left * right,
            arrays["rule_features"],
            arrays["farm_features"],
            arrays["objective_features"],
            arrays["metadata_features"],
            arrays["predictor_features"],
        ),
        axis=1,
    ).astype(np.float64)


def _candidate_feature_names(manifest: dict[str, Any]) -> tuple[str, ...]:
    names = manifest["feature_names"]
    parent = tuple(names["parent"])
    result = (
        tuple(f"parent_sum.{name}" for name in parent)
        + tuple(f"parent_absolute_difference.{name}" for name in parent)
        + tuple(f"parent_product.{name}" for name in parent)
    )
    for group in ("rules", "farm", "objective", "metadata", "predictor"):
        result += tuple(f"{group}.{name}" for name in names.get(group, ()))
    return result


def _write_neat_config(path: Path, input_count: int, population: int) -> None:
    text = f"""[NEAT]
fitness_criterion = max
fitness_threshold = 1.0
pop_size = {population}
reset_on_extinction = False
no_fitness_termination = True

[DefaultGenome]
activation_default = tanh
activation_mutate_rate = 0.0
activation_options = tanh
aggregation_default = sum
aggregation_mutate_rate = 0.0
aggregation_options = sum
bias_init_mean = 0.0
bias_init_stdev = 1.0
bias_max_value = 30.0
bias_min_value = -30.0
bias_mutate_power = 0.5
bias_mutate_rate = 0.7
bias_replace_rate = 0.1
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient = 0.5
conn_add_prob = 0.5
conn_delete_prob = 0.3
enabled_default = True
enabled_mutate_rate = 0.01
feed_forward = True
initial_connection = full_direct
node_add_prob = 0.2
node_delete_prob = 0.1
num_hidden = 0
num_inputs = {input_count}
num_outputs = 1
response_init_mean = 1.0
response_init_stdev = 0.0
response_max_value = 30.0
response_min_value = -30.0
response_mutate_power = 0.0
response_mutate_rate = 0.0
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
"""
    path.write_text(text, encoding="utf-8")


class NeatPairRankerTrainer:
    def __init__(self, training_config: NeatTrainingConfig):
        self.training_config = training_config
        self.output = Path(training_config.output_directory)
        if self.output.exists() and any(self.output.iterdir()):
            raise FileExistsError(f"Training run already exists: {self.output}")
        self.output.mkdir(parents=True, exist_ok=True)
        self.train_arrays, self.train_groups, self.manifest = load_pair_ranking_split(
            training_config.dataset_path, "train"
        )
        self.validation_arrays, self.validation_groups, validation_manifest = load_pair_ranking_split(
            training_config.dataset_path, "validation"
        )
        if self.manifest.get("information_access") != "player":
            raise ValueError("NEAT gameplay training requires a player-observation dataset")
        if validation_manifest.get("observation_schema_version") != self.manifest.get("observation_schema_version"):
            raise ValueError("Training and validation schemas differ")
        self.train_matrix = _candidate_matrix(self.train_arrays)
        self.validation_matrix = _candidate_matrix(self.validation_arrays)
        self.feature_names = _candidate_feature_names(self.manifest)
        if self.train_matrix.shape[1] != len(self.feature_names):
            raise ValueError("NEAT feature names do not match candidate matrix")
        self.schedule = ScenarioSchedule(
            training_config.seed, len(self.train_groups), len(self.validation_groups)
        )
        self.generation = 0
        self.metrics: list[dict[str, Any]] = []
        self.neat_config_path = self.output / "neat_config.ini"
        _write_neat_config(self.neat_config_path, self.train_matrix.shape[1], training_config.population)
        self.neat_config = neat.Config(
            neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
            neat.DefaultStagnation, str(self.neat_config_path),
        )

    def _campaign_inputs(self, stage: str):
        if stage == "supervised" or self.training_config.campaign_scenarios <= 0:
            return []
        maximum_generations = (
            self.training_config.short_campaign_generations
            if stage == "short_campaign"
            else self.training_config.long_campaign_generations
        )
        scenario_rows = []
        for index in range(self.training_config.campaign_scenarios):
            scenario_seed = (
                self.training_config.seed + 2_000_003
                + self.generation * 100_003 + index * 997
            )
            generator = PairRankingDatasetGenerator(PairRankingDataConfig(
                number_of_farms=1,
                trials_per_pair=max(1, self.training_config.oracle_trial_count),
                seed=scenario_seed,
                minimum_rocks=6,
                maximum_rocks=6,
            ))
            farm = generator.create_procedural_farm(index)
            for rock in farm.rocks.values():
                rock.generation = 0
            evaluator = BreedingAgentEvaluator(BreedingCampaignConfig(
                max_generations=maximum_generations,
                max_pairs_per_generation=3,
                max_decisions=max(12, maximum_generations * 6),
            ))
            random_record = evaluator.run_episode(
                RandomBreedingAgent(), seed=scenario_seed, initial_farm=copy.deepcopy(farm)
            )
            oracle_record = evaluator.run_episode(
                OracleBreedingAgent(trial_count=self.training_config.oracle_trial_count),
                seed=scenario_seed,
                initial_farm=copy.deepcopy(farm),
            )
            scenario_rows.append({
                "seed": scenario_seed,
                "farm": farm,
                "evaluator": evaluator,
                "random_utility": float(random_record.final_farm_summary["objective_utility"]),
                "oracle_utility": float(oracle_record.final_farm_summary["objective_utility"]),
                "random_termination_reason": random_record.termination_reason,
                "oracle_termination_reason": oracle_record.termination_reason,
            })
        return scenario_rows

    def _campaign_quality(self, genome, config, scenarios):
        if not scenarios:
            return 0.0, []
        artifact = export_neat_genome(
            genome, config, self.feature_names,
            observation_schema_version=int(self.manifest["observation_schema_version"]),
            normalizer_version=int(self.manifest["player_feature_normalizer"]["version"]),
        )
        rows = []
        for scenario in scenarios:
            policy = NeatPairRankingPolicy(artifact, checkpoint_id="in-memory-training-genome")
            record = scenario["evaluator"].run_episode(
                NeatBreedingAgent(policy),
                seed=scenario["seed"],
                initial_farm=copy.deepcopy(scenario["farm"]),
            )
            agent_utility = float(record.final_farm_summary["objective_utility"])
            score = normalized_campaign_score(
                agent_utility, scenario["random_utility"], scenario["oracle_utility"]
            )
            rows.append({
                "scenario_seed": scenario["seed"],
                "raw_agent_utility": agent_utility,
                "raw_random_utility": scenario["random_utility"],
                "raw_evaluation_oracle_utility": scenario["oracle_utility"],
                "normalized_campaign_score": score,
                "termination_reason": record.termination_reason,
            })
        return float(np.mean([row["normalized_campaign_score"] for row in rows])), rows

    @staticmethod
    def _group_quality(network, matrix, utilities, offsets, group_indices) -> float:
        qualities = []
        for group_index in group_indices:
            start, end = map(int, offsets[group_index: group_index + 2])
            scores = np.asarray([network.activate(row)[0] for row in matrix[start:end]])
            truth = utilities[start:end]
            chosen = int(np.argmax(scores))
            spread = float(np.max(truth) - np.min(truth))
            regret = float(np.max(truth) - truth[chosen])
            qualities.append(1.0 if spread <= 1e-9 else 1.0 - min(1.0, regret / spread))
        return float(np.mean(qualities)) if qualities else 0.0

    @staticmethod
    def _complexity(genome) -> int:
        return len(genome.nodes) + sum(connection.enabled for connection in genome.connections.values())

    def _curriculum_stage(self) -> str:
        progress = self.generation / max(1, self.training_config.generations - 1)
        if progress < self.training_config.short_campaign_start_fraction:
            return "supervised"
        if progress < self.training_config.long_campaign_start_fraction:
            return "short_campaign"
        return "long_campaign"

    def _fitness(self, genomes, config) -> None:
        indices = self.schedule.rotating_training_indices(
            self.generation, self.training_config.training_scenarios_per_generation
        )
        stage = self._curriculum_stage()
        campaign_scenarios = self._campaign_inputs(stage)
        scored = []
        for _, genome in genomes:
            network = neat.nn.FeedForwardNetwork.create(genome, config)
            supervised = self._group_quality(
                network, self.train_matrix, self.train_arrays["utility_scores"],
                self.train_arrays["group_offsets"], indices,
            )
            campaign, campaign_rows = self._campaign_quality(
                genome, config, campaign_scenarios
            )
            combined = (
                supervised if stage == "supervised" else
                self.training_config.supervised_weight * supervised
                + self.training_config.campaign_weight * campaign
            )
            complexity = self._complexity(genome)
            penalty = topology_complexity_penalty(self.training_config, complexity)
            genome.fitness = float(combined - penalty)
            scored.append((genome.fitness, supervised, campaign, complexity, genome, campaign_rows))
        scored.sort(key=lambda row: row[0], reverse=True)
        champion = scored[0][4]
        validation = self._group_quality(
            neat.nn.FeedForwardNetwork.create(champion, config),
            self.validation_matrix, self.validation_arrays["utility_scores"],
            self.validation_arrays["group_offsets"],
            self.schedule.validation_indices(self.training_config.validation_scenarios),
        )
        row = {
            "generation": self.generation,
            "curriculum_stage": stage,
            "training_scenario_indices": list(indices),
            "validation_scenario_indices": list(self.schedule.validation_indices(self.training_config.validation_scenarios)),
            "showcase_scenario_index": self.schedule.showcase_index,
            "best_fitness": scored[0][0],
            "mean_fitness": float(np.mean([item[0] for item in scored])),
            "supervised_quality": scored[0][1],
            "campaign_quality": scored[0][2],
            "validation_quality": validation,
            "topology_complexity": scored[0][3],
            "complexity_penalty": topology_complexity_penalty(self.training_config, scored[0][3]),
            "campaign_scenarios": scored[0][5],
            "evaluation_oracle_note": (
                "PairEvaluator-driven baseline; not a globally optimal long-horizon policy"
            ),
        }
        self.metrics.append(row)
        self._export_champion(champion, config, row)
        with (self.output / "generation_metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        self.generation += 1

    def _export_champion(self, genome, config, metrics: dict[str, Any]) -> None:
        directory = self.output / "champions" / f"generation_{self.generation:04d}"
        artifact = export_neat_genome(
            genome, config, self.feature_names,
            observation_schema_version=int(self.manifest["observation_schema_version"]),
            normalizer_version=int(self.manifest["player_feature_normalizer"]["version"]),
            metadata={"generation": self.generation, "fitness": genome.fitness},
        )
        save_neat_artifact(artifact, directory / "network.json")
        (directory / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        showcase_seed = self.training_config.seed + 7_777_777
        generator = PairRankingDatasetGenerator(PairRankingDataConfig(
            number_of_farms=1,
            trials_per_pair=1,
            seed=showcase_seed,
            minimum_rocks=6,
            maximum_rocks=6,
        ))
        showcase_farm = generator.create_procedural_farm(0)
        for rock in showcase_farm.rocks.values():
            rock.generation = 0
        environment_config = BreedingCampaignConfig(
            max_generations=3,
            max_pairs_per_generation=3,
            max_decisions=24,
        )
        initial_environment = __import__(
            "Rock_AI.environments.breeding_campaign_environment",
            fromlist=["BreedingCampaignEnvironment"],
        ).BreedingCampaignEnvironment(seed=showcase_seed, config=environment_config)
        initial_environment.reset(showcase_seed, initial_farm=copy.deepcopy(showcase_farm))
        evaluator = BreedingAgentEvaluator(environment_config)
        policy = NeatPairRankingPolicy(artifact, checkpoint_id="exported-showcase-champion")
        episode = evaluator.run_episode(
            NeatBreedingAgent(policy),
            seed=showcase_seed,
            initial_farm=copy.deepcopy(showcase_farm),
        )
        showcase = {
            "artifact_version": 1,
            "generation": self.generation,
            "showcase_seed": showcase_seed,
            "initial_game": game_to_dict(initial_environment.game),
            "episode": episode.to_dict(),
            "fitness_contribution": 0.0,
            "note": "Fixed showcase episode is replay-only and never contributes to reproduction fitness",
        }
        directory.mkdir(parents=True, exist_ok=True)
        with gzip.open(directory / "showcase_episode.jsonl.gz", "wt", encoding="utf-8") as stream:
            stream.write(json.dumps(showcase, sort_keys=True) + "\n")
        decision_traces = [
            decision.model_trace for decision in episode.decisions
            if decision.model_trace is not None
        ]
        selected_trace = decision_traces[0] if decision_traces else {
            "input_values": (), "node_activations": {}, "output_scores": {}
        }
        np.savez_compressed(
            directory / "activation_trace.npz",
            input_values=np.asarray(selected_trace.get("input_values", ()), dtype=np.float64),
            output_scores=np.asarray(list(selected_trace.get("output_scores", {}).values()), dtype=np.float64),
            selected_index=np.asarray([0], dtype=np.int64),
            selected_node_ids=np.asarray(
                list(map(int, selected_trace.get("node_activations", {}).keys())), dtype=np.int64
            ),
            selected_node_activations=np.asarray(
                list(selected_trace.get("node_activations", {}).values()), dtype=np.float64
            ),
        )

    def train(self):
        random.seed(self.training_config.seed)
        np.random.seed(self.training_config.seed)
        (self.output / "training_config.json").write_text(
            json.dumps(asdict(self.training_config), indent=2, sort_keys=True), encoding="utf-8"
        )
        (self.output / "run_manifest.json").write_text(json.dumps({
            "run_type": "neat_player_pair_ranker",
            "information_access": "player",
            "observation_schema_version": self.manifest["observation_schema_version"],
            "normalizer": self.manifest["player_feature_normalizer"],
            "feature_names": list(self.feature_names),
            "training_scenarios": "rotating by generation",
            "validation_used_for_fitness": False,
            "showcase_used_for_fitness": False,
        }, indent=2, sort_keys=True), encoding="utf-8")
        population = neat.Population(self.neat_config, seed=self.training_config.seed)
        population.add_reporter(neat.StdOutReporter(True))
        population.add_reporter(neat.Checkpointer(
            self.training_config.checkpoint_frequency,
            filename_prefix=str(self.output / "checkpoints" / "neat-checkpoint-"),
        ))
        (self.output / "checkpoints").mkdir(parents=True, exist_ok=True)
        return population.run(self._fitness, self.training_config.generations)
