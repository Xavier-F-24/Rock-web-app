"""Deterministic recurrent NEAT evolution for complete player-like farmers."""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

import neat

from Rock_AI.actions.action_schema import ActionObservationSchema
from Rock_AI.agents.full_neat_farmer_agent import FullNeatFarmerAgent
from Rock_AI.environments.multi_farm_economy_environment import MultiFarmEconomyConfig, MultiFarmEconomyEnvironment
from Rock_AI.neat.neat_export_helper import export_recurrent_genome, save_recurrent_artifact
from Rock_AI.neat.neat_genome_helper import BoundedRecurrentGenome
from Rock_AI.neat.neat_recurrent_network import RecurrentEvaluationConfig
from Rock_AI.neat.neat_topology_helper import TopologyResourceLimits
from Rock_AI.policies.market_action_policy_adapter import LegalFarmerActionGenerator
from Rock_AI.policies.recurrent_neat_farmer_policy import FULL_FARMER_OBSERVATION_SCHEMA_VERSION, RecurrentNeatFarmerPolicy
from Rock_AI.training.recurrent_neat_training_helper import _write_config

from .action_curriculum import availability_for_stage
from .full_farmer_fitness import farm_objective_utility, normalized_campaign_fitness
from .full_farmer_training_config import FullFarmerTrainingConfig
from .multi_agent_scenario_manager import MultiAgentScenarioManager
from .opponent_pool import OpponentPool


class FullFarmerNeatTrainer:
    def __init__(self, config: FullFarmerTrainingConfig):
        self.training_config = config
        self.output = Path(config.output_directory)
        if self.output.exists() and any(self.output.iterdir()):
            raise FileExistsError(f"Training run already exists: {self.output}")
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "champions").mkdir(exist_ok=True)
        (self.output / "checkpoints").mkdir(exist_ok=True)
        self.action_schema = ActionObservationSchema()
        self.feature_names = self.action_schema.feature_names + tuple(f"{name}.visible" for name in self.action_schema.feature_names)
        self.limits = TopologyResourceLimits()
        BoundedRecurrentGenome.resource_limits = self.limits
        self.neat_config_path = self.output / "neat_config.ini"
        _write_config(self.neat_config_path, len(self.feature_names), config.population)
        self.neat_config = neat.Config(BoundedRecurrentGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, str(self.neat_config_path))
        self.scenarios = MultiAgentScenarioManager()
        self.opponents = OpponentPool()
        self.generation = 0
        self.metrics = []
        (self.output / "training_config.json").write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        (self.output / "action_schema.json").write_text(json.dumps({"version": self.action_schema.version, "feature_names": self.feature_names}, indent=2), encoding="utf-8")
        (self.output / "observation_schema.json").write_text(json.dumps({"version": FULL_FARMER_OBSERVATION_SCHEMA_VERSION, "information_access": "player"}, indent=2), encoding="utf-8")

    def _artifact(self, genome, *, generation=None):
        return export_recurrent_genome(
            genome, self.neat_config, self.feature_names,
            observation_schema_version=FULL_FARMER_OBSERVATION_SCHEMA_VERSION,
            normalizer_version=1, output_names=("action_score", "confidence", "value_estimate"),
            evaluation_config=RecurrentEvaluationConfig(self.training_config.settling_steps),
            resource_limits=self.limits,
            metadata={
                "policy_kind": "full_farmer", "generation": self.generation if generation is None else generation,
                "curriculum_stage": self.training_config.curriculum_start.name.lower(),
                "supported_action_types": [value.value for value in availability_for_stage(self.training_config.curriculum_start).enabled],
            },
        )

    def _evaluate_genome(self, genome) -> float:
        artifact = self._artifact(genome)
        scores = []
        definitions = self.scenarios.scenarios(self.training_config.seed + self.generation * 100_003, self.training_config.worlds_per_genome)
        for scenario_index, definition in enumerate(definitions):
            world = self.scenarios.build(definition)
            farm_ids = sorted(world.farms)
            policy = RecurrentNeatFarmerPolicy(artifact, checkpoint_id="in-memory")
            controlled = FullNeatFarmerAgent(policy, f"candidate-{genome.key}")
            agents = {farm_ids[0]: controlled}
            for farm_id, opponent in zip(farm_ids[1:], self.opponents.select(definition.seed, len(farm_ids) - 1)):
                agents[farm_id] = opponent
            for index, (farm_id, agent) in enumerate(sorted(agents.items())):
                agent.reset(definition.seed + 20_000 + index, f"train-{self.generation}-{scenario_index}")
            env = MultiFarmEconomyEnvironment(definition.seed, MultiFarmEconomyConfig(max_world_turns=self.training_config.max_rounds_per_world))
            env.reset(initial_world=world)
            env.candidate_generator = LegalFarmerActionGenerator(availability=availability_for_stage(self.training_config.curriculum_start))
            env.observation_adapter.candidate_generator = env.candidate_generator
            initial = farm_objective_utility(env.world.farm(farm_ids[0]))
            invalid = 0
            for _ in range(self.training_config.max_rounds_per_world):
                selected = {}
                for farm_id, agent in agents.items():
                    observation = env.observe(farm_id, recurrent_state=getattr(getattr(agent, "policy", None), "state", None))
                    selected[farm_id] = agent.choose_candidate(observation)
                result = env.resolve_round(selected)
                by_farm = {row.actor_farm_id: row for row in result.action_results}
                for farm_id, agent in agents.items():
                    agent.observe_result(selected[farm_id], by_farm[farm_id])
                    invalid += int(not by_farm[farm_id].success and farm_id == farm_ids[0])
                if env.terminated:
                    break
            score = normalized_campaign_fitness(env.world, farm_ids[0], initial) - invalid * .25
            scores.append(score)
        complexity = len(genome.nodes) + sum(connection.enabled for connection in genome.connections.values())
        return sum(scores) / max(1, len(scores)) - self.training_config.complexity_penalty * complexity

    def _fitness(self, genomes, neat_config):
        values = []
        for _, genome in sorted(genomes):
            try:
                genome.fitness = float(self._evaluate_genome(genome))
            except Exception:
                genome.fitness = -10.0
            values.append(genome.fitness)
        row = {
            "generation": self.generation, "curriculum_stage": self.training_config.curriculum_start.name.lower(),
            "best_fitness": max(values), "mean_fitness": sum(values) / len(values),
            "population": len(values), "worlds_evaluated": len(values) * self.training_config.worlds_per_genome,
        }
        self.metrics.append(row)
        with (self.output / "generation_metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        self.generation += 1

    def train(self):
        random.seed(self.training_config.seed)
        population = neat.Population(self.neat_config, seed=self.training_config.seed)
        population.add_reporter(neat.Checkpointer(self.training_config.checkpoint_frequency, filename_prefix=str(self.output / "checkpoints" / "neat-checkpoint-")))
        winner = population.run(self._fitness, self.training_config.generations)
        artifact = self._artifact(winner, generation=self.generation - 1)
        champion = self.output / "champions" / "best_validation"
        champion.mkdir(parents=True, exist_ok=True)
        save_recurrent_artifact(artifact, champion / "network.json")
        save_recurrent_artifact(artifact, champion / "topology.json")
        metadata = {
            "run_type": "recurrent_neat_full_farmer", "information_access": "player",
            "generation": self.generation - 1, "fitness": float(winner.fitness),
            "action_schema_version": self.action_schema.version,
            "observation_schema_version": FULL_FARMER_OBSERVATION_SCHEMA_VERSION,
            "curriculum_stage": self.training_config.curriculum_start.name.lower(),
            "supported_action_types": artifact.metadata["supported_action_types"],
        }
        (champion / "full_farmer_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        (self.output / "run_manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        return artifact, metadata
