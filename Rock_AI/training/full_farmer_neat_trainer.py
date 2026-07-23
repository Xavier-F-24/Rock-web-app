"""Deterministic recurrent NEAT evolution for complete player-like farmers."""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

import neat

from Rock_AI.actions.action_schema import ActionObservationSchema
from Rock_AI.agents.full_neat_farmer_agent import FullNeatFarmerAgent
from Rock_AI.environments.episode_liveness_helper import EpisodeLivenessLimits, EpisodeTerminationReason
from Rock_AI.environments.multi_farm_economy_environment import MultiFarmEconomyConfig, MultiFarmEconomyEnvironment
from Rock_AI.neat.neat_export_helper import export_recurrent_genome, save_recurrent_artifact
from Rock_AI.neat.neat_genome_helper import BoundedRecurrentGenome
from Rock_AI.neat.neat_recurrent_network import RecurrentEvaluationConfig
from Rock_AI.neat.neat_topology_helper import TopologyResourceLimits
from Rock_AI.policies.market_action_policy_adapter import LegalFarmerActionGenerator
from Rock_AI.policies.recurrent_neat_farmer_policy import FULL_FARMER_OBSERVATION_SCHEMA_VERSION, RecurrentNeatFarmerPolicy
from Rock_AI.training.recurrent_neat_training_helper import TrainingCancelled, _write_config
from Rock_AI.training_jobs.training_progress_reader import atomic_write_json
from Rock_AI.training_jobs.worker_heartbeat import BackgroundHeartbeat, HeartbeatPhase, TimedHeartbeat

from .action_curriculum import availability_for_stage
from .full_farmer_fitness import farm_objective_utility, normalized_campaign_fitness
from .full_farmer_training_config import FullFarmerTrainingConfig
from .full_farmer_training_state import FullFarmerTrainingState, should_advance_curriculum
from .multi_agent_scenario_manager import MultiAgentScenarioManager
from .opponent_pool import OpponentPool
from .stagnation_diagnostic_helper import write_stagnation_diagnostic


class HeartbeatCheckpointer(neat.Checkpointer):
    """NEAT checkpointer that keeps durable workers visibly alive during I/O."""

    def __init__(self, *args, heartbeat=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.heartbeat = heartbeat

    def save_checkpoint(self, config, population, species_set, generation):
        if self.heartbeat:
            self.heartbeat(HeartbeatPhase.CHECKPOINT_WRITING, force=True, operation="checkpoint_started", checkpoint_generation=generation)
        result = super().save_checkpoint(config, population, species_set, generation)
        if self.heartbeat:
            self.heartbeat(HeartbeatPhase.CHECKPOINT_WRITING, force=True, operation="checkpoint_completed", checkpoint_generation=generation)
        return result

    def __getstate__(self):
        state = dict(self.__dict__)
        state["heartbeat"] = None
        return state


class FullFarmerNeatTrainer:
    def __init__(self, config: FullFarmerTrainingConfig, *, allow_existing=False, starting_generation=0, progress_callback=None, cancel_path=None):
        self.training_config = config
        self.output = Path(config.output_directory)
        if self.output.exists() and any(self.output.iterdir()) and not allow_existing:
            raise FileExistsError(f"Training run already exists: {self.output}")
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "champions").mkdir(exist_ok=True)
        (self.output / "checkpoints").mkdir(exist_ok=True)
        self.action_schema = ActionObservationSchema()
        self.feature_names = self.action_schema.feature_names + tuple(f"{name}.visible" for name in self.action_schema.feature_names)
        self.limits = TopologyResourceLimits()
        BoundedRecurrentGenome.resource_limits = self.limits
        self.neat_config_path = self.output / "neat_config.ini"
        if not self.neat_config_path.exists():
            _write_config(self.neat_config_path, len(self.feature_names), config.population)
        self.neat_config = neat.Config(BoundedRecurrentGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, str(self.neat_config_path))
        self.scenarios = MultiAgentScenarioManager()
        self.opponents = OpponentPool()
        self.generation = int(starting_generation)
        self.metrics = []
        self.progress_callback = progress_callback
        self.heartbeat = TimedHeartbeat(lambda event: self._emit(event), config.heartbeat_interval_seconds)
        self.background_heartbeat = BackgroundHeartbeat(lambda event: self._emit(event), config.heartbeat_interval_seconds)
        self.run_metadata = {}
        self.cancel_path = Path(cancel_path) if cancel_path else None
        state_path = self.output / "full_farmer_training_state.json"
        if allow_existing and state_path.exists():
            self.state = FullFarmerTrainingState.from_dict(json.loads(state_path.read_text(encoding="utf-8")))
            self.generation = max(self.generation, self.state.generation + 1)
        else:
            self.state = FullFarmerTrainingState(self.generation, config.curriculum_start, self.generation)
        if not allow_existing:
            atomic_write_json(self.output / "training_config.json", config.to_dict())
            atomic_write_json(self.output / "action_schema.json", {"version": self.action_schema.version, "feature_names": self.feature_names})
            atomic_write_json(self.output / "observation_schema.json", {"version": FULL_FARMER_OBSERVATION_SCHEMA_VERSION, "information_access": "player"})
        self.prepare_run_metadata()

    @property
    def curriculum_stage(self):
        return self.state.curriculum_stage

    def prepare_run_metadata(self, extra=None):
        metadata = {
            "run_type": "recurrent_neat_full_farmer", "information_access": "player",
            "action_schema_version": self.action_schema.version,
            "observation_schema_version": FULL_FARMER_OBSERVATION_SCHEMA_VERSION,
            "topology_implementation_version": 1, "fitness_implementation_version": 2,
        }
        metadata.update(extra or {})
        self.run_metadata = metadata
        atomic_write_json(self.output / "run_manifest.json", metadata)
        return metadata

    def _cancel_if_requested(self):
        if self.cancel_path and self.cancel_path.exists():
            raise TrainingCancelled("Full-farmer training cancellation requested")

    def _emit(self, event):
        if self.progress_callback:
            self.progress_callback(event)

    def _pulse(self, phase, *, force=False, **payload):
        self.background_heartbeat.update(phase, generation=self.generation, **payload)
        self.heartbeat.pulse(phase, force=force, generation=self.generation, **payload)

    def _environment_heartbeat(self, phase, payload):
        self._pulse(phase, **payload)

    def _artifact(self, genome, *, generation=None):
        return export_recurrent_genome(
            genome, self.neat_config, self.feature_names,
            observation_schema_version=FULL_FARMER_OBSERVATION_SCHEMA_VERSION,
            normalizer_version=1, output_names=("action_score", "confidence", "value_estimate"),
            evaluation_config=RecurrentEvaluationConfig(self.training_config.settling_steps),
            resource_limits=self.limits,
            metadata={
                "policy_kind": "full_farmer", "generation": self.generation if generation is None else generation,
                "curriculum_stage": self.curriculum_stage.name.lower(),
                "supported_action_types": [value.value for value in availability_for_stage(self.curriculum_stage).enabled],
            },
        )

    def _evaluate_genome(self, genome, *, scenario_seed=None):
        self._pulse(HeartbeatPhase.GENOME_EVALUATION, force=True, genome_id=genome.key, operation="genome_started")
        artifact = self._artifact(genome)
        scores = []
        definitions = self.scenarios.scenarios(scenario_seed if scenario_seed is not None else self.training_config.seed + self.generation * 100_003, self.training_config.worlds_per_genome)
        invalid_total = 0
        action_total = 0
        market_total = 0
        for scenario_index, definition in enumerate(definitions):
            self._pulse(HeartbeatPhase.SCENARIO_EVALUATION, force=True, genome_id=genome.key, scenario_id=scenario_index, operation="scenario_started")
            world = self.scenarios.build(definition)
            farm_ids = sorted(world.farms)
            policy = RecurrentNeatFarmerPolicy(artifact, checkpoint_id="in-memory")
            controlled = FullNeatFarmerAgent(policy, f"candidate-{genome.key}")
            agents = {farm_ids[0]: controlled}
            for farm_id, opponent in zip(farm_ids[1:], self.opponents.select(definition.seed, len(farm_ids) - 1)):
                agents[farm_id] = opponent
            for index, (farm_id, agent) in enumerate(sorted(agents.items())):
                agent.reset(definition.seed + 20_000 + index, f"train-{self.generation}-{scenario_index}")
            liveness_limits = EpisodeLivenessLimits(
                maximum_world_turns=self.training_config.max_rounds_per_world,
                maximum_decisions_per_farm=self.training_config.maximum_decisions_per_farm,
                maximum_no_progress_rounds=self.training_config.maximum_no_progress_rounds,
                maximum_consecutive_passes=self.training_config.maximum_consecutive_passes,
                maximum_failed_transactions=self.training_config.maximum_failed_transactions,
                maximum_wall_clock_seconds=self.training_config.maximum_episode_wall_clock_seconds,
                cycle_history_size=self.training_config.cycle_history_size,
                cycle_repeat_limit=self.training_config.cycle_repeat_limit,
            )
            env = MultiFarmEconomyEnvironment(
                definition.seed,
                MultiFarmEconomyConfig(max_world_turns=self.training_config.max_rounds_per_world, liveness_limits=liveness_limits),
                heartbeat_callback=self._environment_heartbeat,
            )
            env.reset(initial_world=world)
            env.candidate_generator = LegalFarmerActionGenerator(
                limits=env.config.candidate_limits,
                availability=availability_for_stage(self.curriculum_stage),
                heartbeat_callback=self._environment_heartbeat,
            )
            env.observation_adapter.candidate_generator = env.candidate_generator
            initial = farm_objective_utility(env.world.farm(farm_ids[0]))
            invalid = 0
            active_farm = None
            try:
                for round_index in range(self.training_config.max_rounds_per_world):
                    self._pulse(HeartbeatPhase.WORLD_EPISODE, genome_id=genome.key, scenario_id=scenario_index, world_turn=env.world.turn, operation="round_observation")
                    selected = {}
                    for farm_id, agent in agents.items():
                        active_farm = farm_id
                        observation = env.observe(farm_id, recurrent_state=getattr(getattr(agent, "policy", None), "state", None))
                        selected[farm_id] = agent.choose_candidate(observation)
                    result = env.resolve_round(selected)
                    by_farm = {row.actor_farm_id: row for row in result.action_results}
                    for farm_id, agent in agents.items():
                        agent.observe_result(selected[farm_id], by_farm[farm_id])
                        invalid += int(not by_farm[farm_id].success and farm_id == farm_ids[0])
                    action_total += 1
                    invalid_total += int(not by_farm[farm_ids[0]].success)
                    if selected[farm_ids[0]].action.action_type.value not in {"breed_pair", "stop_breeding", "pass_turn"} and by_farm[farm_ids[0]].success:
                        market_total += 1
                    if env.terminated:
                        break
            except Exception as error:
                env._terminate(EpisodeTerminationReason.ENVIRONMENT_FAILURE)
                write_stagnation_diagnostic(
                    self.output, job_id=self.run_metadata.get("training_job_id"), generation=self.generation,
                    genome_id=genome.key, scenario_id=scenario_index, active_farm=active_farm,
                    environment=env, error=error,
                )
            diagnostic_reasons = {
                EpisodeTerminationReason.WALL_CLOCK_TIMEOUT,
                EpisodeTerminationReason.STATE_CYCLE,
                EpisodeTerminationReason.ECONOMY_STALLED,
                EpisodeTerminationReason.MAX_FAILED_TRANSACTIONS,
                EpisodeTerminationReason.ENVIRONMENT_FAILURE,
            }
            if env.termination_reason in diagnostic_reasons:
                write_stagnation_diagnostic(
                    self.output, job_id=self.run_metadata.get("training_job_id"), generation=self.generation,
                    genome_id=genome.key, scenario_id=scenario_index, active_farm=active_farm,
                    environment=env,
                )
            score = env.terminal_fitness(normalized_campaign_fitness(env.world, farm_ids[0], initial) - invalid * .25)
            scores.append(score)
            self._pulse(HeartbeatPhase.SCENARIO_EVALUATION, force=True, genome_id=genome.key, scenario_id=scenario_index, world_turn=env.world.turn, operation="scenario_completed")
        complexity = len(genome.nodes) + sum(connection.enabled for connection in genome.connections.values())
        score = sum(scores) / max(1, len(scores)) - self.training_config.complexity_penalty * complexity
        self._pulse(HeartbeatPhase.GENOME_EVALUATION, force=True, genome_id=genome.key, operation="genome_completed")
        return score, {
            "invalid_action_rate": invalid_total / max(1, action_total),
            "market_transaction_rate": market_total / max(1, action_total),
            "worlds_evaluated": len(definitions),
        }

    def _export_generation_champion(self, genome, validation_quality, row):
        generation_dir = self.output / "champions" / f"generation_{self.generation:04d}"
        generation_dir.mkdir(parents=True, exist_ok=True)
        artifact = self._artifact(genome)
        save_recurrent_artifact(artifact, generation_dir / "network.json")
        save_recurrent_artifact(artifact, generation_dir / "topology.json")
        atomic_write_json(generation_dir / "validation_metrics.json", {"validation_quality": validation_quality, **row})
        atomic_write_json(generation_dir / "full_farmer_metadata.json", {
            "run_type": "recurrent_neat_full_farmer", "generation": self.generation,
            "fitness": float(genome.fitness), "validation_quality": validation_quality,
            "curriculum_stage": self.curriculum_stage.name.lower(),
            "action_schema_version": self.action_schema.version,
            "observation_schema_version": FULL_FARMER_OBSERVATION_SCHEMA_VERSION,
            "supported_action_types": artifact.metadata["supported_action_types"],
        })
        return generation_dir / "network.json"

    def _fitness(self, genomes, neat_config):
        archive_paths = [str(self.output / path) for path in self.state.champion_archive]
        self.opponents.historical_artifacts = archive_paths[:-3]
        self.opponents.recent_artifacts = archive_paths[-3:]
        values = []
        details = []
        ordered = sorted(genomes)
        for index, (_, genome) in enumerate(ordered):
            self._cancel_if_requested()
            try:
                genome.fitness, detail = self._evaluate_genome(genome)
            except Exception as error:
                genome.fitness = -10.0; detail = {"invalid_action_rate": 1.0, "market_transaction_rate": 0.0, "worlds_evaluated": 0}
                self._emit({"event_type": "genome_evaluation_failed", "generation": self.generation, "genome_id": genome.key, "summary": f"{type(error).__name__}: {error}"})
            values.append(genome.fitness)
            details.append(detail)
            self._emit({"event_type": "genome_evaluation_progress", "generation": self.generation, "completed": index + 1, "total": len(ordered)})
        champion = max((genome for _, genome in ordered), key=lambda row: row.fitness)
        validation_quality, _ = self._evaluate_genome(champion, scenario_seed=self.training_config.seed + 90_000_001 + self.generation * 4099)
        invalid_rate = sum(row["invalid_action_rate"] for row in details) / len(details)
        market_rate = sum(row["market_transaction_rate"] for row in details) / len(details)
        row = {
            "generation": self.generation, "curriculum_stage": self.curriculum_stage.name.lower(),
            "best_fitness": max(values), "mean_fitness": sum(values) / len(values),
            "population": len(values), "worlds_evaluated": len(values) * self.training_config.worlds_per_genome,
            "validation_quality": validation_quality, "invalid_action_rate": invalid_rate,
            "market_transaction_rate": market_rate,
            "topology_complexity": len(champion.nodes) + sum(gene.enabled for gene in champion.connections.values()),
        }
        champion_path = self._export_generation_champion(champion, validation_quality, row)
        self.metrics.append(row)
        with (self.output / "generation_metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        self.state.generation = self.generation
        self.state.validation_history.append(validation_quality)
        self.state.invalid_rate_history.append(invalid_rate)
        self.state.champion_archive.append(str(champion_path.relative_to(self.output)))
        if should_advance_curriculum(self.state, self.training_config):
            self.state.curriculum_stage = type(self.state.curriculum_stage)(int(self.state.curriculum_stage) + 1)
            self.state.stage_entry_generation = self.generation + 1
            row["next_curriculum_stage"] = self.state.curriculum_stage.name.lower()
        atomic_write_json(self.output / "full_farmer_training_state.json", self.state.to_dict())
        self._emit({"event_type": "generation_completed", **row})
        self.generation += 1

    def train_population(self, population, generations, extra_reporters=()):
        self._cancel_if_requested()
        for reporter in extra_reporters:
            population.add_reporter(reporter)
        population.add_reporter(HeartbeatCheckpointer(
            self.training_config.checkpoint_frequency,
            filename_prefix=str(self.output / "checkpoints" / "neat-checkpoint-"),
            heartbeat=self._pulse,
        ))
        self.background_heartbeat.start()
        try:
            winner = population.run(self._fitness, generations)
        finally:
            self.background_heartbeat.stop()
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
        atomic_write_json(champion / "full_farmer_metadata.json", metadata)
        self.prepare_run_metadata(metadata)
        return artifact, metadata

    def train(self):
        random.seed(self.training_config.seed)
        population = neat.Population(self.neat_config, seed=self.training_config.seed)
        return self.train_population(population, self.training_config.generations)
