"""Authoritative implementation of one manifest-driven local training job."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from Rock_AI.training.neat_progress_reporter import NeatProgressReporter
from Rock_AI.training.action_curriculum import ActionCurriculumStage
from Rock_AI.training.full_farmer_neat_trainer import FullFarmerNeatTrainer
from Rock_AI.training.full_farmer_training_config import FullFarmerTrainingConfig
from Rock_AI.policies.recurrent_neat_farmer_policy import FULL_FARMER_OBSERVATION_SCHEMA_VERSION
from Rock_AI.training.recurrent_neat_training_helper import (
    RecurrentNeatTrainer, RecurrentNeatTrainingConfig, TrainingCancelled,
)
from .champion_branch_builder import build_champion_branch_population
from .run_continuation_helper import resolve_checkpoint, restore_population, validate_run_compatibility
from .training_job_config import TrainingBackendKind, TrainingOperation
from .training_job_lock import TrainingJobLock
from .training_job_manifest import TrainingJobManifest
from .training_job_status import TrainingJobState, TrainingJobStatus
from .training_progress_reader import TrainingProgressReader, atomic_write_json


def _now(): return datetime.now(timezone.utc).isoformat()


def _append_event(path: Path, event_type: str, **payload):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"event_type": event_type, "timestamp": _now(), **payload}, sort_keys=True) + "\n"); stream.flush()


def _latest(path: Path, pattern: str) -> str | None:
    def key(row):
        try: return (0, int(row.name.rsplit("-", 1)[-1]))
        except ValueError: return (1, row.as_posix())
    rows = sorted(path.glob(pattern), key=key)
    return str(rows[-1]) if rows else None


class WorkerReporterStatusCallback:
    """Pickle-safe reporter callback retained inside trusted NEAT checkpoints."""
    def __init__(self, job_directory: str | Path):
        self.job_directory = str(job_directory)

    def __call__(self, event):
        if event.get("event_type") != "generation_evaluated": return
        job = Path(self.job_directory)
        status = TrainingProgressReader(job).status()
        status = replace(
            status, current_species_count=event.get("species_count"),
            current_best_training_fitness=event.get("best_fitness"),
            champion_topology_size=(event.get("best_genome_node_count", 0) + event.get("best_genome_connection_count", 0)),
            last_heartbeat_time=_now(),
        )
        atomic_write_json(job / "status.json", status.to_dict())


def run_training_job(job_directory: str | Path) -> TrainingJobStatus:
    job = Path(job_directory).resolve()
    manifest = TrainingJobManifest.from_dict(json.loads((job / "job_manifest.json").read_text(encoding="utf-8")))
    config = manifest.config
    root = Path(manifest.repository_root).resolve(); config.validate_paths(root)
    source_run = (root / config.source_run).resolve() if config.source_run and not Path(config.source_run).is_absolute() else Path(config.source_run).resolve() if config.source_run else root
    output_run = (root / config.output_run).resolve() if not Path(config.output_run).is_absolute() else Path(config.output_run).resolve()
    dataset = config.dataset_path
    status = TrainingProgressReader(job).status()

    def write_status(new_status):
        nonlocal status
        status = new_status
        atomic_write_json(job / "status.json", status.to_dict())

    run_lock_path = output_run.parent / f".{output_run.name}.training_writer_lock"
    with TrainingJobLock(job / "lock", manifest.job_id), TrainingJobLock(run_lock_path, manifest.job_id):
        try:
            if status.status is TrainingJobState.CREATED: status = status.transition(TrainingJobState.VALIDATING)
            write_status(status)
            status = status.transition(TrainingJobState.QUEUED); write_status(status)
            if (job / "cancel.request").exists():
                status = status.transition(TrainingJobState.CANCELLATION_REQUESTED)
                status = status.transition(TrainingJobState.CANCELLED, end_time=_now(), cancellation_state="before_start")
                write_status(status); return status
            status = status.transition(TrainingJobState.STARTING, process_id=os.getpid(), start_time=_now(), last_heartbeat_time=_now()); write_status(status)
            (job / "worker.pid").write_text(str(os.getpid()), encoding="ascii")
            source_training = None; population = None; starting_generation = 0
            if config.operation in {TrainingOperation.CONTINUE, TrainingOperation.CONTINUE_AS_BRANCH}:
                requested_checkpoint = config.source_checkpoint
                if requested_checkpoint and requested_checkpoint != "latest" and not Path(requested_checkpoint).is_absolute():
                    root_relative = (root / requested_checkpoint).resolve()
                    if root_relative.exists():
                        requested_checkpoint = str(root_relative)
                checkpoint = resolve_checkpoint(source_run, requested_checkpoint)
                expected_type = "recurrent_neat_full_farmer" if config.trainer_kind is TrainingBackendKind.FULL_FARMER else "recurrent_neat_player_farmer"
                compatibility = validate_run_compatibility(source_run, checkpoint, expected_run_type=expected_type)
                source_training = compatibility["training_config"]
                if config.operation is TrainingOperation.CONTINUE_AS_BRANCH:
                    if output_run.exists() and any(output_run.iterdir()): raise FileExistsError("Continuation branch output already exists")
                    output_run.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_run / "neat_config.ini", output_run / "neat_config.ini")
                    if config.trainer_kind is TrainingBackendKind.FULL_FARMER:
                        for name in ("training_config.json", "action_schema.json", "observation_schema.json", "full_farmer_training_state.json"):
                            source_file = source_run / name
                            if source_file.exists():
                                shutil.copy2(source_file, output_run / name)
                population = restore_population(checkpoint)
                starting_generation = int(population.generation)
                dataset = dataset or source_training.get("dataset_path")
            elif config.operation is TrainingOperation.BRANCH_CHAMPION:
                if output_run.exists() and any(output_run.iterdir()): raise FileExistsError("Champion branch output already exists")
                source_training = json.loads((source_run / "training_config.json").read_text(encoding="utf-8"))
                dataset = dataset or source_training.get("dataset_path")
            allow_existing = config.operation in {TrainingOperation.CONTINUE, TrainingOperation.CONTINUE_AS_BRANCH}
            if config.trainer_kind is TrainingBackendKind.FULL_FARMER:
                training = FullFarmerTrainingConfig(
                    output_directory=str(output_run), seed=config.seed, population=config.population_size,
                    generations=config.additional_generations, worlds_per_genome=config.worlds_per_genome,
                    max_rounds_per_world=config.max_rounds_per_world,
                    maximum_decisions_per_farm=config.maximum_decisions_per_farm,
                    maximum_no_progress_rounds=config.maximum_no_progress_rounds,
                    maximum_consecutive_passes=config.maximum_consecutive_passes,
                    maximum_failed_transactions=config.maximum_failed_transactions,
                    maximum_episode_wall_clock_seconds=config.maximum_episode_wall_clock_seconds,
                    heartbeat_interval_seconds=config.heartbeat_interval_seconds,
                    curriculum_start=ActionCurriculumStage[config.curriculum_start.upper()],
                    checkpoint_frequency=config.checkpoint_frequency,
                    showcase_frequency=config.showcase_frequency,
                    settling_steps=int((source_training or {}).get("settling_steps", 3)),
                    complexity_penalty=config.complexity_penalty,
                    curriculum_max=ActionCurriculumStage[config.curriculum_max.upper()],
                    minimum_generations_per_stage=config.minimum_generations_per_stage,
                    curriculum_stability_window=config.curriculum_stability_window,
                    curriculum_invalid_rate_threshold=config.curriculum_invalid_rate_threshold,
                    curriculum_validation_threshold=config.curriculum_validation_threshold,
                )
                trainer = FullFarmerNeatTrainer(
                    training, allow_existing=allow_existing, starting_generation=starting_generation,
                    cancel_path=job / "cancel.request",
                )
                expected_schema = FULL_FARMER_OBSERVATION_SCHEMA_VERSION
            else:
                if not dataset: raise ValueError("Training dataset path is missing")
                training = RecurrentNeatTrainingConfig(
                    dataset_path=str((root / dataset).resolve() if not Path(dataset).is_absolute() else Path(dataset).resolve()),
                    output_directory=str(output_run), seed=config.seed, population=config.population_size,
                    generations=config.additional_generations, training_scenarios_per_generation=config.training_scenarios,
                    validation_scenarios=config.validation_scenarios, checkpoint_frequency=config.checkpoint_frequency,
                    campaign_generations=config.campaign_generations, supervised_weight=config.supervised_weight,
                    campaign_weight=config.campaign_weight, complexity_penalty=config.complexity_penalty,
                    settling_steps=int((source_training or {}).get("settling_steps", 3)),
                    heartbeat_interval_seconds=config.heartbeat_interval_seconds,
                )
                trainer = RecurrentNeatTrainer(
                    training, allow_existing=allow_existing, starting_generation=starting_generation,
                    cancel_path=job / "cancel.request",
                )
                expected_schema = int(trainer.manifest["observation_schema_version"])
            trainer.prepare_run_metadata({
                "training_job_id": manifest.job_id,
                "training_operation": config.operation.value,
                "parent_run": str(source_run) if config.operation is not TrainingOperation.CONTINUE else None,
            })
            if population is not None:
                if population.config.genome_config.num_inputs != len(trainer.feature_names) or population.config.genome_config.num_outputs != 3:
                    raise ValueError("Resumable checkpoint observation or output schema is incompatible")
            if config.operation is TrainingOperation.BRANCH_CHAMPION:
                historical = sorted(source_run.glob("champions/generation_*/network.json"))[:-1]
                population, branch = build_champion_branch_population(
                    replace(config, source_champion=str((root / config.source_champion).resolve() if not Path(config.source_champion).is_absolute() else Path(config.source_champion).resolve())),
                    trainer.neat_config,
                    expected_schema=expected_schema,
                    expected_features=trainer.feature_names,
                    historical_champions=historical,
                )
                atomic_write_json(output_run / "branch_manifest.json", branch.to_dict())
            status = status.transition(
                TrainingJobState.RUNNING, starting_generation=starting_generation,
                current_evolutionary_generation=starting_generation,
                requested_ending_generation=starting_generation + config.additional_generations - 1,
                output_run=str(output_run), source_checkpoint=str(config.source_checkpoint or ""),
            ); write_status(status)
            _append_event(job / "progress.jsonl", "job_started", operation=config.operation.value, starting_generation=starting_generation)

            def progress_update(event, *, append=True):
                nonlocal status
                if append:
                    _append_event(job / "progress.jsonl", **event)
                if event.get("event_type") == "worker_heartbeat":
                    status = TrainingProgressReader(job).status()
                    status = replace(
                        status,
                        last_heartbeat_time=event.get("heartbeat_time", _now()),
                        heartbeat_health=event.get("health", "healthy"),
                        heartbeat_phase=event.get("phase"),
                        current_genome_id=None if event.get("genome_id") is None else str(event.get("genome_id")),
                        current_scenario_id=None if event.get("scenario_id") is None else str(event.get("scenario_id")),
                        current_world_turn=event.get("world_turn"),
                        last_completed_operation=event.get("operation"),
                        operation_progress_current=int(event.get("progress_current", status.operation_progress_current)),
                        operation_progress_total=int(event.get("progress_total", status.operation_progress_total)),
                        operation_progress_label=event.get("progress_label", status.operation_progress_label),
                    )
                    write_status(status)
                if event.get("event_type") == "genome_evaluation_progress":
                    status = TrainingProgressReader(job).status()
                    status = replace(
                        status, last_heartbeat_time=_now(),
                        operation_progress_current=int(event.get("genomes_evaluated", status.operation_progress_current)),
                        operation_progress_total=int(event.get("genomes_total", status.operation_progress_total)),
                        operation_progress_label="Genome evaluation",
                    )
                    write_status(status)
                if event.get("event_type") == "generation_completed":
                    status = TrainingProgressReader(job).status()
                    generation = int(event["generation"])
                    status = replace(
                        status, current_evolutionary_generation=generation,
                        completed_generations=generation - starting_generation + 1,
                        current_curriculum_stage=event.get("curriculum_stage"),
                        current_best_training_fitness=event.get("best_fitness"),
                        current_best_validation_fitness=event.get("validation_quality"),
                        champion_topology_size=event.get("topology_complexity"),
                        worlds_evaluated=int(event.get("worlds_evaluated", status.worlds_evaluated)),
                        invalid_action_rate=event.get("invalid_action_rate"),
                        market_transaction_rate=event.get("market_transaction_rate"),
                        operation_progress_current=0,
                        operation_progress_total=0,
                        operation_progress_label=None,
                        last_heartbeat_time=_now(),
                        latest_checkpoint=_latest(output_run, "checkpoints/neat-checkpoint-*"),
                        latest_safe_champion_export=_latest(output_run, "champions/generation_*/network.json"),
                    )
                    write_status(status)

            trainer.progress_callback = progress_update
            reporter = NeatProgressReporter(job / "progress.jsonl", event_callback=WorkerReporterStatusCallback(job))
            if population is None:
                import neat
                population = neat.Population(trainer.neat_config, seed=config.seed)
            trainer.train_population(population, config.additional_generations, extra_reporters=(reporter,))
            if (job / "cancel.request").exists(): raise TrainingCancelled("Cancellation observed after generation")
            status = TrainingProgressReader(job).status()
            status = status.transition(
                TrainingJobState.COMPLETED, end_time=_now(), last_heartbeat_time=_now(),
                latest_checkpoint=_latest(output_run, "checkpoints/neat-checkpoint-*"),
                latest_safe_champion_export=_latest(output_run, "champions/generation_*/network.json"),
            ); write_status(status)
            atomic_write_json(job / "output_run_reference.json", {"output_run": str(output_run), "final_generation": status.current_evolutionary_generation, "latest_champion": status.latest_safe_champion_export})
            atomic_write_json(job / "complete.json", {"completed_at": _now(), "status": status.to_dict()})
            _append_event(job / "progress.jsonl", "training_completed", final_generation=status.current_evolutionary_generation)
            return status
        except TrainingCancelled as error:
            if status.status is not TrainingJobState.CANCELLATION_REQUESTED:
                status = status.transition(TrainingJobState.CANCELLATION_REQUESTED)
            status = status.transition(
                TrainingJobState.CANCELLED, end_time=_now(), cancellation_state="cooperative",
                latest_checkpoint=_latest(output_run, "checkpoints/neat-checkpoint-*"),
                latest_safe_champion_export=_latest(output_run, "champions/generation_*/network.json"),
            ); write_status(status); _append_event(job / "progress.jsonl", "cancellation_observed", summary=str(error)); return status
        except Exception as error:
            if status.status in {TrainingJobState.CREATED, TrainingJobState.VALIDATING, TrainingJobState.QUEUED, TrainingJobState.STARTING, TrainingJobState.RUNNING, TrainingJobState.CHECKPOINTING, TrainingJobState.CANCELLATION_REQUESTED}:
                status = status.transition(TrainingJobState.FAILED, end_time=_now(), failure_summary=f"{type(error).__name__}: {error}", heartbeat_health="failed")
            write_status(status); atomic_write_json(job / "failure.json", {"failed_at": _now(), "error_type": type(error).__name__, "summary": str(error)}); _append_event(job / "progress.jsonl", "training_failed", summary=str(error)); raise
