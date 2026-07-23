"""Explicit training worker lifecycle and atomic status records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class TrainingJobState(str, Enum):
    CREATED = "created"
    VALIDATING = "validating"
    QUEUED = "queued"
    STARTING = "starting"
    RUNNING = "running"
    CHECKPOINTING = "checkpointing"
    CANCELLATION_REQUESTED = "cancellation_requested"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"
    ORPHANED = "orphaned"


TERMINAL_JOB_STATES = {TrainingJobState.CANCELLED, TrainingJobState.COMPLETED, TrainingJobState.FAILED, TrainingJobState.ORPHANED}

LEGAL_TRANSITIONS = {
    TrainingJobState.CREATED: {TrainingJobState.VALIDATING, TrainingJobState.CANCELLED},
    TrainingJobState.VALIDATING: {TrainingJobState.QUEUED, TrainingJobState.FAILED, TrainingJobState.CANCELLED},
    TrainingJobState.QUEUED: {TrainingJobState.STARTING, TrainingJobState.CANCELLATION_REQUESTED, TrainingJobState.FAILED},
    TrainingJobState.STARTING: {TrainingJobState.RUNNING, TrainingJobState.CANCELLATION_REQUESTED, TrainingJobState.FAILED},
    TrainingJobState.RUNNING: {TrainingJobState.CHECKPOINTING, TrainingJobState.CANCELLATION_REQUESTED, TrainingJobState.COMPLETED, TrainingJobState.FAILED, TrainingJobState.ORPHANED},
    TrainingJobState.CHECKPOINTING: {TrainingJobState.RUNNING, TrainingJobState.CANCELLATION_REQUESTED, TrainingJobState.COMPLETED, TrainingJobState.FAILED},
    TrainingJobState.CANCELLATION_REQUESTED: {TrainingJobState.CANCELLED, TrainingJobState.FAILED},
    TrainingJobState.ORPHANED: {TrainingJobState.FAILED},
    TrainingJobState.CANCELLED: set(), TrainingJobState.COMPLETED: set(), TrainingJobState.FAILED: set(),
}


@dataclass(frozen=True)
class TrainingJobStatus:
    job_id: str
    status: TrainingJobState
    operation_type: str
    source_run: str | None = None
    source_checkpoint: str | None = None
    source_generation: int | None = None
    source_champion: str | None = None
    output_run: str | None = None
    current_evolutionary_generation: int = 0
    starting_generation: int = 0
    requested_ending_generation: int = 0
    completed_generations: int = 0
    current_curriculum_stage: str | None = None
    current_species_count: int | None = None
    current_best_training_fitness: float | None = None
    current_best_validation_fitness: float | None = None
    champion_topology_size: int | None = None
    last_heartbeat_time: str | None = None
    process_id: int | None = None
    start_time: str | None = None
    end_time: str | None = None
    latest_checkpoint: str | None = None
    latest_safe_champion_export: str | None = None
    failure_summary: str | None = None
    cancellation_state: str | None = None
    warnings: tuple[str, ...] = ()
    trainer_kind: str = "breeding_pair"
    worlds_evaluated: int = 0
    invalid_action_rate: float | None = None
    market_transaction_rate: float | None = None
    heartbeat_health: str = "healthy"
    heartbeat_phase: str | None = None
    current_genome_id: str | None = None
    current_scenario_id: str | None = None
    current_world_turn: int | None = None
    last_completed_operation: str | None = None

    def transition(self, status: TrainingJobState, **changes: Any) -> "TrainingJobStatus":
        if status != self.status and status not in LEGAL_TRANSITIONS[self.status]:
            raise ValueError(f"Illegal training job transition: {self.status.value} -> {status.value}")
        payload = asdict(self); payload.update(changes); payload["status"] = status
        return TrainingJobStatus(**payload)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self); payload["status"] = self.status.value
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingJobStatus":
        payload = dict(data); payload["status"] = TrainingJobState(payload["status"]); payload["warnings"] = tuple(payload.get("warnings", ()))
        return cls(**payload)
