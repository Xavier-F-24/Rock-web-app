"""Typed cleanup inspection for durable local training jobs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .training_job_status import TERMINAL_JOB_STATES
from .worker_heartbeat import HeartbeatHealth, classify_heartbeat


@dataclass(frozen=True)
class TrainingJobCleanupRecord:
    job_id: str
    status: str
    heartbeat_health: HeartbeatHealth
    process_id: int | None
    process_alive: bool
    cleanup_eligible: bool
    reason: str
    heartbeat_age_seconds: float | None
    output_run: str | None


def inspect_training_job(status, *, process_alive: bool, stagnant_seconds: float = 120.0):
    age = None
    if status.last_heartbeat_time:
        heartbeat = datetime.fromisoformat(status.last_heartbeat_time)
        age = max(0.0, (datetime.now(timezone.utc) - heartbeat).total_seconds())
    if status.status in TERMINAL_JOB_STATES:
        health = HeartbeatHealth.FAILED if status.status.value == "failed" else HeartbeatHealth.HEALTHY
        eligible = not process_alive
        reason = (
            f"terminal_{status.status.value}"
            if eligible
            else "terminal_worker_still_exiting"
        )
    elif not process_alive:
        health = HeartbeatHealth.ORPHANED
        eligible = True
        reason = "worker_process_missing"
    else:
        health = classify_heartbeat(
            age or 0.0,
            process_alive=True,
            stagnant_seconds=stagnant_seconds,
            slow_seconds=min(30.0, stagnant_seconds / 2),
        )
        eligible = False
        reason = (
            "live_worker_stagnant_request_cancellation"
            if health is HeartbeatHealth.STAGNANT
            else "live_worker"
        )
    return TrainingJobCleanupRecord(
        job_id=status.job_id,
        status=status.status.value,
        heartbeat_health=health,
        process_id=status.process_id,
        process_alive=process_alive,
        cleanup_eligible=eligible,
        reason=reason,
        heartbeat_age_seconds=age,
        output_run=status.output_run,
    )
