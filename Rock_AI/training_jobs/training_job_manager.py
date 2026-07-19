"""Create, launch, cancel, inspect, and recover immutable training jobs."""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .training_job_config import TrainingJobConfig, TrainingOperation
from .training_job_manifest import TrainingJobManifest
from .training_job_status import TERMINAL_JOB_STATES, TrainingJobState, TrainingJobStatus
from .training_process_launcher import TrainingProcessLauncher
from .training_progress_reader import TrainingProgressReader, atomic_write_json


class TrainingJobManager:
    def __init__(self, repository_root: str | Path, jobs_root: str | Path | None = None, launcher=None):
        self.repository_root = Path(repository_root).resolve()
        self.jobs_root = Path(jobs_root).resolve() if jobs_root else self.repository_root / "training_jobs"
        self.launcher = launcher or TrainingProcessLauncher()

    def create_job(self, config: TrainingJobConfig, *, job_id: str | None = None) -> TrainingJobManifest:
        config.validate_paths(self.repository_root)
        identifier = job_id or f"job_{uuid.uuid4().hex}"
        if not identifier.startswith("job_") or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in identifier):
            raise ValueError("Invalid job ID")
        directory = self.jobs_root / identifier
        directory.mkdir(parents=True, exist_ok=False)
        command = (sys.executable, "-m", "Rock_AI.scripts.run_neat_training_job", "--job", str(directory))
        manifest = TrainingJobManifest(identifier, datetime.now(timezone.utc).isoformat(), str(self.repository_root), str(directory), config, command)
        atomic_write_json(directory / "job_manifest.json", manifest.to_dict())
        atomic_write_json(directory / "job_config.json", config.to_dict())
        status = TrainingJobStatus(
            identifier, TrainingJobState.CREATED, config.operation.value,
            config.source_run, config.source_checkpoint, config.source_generation,
            config.source_champion, config.output_run, 0, 0, config.additional_generations,
            trainer_kind=config.trainer_kind.value,
        )
        atomic_write_json(directory / "status.json", status.to_dict())
        return manifest

    def launch(self, job_id: str) -> int:
        directory = self.jobs_root / job_id
        reader = TrainingProgressReader(directory); status = reader.status()
        if status.status in TERMINAL_JOB_STATES: raise ValueError(f"Cannot launch terminal job {status.status.value}")
        pid_path = directory / "worker.pid"
        if pid_path.exists():
            pid = int(pid_path.read_text(encoding="ascii"))
            if self.launcher.process_exists(pid): return pid
        current = reader.status()
        if current.status is TrainingJobState.CREATED:
            current = current.transition(TrainingJobState.VALIDATING)
            current = current.transition(TrainingJobState.QUEUED)
            atomic_write_json(directory / "status.json", current.to_dict())
        return self.launcher.launch(directory)

    def request_cancel(self, job_id: str) -> Path:
        directory = self.jobs_root / job_id
        marker = directory / "cancel.request"
        marker.write_text(datetime.now(timezone.utc).isoformat(), encoding="ascii")
        status = TrainingProgressReader(directory).status()
        if status.status not in TERMINAL_JOB_STATES and TrainingJobState.CANCELLATION_REQUESTED in __import__("Rock_AI.training_jobs.training_job_status", fromlist=["LEGAL_TRANSITIONS"]).LEGAL_TRANSITIONS[status.status]:
            atomic_write_json(directory / "status.json", status.transition(TrainingJobState.CANCELLATION_REQUESTED, cancellation_state="requested").to_dict())
        return marker

    def recover(self, job_id: str, *, new_output_run: str | None = None) -> TrainingJobManifest:
        directory = self.jobs_root / job_id
        manifest = TrainingJobManifest.from_dict(json.loads((directory / "job_manifest.json").read_text(encoding="utf-8")))
        status = TrainingProgressReader(directory).status()
        checkpoint = status.latest_checkpoint
        if not checkpoint: raise ValueError("Job has no resumable checkpoint")
        payload = manifest.config.to_dict()
        payload.update({"operation": TrainingOperation.CONTINUE_AS_BRANCH.value, "source_checkpoint": checkpoint, "output_run": new_output_run or f"{manifest.config.output_run}_recovery_{uuid.uuid4().hex[:8]}"})
        return self.create_job(TrainingJobConfig.from_dict(payload))
