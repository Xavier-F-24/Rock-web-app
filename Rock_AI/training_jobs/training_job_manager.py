"""Create, launch, cancel, inspect, and recover immutable training jobs."""

from __future__ import annotations

import json
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .training_job_config import TrainingJobConfig, TrainingOperation
from .training_job_manifest import TrainingJobManifest
from .training_job_status import TERMINAL_JOB_STATES, TrainingJobState, TrainingJobStatus
from .training_process_launcher import TrainingProcessLauncher
from .training_progress_reader import TrainingProgressReader, atomic_write_json
from .training_job_cleanup_helper import inspect_training_job


class TrainingJobManager:
    def __init__(self, repository_root: str | Path, jobs_root: str | Path | None = None, launcher=None):
        self.repository_root = Path(repository_root).resolve()
        self.jobs_root = Path(jobs_root).resolve() if jobs_root else self.repository_root / "training_jobs"
        self.launcher = launcher or TrainingProcessLauncher()
        self.cleanup_scan_errors: tuple[dict[str, str], ...] = ()

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

    def inspect_cleanup(self, job_id: str, *, stagnant_seconds: float = 120.0):
        directory = self._job_directory(job_id)
        status = TrainingProgressReader(directory).status()
        process_alive = bool(status.process_id and self.launcher.process_exists(int(status.process_id)))
        return inspect_training_job(
            status,
            process_alive=process_alive,
            stagnant_seconds=stagnant_seconds,
        )

    def cleanup_candidates(self, *, stagnant_seconds: float = 120.0):
        rows = []
        errors = []
        if not self.jobs_root.exists():
            self.cleanup_scan_errors = ()
            return tuple(rows)
        for directory in sorted(self.jobs_root.glob("job_*"), key=lambda path: path.name):
            if not directory.is_dir() or not (directory / "status.json").exists():
                continue
            try:
                status = TrainingProgressReader(directory).status()
            except (OSError, ValueError, json.JSONDecodeError) as error:
                errors.append({
                    "job_id": directory.name,
                    "error": str(error),
                })
                continue
            process_alive = (
                False
                if status.status in TERMINAL_JOB_STATES
                else bool(status.process_id and self.launcher.process_exists(int(status.process_id)))
            )
            record = inspect_training_job(
                status,
                process_alive=process_alive,
                stagnant_seconds=stagnant_seconds,
            )
            if record.cleanup_eligible or record.heartbeat_health.value in {"slow", "stagnant", "orphaned", "failed"}:
                rows.append(record)
        self.cleanup_scan_errors = tuple(errors)
        return tuple(rows)

    def delete_job_record(self, job_id: str) -> dict[str, object]:
        """Delete one inactive job record while preserving its training run."""
        directory = self._job_directory(job_id)
        record = self.inspect_cleanup(job_id)
        if not record.cleanup_eligible:
            raise RuntimeError(f"Cannot delete job {job_id}: {record.reason}")
        released_lock = self._release_owned_output_lock(record.output_run, job_id)
        shutil.rmtree(directory)
        return {
            "job_id": job_id,
            "job_record_deleted": True,
            "training_run_preserved": True,
            "released_output_lock": released_lock,
        }

    def _job_directory(self, job_id: str) -> Path:
        identifier = str(job_id)
        if not identifier.startswith("job_") or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for character in identifier
        ):
            raise ValueError("Invalid job ID")
        directory = (self.jobs_root / identifier).resolve()
        if directory.parent != self.jobs_root.resolve() or not directory.is_dir():
            raise ValueError(f"Unknown training job: {identifier}")
        return directory

    def _release_owned_output_lock(self, output_run: str | None, job_id: str) -> bool:
        if not output_run:
            return False
        run = Path(output_run)
        run = run.resolve() if run.is_absolute() else (self.repository_root / run).resolve()
        if run != self.repository_root and self.repository_root not in run.parents:
            return False
        lock = run.parent / f".{run.name}.training_writer_lock"
        owner_path = lock / "owner.json"
        if not owner_path.exists():
            return False
        try:
            owner = json.loads(owner_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if owner.get("owner") != job_id:
            return False
        shutil.rmtree(lock)
        return True
