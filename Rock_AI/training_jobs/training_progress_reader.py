"""Safe, tolerant reads of worker status and progress artifacts."""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .training_job_status import TERMINAL_JOB_STATES, TrainingJobState, TrainingJobStatus


def atomic_write_json(path: str | Path, payload: dict) -> None:
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    try:
        for attempt in range(20):
            try:
                os.replace(temporary, destination)
                return
            except PermissionError:
                if attempt == 19: raise
                time.sleep(0.05)
    finally:
        temporary.unlink(missing_ok=True)


class TrainingProgressReader:
    def __init__(self, job_directory: str | Path): self.job_directory = Path(job_directory)
    def status(self) -> TrainingJobStatus:
        return TrainingJobStatus.from_dict(json.loads((self.job_directory / "status.json").read_text(encoding="utf-8")))
    def progress(self) -> list[dict]:
        path = self.job_directory / "progress.jsonl"
        if not path.exists(): return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try: rows.append(json.loads(line))
            except json.JSONDecodeError: continue
        return rows
    def console_tail(self, lines: int = 200) -> str:
        path = self.job_directory / "console.log"
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]) if path.exists() else ""
    def orphan_warning(self, stale_seconds: float = 120.0) -> str | None:
        status = self.status()
        if status.status in TERMINAL_JOB_STATES or not status.last_heartbeat_time: return None
        heartbeat = datetime.fromisoformat(status.last_heartbeat_time)
        age = (datetime.now(timezone.utc) - heartbeat).total_seconds()
        return f"Worker heartbeat is stale by {age:.0f} seconds" if age > stale_seconds else None
