import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from Rock_AI.training_jobs.training_job_status import TrainingJobState, TrainingJobStatus
from Rock_AI.training_jobs.training_progress_reader import TrainingProgressReader, atomic_write_json


def test_atomic_status_round_trip(tmp_path):
    status = TrainingJobStatus("job_x", TrainingJobState.CREATED, "branch_champion")
    atomic_write_json(tmp_path / "status.json", status.to_dict())
    assert TrainingProgressReader(tmp_path).status() == status


def test_status_read_retries_transient_permission_error(tmp_path, monkeypatch):
    status = TrainingJobStatus("job_x", TrainingJobState.RUNNING, "continue")
    path = tmp_path / "status.json"
    atomic_write_json(path, status.to_dict())
    original = Path.read_text
    attempts = 0

    def flaky_read_text(self, *args, **kwargs):
        nonlocal attempts
        if self == path and attempts < 2:
            attempts += 1
            raise PermissionError("temporary Windows lock")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", flaky_read_text)

    assert TrainingProgressReader(tmp_path).status() == status
    assert attempts == 2


def test_partial_progress_line_is_ignored(tmp_path):
    (tmp_path / "progress.jsonl").write_text('{"generation": 1}\n{"broken"', encoding="utf-8")
    assert TrainingProgressReader(tmp_path).progress() == [{"generation": 1}]


def test_stale_heartbeat_warns_without_mutating_status(tmp_path):
    old = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    status = TrainingJobStatus("job_x", TrainingJobState.RUNNING, "continue", last_heartbeat_time=old)
    atomic_write_json(tmp_path / "status.json", status.to_dict())
    reader = TrainingProgressReader(tmp_path)
    assert "stale" in reader.orphan_warning(30)
    assert reader.status().status is TrainingJobState.RUNNING


def test_process_probe_is_safe_for_current_and_missing_processes():
    assert TrainingProgressReader._process_alive(os.getpid())
    assert isinstance(TrainingProgressReader._process_alive(2_147_483_647), bool)
