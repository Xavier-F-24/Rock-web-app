import json
from datetime import datetime, timedelta, timezone

import pytest

from Rock_AI.training_jobs.training_job_manager import TrainingJobManager
from Rock_AI.training_jobs.training_job_status import TrainingJobState, TrainingJobStatus
from Rock_AI.training_jobs.training_progress_reader import TrainingProgressReader, atomic_write_json


class FakeLauncher:
    def __init__(self, alive=()):
        self.alive = set(alive)

    def process_exists(self, pid):
        return int(pid) in self.alive


def _job(jobs, job_id, status):
    directory = jobs / job_id
    directory.mkdir(parents=True)
    atomic_write_json(directory / "status.json", status.to_dict())
    return directory


def test_terminal_job_deletion_preserves_run_and_releases_owned_lock(tmp_path):
    jobs = tmp_path / "training_jobs"
    run = tmp_path / "training_runs" / "kept_run"
    run.mkdir(parents=True)
    (run / "champion.json").write_text("{}", encoding="utf-8")
    lock = run.parent / f".{run.name}.training_writer_lock"
    lock.mkdir()
    (lock / "owner.json").write_text(
        json.dumps({"owner": "job_done", "pid": 991}),
        encoding="utf-8",
    )
    status = TrainingJobStatus(
        "job_done",
        TrainingJobState.COMPLETED,
        "continue",
        output_run=str(run),
        process_id=991,
    )
    directory = _job(jobs, "job_done", status)
    manager = TrainingJobManager(tmp_path, jobs, launcher=FakeLauncher())

    result = manager.delete_job_record("job_done")

    assert result["training_run_preserved"]
    assert result["released_output_lock"]
    assert not directory.exists()
    assert (run / "champion.json").exists()
    assert not lock.exists()


def test_live_worker_cannot_be_deleted_and_stagnant_worker_can_be_cancelled(tmp_path):
    jobs = tmp_path / "training_jobs"
    old = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    status = TrainingJobStatus(
        "job_live",
        TrainingJobState.RUNNING,
        "continue",
        process_id=992,
        last_heartbeat_time=old,
    )
    directory = _job(jobs, "job_live", status)
    manager = TrainingJobManager(tmp_path, jobs, launcher=FakeLauncher({992}))

    record = manager.inspect_cleanup("job_live", stagnant_seconds=30)
    assert record.process_alive
    assert record.heartbeat_health.value == "stagnant"
    assert not record.cleanup_eligible
    with pytest.raises(RuntimeError, match="live_worker_stagnant"):
        manager.delete_job_record("job_live")

    marker = manager.request_cancel("job_live")
    assert marker.exists()
    assert directory.exists()


def test_missing_worker_is_cleanup_ready_and_listed(tmp_path):
    jobs = tmp_path / "training_jobs"
    status = TrainingJobStatus(
        "job_orphan",
        TrainingJobState.RUNNING,
        "continue",
        process_id=993,
        last_heartbeat_time=datetime.now(timezone.utc).isoformat(),
    )
    _job(jobs, "job_orphan", status)
    manager = TrainingJobManager(tmp_path, jobs, launcher=FakeLauncher())

    records = manager.cleanup_candidates()

    assert len(records) == 1
    assert records[0].job_id == "job_orphan"
    assert records[0].cleanup_eligible
    assert records[0].reason == "worker_process_missing"


def test_cleanup_scan_skips_unreadable_job_without_losing_other_records(
    tmp_path,
    monkeypatch,
):
    jobs = tmp_path / "training_jobs"
    status = TrainingJobStatus(
        "job_visible",
        TrainingJobState.RUNNING,
        "continue",
        process_id=994,
        last_heartbeat_time=datetime.now(timezone.utc).isoformat(),
    )
    _job(jobs, "job_visible", status)
    blocked = _job(
        jobs,
        "job_blocked",
        TrainingJobStatus("job_blocked", TrainingJobState.COMPLETED, "continue"),
    )
    original = TrainingProgressReader.status

    def guarded_status(self):
        if self.job_directory == blocked:
            raise PermissionError("status is locked")
        return original(self)

    monkeypatch.setattr(TrainingProgressReader, "status", guarded_status)
    manager = TrainingJobManager(tmp_path, jobs, launcher=FakeLauncher())

    records = manager.cleanup_candidates()

    assert [record.job_id for record in records] == ["job_visible"]
    assert manager.cleanup_scan_errors == ({
        "job_id": "job_blocked",
        "error": "status is locked",
    },)
