import pytest

from Rock_AI.training_jobs.training_job_lock import TrainingJobLock


def test_writer_lock_rejects_second_owner(tmp_path):
    first = TrainingJobLock(tmp_path / "lock", "one"); first.acquire()
    try:
        with pytest.raises(RuntimeError, match="already held"):
            TrainingJobLock(tmp_path / "lock", "two").acquire()
    finally:
        first.release()


def test_lock_releases_cleanly(tmp_path):
    path = tmp_path / "lock"
    with TrainingJobLock(path, "one"): assert path.exists()
    assert not path.exists()
