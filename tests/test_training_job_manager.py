from pathlib import Path

from Rock_AI.training_jobs import TrainingJobConfig, TrainingJobManager, TrainingOperation


class FakeLauncher:
    def __init__(self): self.calls = 0; self.running = set()
    def launch(self, directory):
        self.calls += 1; self.running.add(4242); Path(directory, "worker.pid").write_text("4242", encoding="ascii"); return 4242
    def process_exists(self, pid): return pid in self.running


def _config():
    return TrainingJobConfig(operation=TrainingOperation.BRANCH_CHAMPION, source_run="training_runs/source", output_run="training_runs/output", additional_generations=1, seed=3, source_champion="training_runs/source/network.json")


def test_jobs_receive_unique_directories(tmp_path):
    manager = TrainingJobManager(tmp_path, jobs_root=tmp_path / "jobs", launcher=FakeLauncher())
    left = manager.create_job(_config()); right = manager.create_job(_config())
    assert left.job_id != right.job_id
    assert Path(left.job_directory).is_dir() and Path(right.job_directory).is_dir()


def test_repeated_launch_is_idempotent(tmp_path):
    launcher = FakeLauncher(); manager = TrainingJobManager(tmp_path, jobs_root=tmp_path / "jobs", launcher=launcher)
    manifest = manager.create_job(_config())
    assert manager.launch(manifest.job_id) == manager.launch(manifest.job_id) == 4242
    assert launcher.calls == 1


def test_cancel_marker_is_safe_file(tmp_path):
    manager = TrainingJobManager(tmp_path, jobs_root=tmp_path / "jobs", launcher=FakeLauncher())
    manifest = manager.create_job(_config())
    marker = manager.request_cancel(manifest.job_id)
    assert marker.name == "cancel.request" and marker.is_file()
