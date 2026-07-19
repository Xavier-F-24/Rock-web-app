"""Filesystem registry for durable local training jobs."""

from __future__ import annotations

from pathlib import Path

from .training_progress_reader import TrainingProgressReader


class TrainingJobRegistry:
    def __init__(self, root: str | Path): self.root = Path(root)
    def job_directories(self) -> tuple[Path, ...]:
        return tuple(sorted(path for path in self.root.glob("job_*") if (path / "job_manifest.json").exists())) if self.root.exists() else ()
    def statuses(self):
        rows = []
        for directory in self.job_directories():
            try: rows.append(TrainingProgressReader(directory).status())
            except (OSError, ValueError, KeyError): continue
        return tuple(rows)
