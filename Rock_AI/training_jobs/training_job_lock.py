"""Atomic directory locks for jobs and mutable training runs."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


class TrainingJobLock:
    def __init__(self, path: str | Path, owner: str):
        self.path = Path(path); self.owner = owner; self.acquired = False

    def acquire(self) -> None:
        try:
            self.path.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise RuntimeError(f"Training lock is already held: {self.path}") from error
        (self.path / "owner.json").write_text(json.dumps({"owner": self.owner, "pid": os.getpid(), "acquired_at": datetime.now(timezone.utc).isoformat()}), encoding="utf-8")
        self.acquired = True

    def release(self) -> None:
        if not self.acquired:
            return
        owner_path = self.path / "owner.json"
        owner_path.unlink(missing_ok=True)
        self.path.rmdir()
        self.acquired = False

    def __enter__(self): self.acquire(); return self
    def __exit__(self, *_): self.release()
