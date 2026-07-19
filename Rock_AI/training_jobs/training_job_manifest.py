"""Versioned immutable training-job manifest."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .training_job_config import TrainingJobConfig


TRAINING_JOB_MANIFEST_VERSION = 2


@dataclass(frozen=True)
class TrainingJobManifest:
    job_id: str
    created_at: str
    repository_root: str
    job_directory: str
    config: TrainingJobConfig
    command: tuple[str, ...]
    manifest_version: int = TRAINING_JOB_MANIFEST_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self); payload["config"] = self.config.to_dict(); payload["command"] = list(self.command)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingJobManifest":
        version = int(data.get("manifest_version", 1))
        if version not in {1, TRAINING_JOB_MANIFEST_VERSION}:
            raise ValueError("Unsupported training-job manifest version")
        return cls(str(data["job_id"]), str(data["created_at"]), str(data["repository_root"]), str(data["job_directory"]), TrainingJobConfig.from_dict(data["config"]), tuple(data["command"]), TRAINING_JOB_MANIFEST_VERSION)
