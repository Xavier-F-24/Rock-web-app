"""Detect whether this deployment can safely launch persistent local workers."""

from __future__ import annotations

import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingEnvironmentCapabilities:
    subprocess_supported: bool
    persistent_storage_supported: bool
    writable_training_directory: bool
    maximum_recommended_workers: int
    cuda_available: bool
    process_inspection_supported: bool
    cancellation_supported: bool
    reason: str | None = None

    def to_dict(self): return asdict(self)


def detect_training_environment(repository_root: str | Path) -> TrainingEnvironmentCapabilities:
    root = Path(repository_root)
    hosted = bool(os.environ.get("STREAMLIT_SHARING_MODE") or os.environ.get("STREAMLIT_CLOUD"))
    writable = os.access(root, os.W_OK)
    subprocess_supported = hasattr(subprocess, "Popen") and not hosted
    return TrainingEnvironmentCapabilities(
        subprocess_supported, not hosted, writable, max(1, min(4, os.cpu_count() or 1)),
        _cuda_available(), subprocess_supported, subprocess_supported,
        "Hosted Streamlit deployments are replay/inference only." if hosted else None,
    )


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return False
