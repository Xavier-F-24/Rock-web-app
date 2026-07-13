"""Epoch history persistence and readable summaries."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrainingHistory:
    epochs: list[dict[str, Any]] = field(default_factory=list)

    def add(self, epoch: int, train: dict[str, float], validation: dict[str, float]) -> None:
        self.epochs.append({"epoch": int(epoch), "train": train, "validation": validation})

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")
        return output


def format_epoch_summary(epoch: int, train: dict[str, float], validation: dict[str, float]) -> str:
    return (
        f"Epoch {epoch:03d} | train={train['total_loss']:.5f} "
        f"| validation={validation['total_loss']:.5f} "
        f"| scalar={validation['scalar_loss']:.5f} "
        f"| binary={validation['probability_loss']:.5f} "
        f"| genotype={validation['genotype_distribution_loss']:.5f} "
        f"| phenotype={validation['phenotype_distribution_loss']:.5f}"
    )
