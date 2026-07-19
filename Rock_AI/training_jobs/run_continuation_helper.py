"""Validation and loading for trusted local full-population checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import neat


def resolve_checkpoint(source_run: str | Path, requested: str | None = None) -> Path:
    run = Path(source_run)
    if requested and requested != "latest":
        checkpoint = Path(requested)
        if not checkpoint.is_absolute() and not checkpoint.exists(): checkpoint = run / checkpoint
    else:
        candidates = sorted((run / "checkpoints").glob("neat-checkpoint-*"), key=lambda path: int(path.name.rsplit("-", 1)[-1]))
        if not candidates: raise FileNotFoundError(f"No resumable checkpoints in {run}")
        checkpoint = candidates[-1]
    if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
    return checkpoint.resolve()


def validate_run_compatibility(source_run: str | Path, checkpoint: str | Path) -> dict:
    run = Path(source_run); manifest_path = run / "run_manifest.json"; config_path = run / "training_config.json"
    if not manifest_path.exists() or not config_path.exists(): raise ValueError("Source run lacks manifest or training configuration")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("information_access") != "player": raise ValueError("Only player-certified runs may continue")
    if manifest.get("run_type") != "recurrent_neat_player_farmer": raise ValueError("Checkpoint topology implementation is incompatible")
    if int(manifest.get("topology_implementation_version", 1)) != 1: raise ValueError("Unsupported topology implementation version")
    if int(manifest.get("fitness_implementation_version", 1)) != 1: raise ValueError("Unsupported fitness implementation version")
    if not Path(checkpoint).exists(): raise FileNotFoundError(checkpoint)
    return {"manifest": manifest, "training_config": json.loads(config_path.read_text(encoding="utf-8"))}


def restore_population(checkpoint: str | Path):
    return neat.Checkpointer.restore_checkpoint(str(checkpoint))
