"""Serialization-safe diagnostics for stalled or failed full-farmer scenarios."""

from __future__ import annotations

import traceback
from pathlib import Path

from Rock_AI.training_jobs.training_progress_reader import atomic_write_json


def write_stagnation_diagnostic(
    output_directory: str | Path,
    *,
    job_id: str | None,
    generation: int,
    genome_id: int | str,
    scenario_id: int | str,
    active_farm: str | None,
    environment,
    error: BaseException | None = None,
) -> Path:
    directory = Path(output_directory) / "diagnostics"
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = environment.diagnostic_snapshot()
    payload = {
        "job_id": job_id,
        "generation": int(generation),
        "genome_id": str(genome_id),
        "scenario_id": str(scenario_id),
        "world_turn": snapshot.get("world_turn"),
        "active_farm": active_farm,
        "legal_action_counts_by_type": snapshot.get("legal_action_counts_by_type", {}),
        "current_reservations": snapshot.get("current_reservations", {}),
        "pass_counters": {
            "consecutive": snapshot.get("consecutive_passes", 0),
            "decisions_by_farm": snapshot.get("decisions_by_farm", {}),
        },
        "no_progress_counter": snapshot.get("no_progress_counter", 0),
        "failed_transactions": snapshot.get("failed_transactions", 0),
        "recent_state_hashes": snapshot.get("recent_state_hashes", []),
        "last_completed_operation": snapshot.get("last_completed_operation"),
        "termination_reason": snapshot.get("termination_reason"),
        "elapsed_seconds": snapshot.get("elapsed_seconds"),
        "stack_trace": None if error is None else "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        ),
    }
    path = directory / f"stagnation_g{generation:04d}_genome_{genome_id}_scenario_{scenario_id}.json"
    atomic_write_json(path, payload)
    return path
