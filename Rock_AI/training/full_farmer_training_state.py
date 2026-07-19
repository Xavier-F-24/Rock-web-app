"""Serializable generation-boundary state for durable full-farmer evolution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .action_curriculum import ActionCurriculumStage


FULL_FARMER_TRAINING_STATE_VERSION = 1


@dataclass
class FullFarmerTrainingState:
    generation: int
    curriculum_stage: ActionCurriculumStage
    stage_entry_generation: int
    validation_history: list[float] = field(default_factory=list)
    invalid_rate_history: list[float] = field(default_factory=list)
    champion_archive: list[str] = field(default_factory=list)
    scenario_schedule_version: int = 1
    fitness_version: int = 2
    state_version: int = FULL_FARMER_TRAINING_STATE_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["curriculum_stage"] = self.curriculum_stage.name.lower()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FullFarmerTrainingState":
        payload = dict(data)
        if int(payload.get("state_version", 1)) != FULL_FARMER_TRAINING_STATE_VERSION:
            raise ValueError("Unsupported full-farmer training state version")
        payload["curriculum_stage"] = ActionCurriculumStage[str(payload["curriculum_stage"]).upper()]
        return cls(**payload)


def should_advance_curriculum(state: FullFarmerTrainingState, config) -> bool:
    if state.curriculum_stage >= config.curriculum_max:
        return False
    if state.generation - state.stage_entry_generation + 1 < config.minimum_generations_per_stage:
        return False
    window = config.curriculum_stability_window
    if len(state.validation_history) < window or len(state.invalid_rate_history) < window:
        return False
    validation = state.validation_history[-window:]
    invalid = state.invalid_rate_history[-window:]
    return (
        all(value >= config.curriculum_validation_threshold for value in validation)
        and max(invalid) <= config.curriculum_invalid_rate_threshold
        and max(validation) - min(validation) <= .20
    )
