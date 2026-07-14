"""Stored presentation pacing and data-driven pause recommendations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class RuntimeSpeedMode(str, Enum):
    MANUAL_STEP = "manual_step"
    GENERATION_STEP = "generation_step"
    AUTO = "auto"


@dataclass(frozen=True)
class RuntimeSpeedConfig:
    mode: RuntimeSpeedMode = RuntimeSpeedMode.MANUAL_STEP
    delay_after_decision_seconds: float = 0.0
    delay_after_breeding_seconds: float = 0.0
    delay_after_generation_seconds: float = 0.0
    pause_on_mutation: bool = False
    pause_on_rare_trait: bool = False
    pause_on_new_high_value_rock: bool = False
    pause_on_close_decision: bool = False
    close_decision_threshold: float = 0.05

    def __post_init__(self) -> None:
        if min(
            self.delay_after_decision_seconds,
            self.delay_after_breeding_seconds,
            self.delay_after_generation_seconds,
            self.close_decision_threshold,
        ) < 0:
            raise ValueError("Runtime delays and close-decision threshold cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["mode"] = self.mode.value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuntimeSpeedConfig":
        values = dict(data)
        values["mode"] = RuntimeSpeedMode(values.get("mode", RuntimeSpeedMode.MANUAL_STEP.value))
        return cls(**values)


@dataclass(frozen=True)
class PauseRecommendation:
    should_pause: bool
    pause_reasons: tuple[str, ...] = ()


def evaluate_pause_conditions(
    config: RuntimeSpeedConfig,
    *,
    mutation_count: int = 0,
    rare_trait_increase: float = 0.0,
    maximum_value_increase: float = 0.0,
    candidate_score_gap: float | None = None,
) -> PauseRecommendation:
    reasons = []
    if config.pause_on_mutation and mutation_count > 0:
        reasons.append("mutation_occurred")
    if config.pause_on_rare_trait and rare_trait_increase > 0:
        reasons.append("rare_trait_produced")
    if config.pause_on_new_high_value_rock and maximum_value_increase > 0:
        reasons.append("new_high_value_rock")
    if (
        config.pause_on_close_decision
        and candidate_score_gap is not None
        and candidate_score_gap <= config.close_decision_threshold
    ):
        reasons.append("candidate_scores_nearly_tied")
    return PauseRecommendation(bool(reasons), tuple(reasons))
