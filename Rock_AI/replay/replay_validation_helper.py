"""Numerical replay comparison with explicit divergence details."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ReplayDivergence:
    decision_index: int
    field_name: str
    expected: Any
    actual: Any
    message: str


@dataclass
class ReplayValidationReport:
    valid: bool = True
    divergences: list[ReplayDivergence] = field(default_factory=list)

    def add(self, decision_index: int, field_name: str, expected: Any, actual: Any) -> None:
        self.valid = False
        self.divergences.append(
            ReplayDivergence(
                decision_index,
                field_name,
                expected,
                actual,
                f"Decision {decision_index}: {field_name} expected {expected!r}, got {actual!r}",
            )
        )


def compare_metric_summaries(
    report: ReplayValidationReport,
    decision_index: int,
    expected: dict[str, Any],
    actual: dict[str, Any],
    *,
    tolerance: float = 1e-9,
) -> None:
    for name, expected_value in expected.items():
        if name not in actual:
            report.add(decision_index, name, expected_value, "<missing>")
            continue
        actual_value = actual[name]
        if isinstance(expected_value, (int, float)) and isinstance(actual_value, (int, float)):
            if abs(float(expected_value) - float(actual_value)) > tolerance:
                report.add(decision_index, name, expected_value, actual_value)
        elif expected_value != actual_value:
            report.add(decision_index, name, expected_value, actual_value)
