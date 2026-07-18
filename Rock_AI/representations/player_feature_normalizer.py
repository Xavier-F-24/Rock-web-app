"""Versioned, serialization-safe normalization for player-visible features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


PLAYER_NORMALIZER_VERSION = 1


@dataclass(frozen=True)
class PlayerFeatureNormalizer:
    feature_names: tuple[str, ...]
    lower_bounds: tuple[float, ...]
    upper_bounds: tuple[float, ...]
    version: int = PLAYER_NORMALIZER_VERSION
    transform_type: str = "bounded_linear"
    unknown_value: float = 0.0
    mask_semantics: str = "true_when_observed"

    def __post_init__(self) -> None:
        width = len(self.feature_names)
        if len(self.lower_bounds) != width or len(self.upper_bounds) != width:
            raise ValueError("Normalizer names and bounds must have equal lengths")
        if any(high <= low for low, high in zip(self.lower_bounds, self.upper_bounds)):
            raise ValueError(
                "Every normalizer upper bound must exceed its lower bound"
            )

    def normalize(
        self,
        values: Iterable[float],
        visibility_mask: Iterable[bool],
    ) -> tuple[tuple[float, ...], tuple[bool, ...]]:
        raw = tuple(float(value) for value in values)
        mask = tuple(bool(value) for value in visibility_mask)
        if len(raw) != len(self.feature_names) or len(mask) != len(
            self.feature_names
        ):
            raise ValueError(
                "Feature values and masks must match the normalizer width"
            )
        normalized = tuple(
            self.unknown_value
            if not visible
            else max(0.0, min(1.0, (value - low) / (high - low)))
            for value, visible, low, high in zip(
                raw, mask, self.lower_bounds, self.upper_bounds
            )
        )
        return normalized, mask

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "feature_names": list(self.feature_names),
            "transform_type": self.transform_type,
            "lower_bounds": list(self.lower_bounds),
            "upper_bounds": list(self.upper_bounds),
            "unknown_value": self.unknown_value,
            "mask_semantics": self.mask_semantics,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "PlayerFeatureNormalizer":
        return cls(
            feature_names=tuple(values["feature_names"]),
            lower_bounds=tuple(float(value) for value in values["lower_bounds"]),
            upper_bounds=tuple(float(value) for value in values["upper_bounds"]),
            version=int(values["version"]),
            transform_type=str(values["transform_type"]),
            unknown_value=float(values["unknown_value"]),
            mask_semantics=str(values["mask_semantics"]),
        )
