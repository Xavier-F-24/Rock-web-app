"""Curriculum action availability without changing external input width."""

from dataclasses import dataclass

from .farmer_action_type import ACTION_TYPE_ORDER, FarmerActionType


@dataclass(frozen=True)
class ActionAvailability:
    enabled: frozenset[FarmerActionType]
    schema_version: int = 1

    def permits(self, action_type: FarmerActionType) -> bool:
        return action_type in self.enabled

    @classmethod
    def all(cls) -> "ActionAvailability":
        return cls(frozenset(ACTION_TYPE_ORDER))
