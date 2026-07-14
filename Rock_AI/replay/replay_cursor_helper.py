"""Bounded cursor movement for replay frames."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ReplayCursor:
    frame_count: int
    position: int = 0

    def __post_init__(self) -> None:
        if self.frame_count <= 0:
            raise ValueError("A replay cursor requires at least one frame")
        self.seek(self.position)

    def seek(self, position: int | str) -> int:
        if position == "first":
            value = 0
        elif position == "last":
            value = self.frame_count - 1
        else:
            value = int(position)
        if not 0 <= value < self.frame_count:
            raise IndexError(f"Replay position {value} is outside 0..{self.frame_count - 1}")
        self.position = value
        return self.position

    def next(self) -> int:
        return self.seek(min(self.position + 1, self.frame_count - 1))

    def previous(self) -> int:
        return self.seek(max(self.position - 1, 0))

    def first(self) -> int:
        return self.seek("first")

    def last(self) -> int:
        return self.seek("last")
