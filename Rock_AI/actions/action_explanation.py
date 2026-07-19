"""Structured, non-chain-of-thought action explanation records."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionExplanation:
    action_hash: str
    summary: str
    score: float
    rank: int
    candidate_count: int
    confidence_proxy: float
    contributions: tuple[tuple[str, float], ...] = ()
    observations: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
