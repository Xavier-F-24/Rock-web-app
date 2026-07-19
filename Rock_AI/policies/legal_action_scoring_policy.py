"""Interfaces and result records for heterogeneous legal-action scoring."""

from dataclasses import dataclass
from typing import Protocol

from Rock_AI.actions.action_candidate import ActionCandidate
from Rock_AI.actions.action_explanation import ActionExplanation


@dataclass(frozen=True)
class RankedAction:
    candidate: ActionCandidate
    score: float
    rank: int
    confidence_proxy: float = 0.0


@dataclass(frozen=True)
class ActionRankingDecision:
    ranked_actions: tuple[RankedAction, ...]
    selected: ActionCandidate | None
    explanation: ActionExplanation | None
    model_trace: dict | None = None


class LegalActionScoringPolicy(Protocol):
    def rank_actions(self, observation) -> ActionRankingDecision: ...
    def commit_selected(self, selected: ActionCandidate, visible_result: object) -> None: ...
    def reset(self, episode_id: str) -> None: ...
