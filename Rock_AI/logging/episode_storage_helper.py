"""JSON Lines persistence for replayable campaign records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .agent_decision_record import AgentDecisionRecord
from .episode_record import EpisodeRecord


def episode_record_from_dict(data: dict) -> EpisodeRecord:
    values = dict(data)
    decisions = []
    for decision in values.get("decisions", []):
        decision_values = dict(decision)
        if decision_values.get("selected_parent_ids") is not None:
            decision_values["selected_parent_ids"] = tuple(decision_values["selected_parent_ids"])
        decisions.append(AgentDecisionRecord(**decision_values))
    values["decisions"] = decisions
    return EpisodeRecord(**values)


def save_episode_records(path: str | Path, records: Iterable[EpisodeRecord]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
    return destination


def load_episode_records(path: str | Path) -> list[EpisodeRecord]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return [episode_record_from_dict(json.loads(line)) for line in stream if line.strip()]
