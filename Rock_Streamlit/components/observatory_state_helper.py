"""Pure state-to-view helpers used by the AI Observatory."""

from __future__ import annotations

from Rock_AI.runtime.runtime_event_helper import RuntimeEventType
from Rock_AI.runtime.runtime_state_helper import SessionStatus


TERMINAL = {SessionStatus.COMPLETED, SessionStatus.FAILED, SessionStatus.CANCELLED}


def control_states(status: SessionStatus | None, *, replay_mode: bool = False) -> dict[str, bool]:
    if replay_mode:
        return {"start": False, "step": False, "run_generation": False, "auto": False,
                "pause": False, "resume": False, "reset": False, "save": False}
    if status is None:
        return {"start": True, "step": False, "run_generation": False, "auto": False,
                "pause": False, "resume": False, "reset": False, "save": False}
    terminal = status in TERMINAL
    paused = status == SessionStatus.PAUSED
    created = status == SessionStatus.CREATED
    return {
        "start": created,
        "step": not created and not terminal,
        "run_generation": not created and not terminal and not paused,
        "auto": not created and not terminal and not paused,
        "pause": status in {SessionStatus.READY, SessionStatus.RUNNING, SessionStatus.WAITING_FOR_STEP},
        "resume": paused,
        "reset": True,
        "save": True,
    }


def timeline_rows(events) -> list[dict]:
    category = {
        RuntimeEventType.DECISION_STARTED: "decision",
        RuntimeEventType.PAIR_SELECTED: "decision",
        RuntimeEventType.BREEDING_EXECUTED: "breeding",
        RuntimeEventType.CHILDREN_CREATED: "birth",
        RuntimeEventType.MUTATION_OCCURRED: "mutation",
        RuntimeEventType.ROCK_STATUS_CHANGED: "status",
        RuntimeEventType.GENERATION_ADVANCED: "generation",
        RuntimeEventType.SESSION_PAUSED: "pause",
        RuntimeEventType.SESSION_COMPLETED: "completion",
        RuntimeEventType.SESSION_FAILED: "failure",
    }
    return [
        {
            "event_index": event.event_index,
            "generation": event.generation,
            "decision": event.decision_index,
            "category": category.get(event.event_type, "info"),
            "summary": event.summary,
            "rock_ids": list(event.rock_ids),
            "details": event.to_dict(),
        }
        for event in events
    ]


def latest_event_rock_ids(events) -> dict[str, tuple]:
    result = {"children": (), "mutations": ()}
    for event in reversed(list(events)):
        if not result["children"] and event.event_type == RuntimeEventType.CHILDREN_CREATED:
            result["children"] = event.rock_ids
        if not result["mutations"] and event.event_type == RuntimeEventType.MUTATION_OCCURRED:
            result["mutations"] = event.rock_ids
        if result["children"] and result["mutations"]:
            break
    return result


def auto_run_should_continue(status: SessionStatus, enabled: bool, should_pause: bool) -> bool:
    return bool(enabled and not should_pause and status not in TERMINAL and status != SessionStatus.PAUSED)
