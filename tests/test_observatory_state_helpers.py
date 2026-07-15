from __future__ import annotations

from Rock_AI.runtime.runtime_event_helper import RuntimeEvent, RuntimeEventType
from Rock_AI.runtime.runtime_state_helper import SessionStatus
from Rock_Streamlit.components.observatory_state_helper import (
    auto_run_should_continue,
    control_states,
    latest_event_rock_ids,
    timeline_rows,
)


def _event(index, kind, rock_ids=()):
    return RuntimeEvent(
        session_id="viewer",
        event_index=index,
        decision_index=1,
        generation=2,
        event_type=kind,
        summary=kind.value,
        rock_ids=rock_ids,
    )


def test_controls_follow_runtime_statuses():
    assert control_states(None)["start"]
    assert control_states(SessionStatus.WAITING_FOR_STEP)["step"]
    assert control_states(SessionStatus.PAUSED)["resume"]
    assert not control_states(SessionStatus.PAUSED)["run_generation"]
    assert not control_states(SessionStatus.COMPLETED)["auto"]
    assert not any(control_states(SessionStatus.READY, replay_mode=True).values())


def test_events_convert_to_timeline_and_highlight_latest_children_and_mutations():
    events = [
        _event(0, RuntimeEventType.DECISION_STARTED),
        _event(1, RuntimeEventType.CHILDREN_CREATED, (7, 8)),
        _event(2, RuntimeEventType.MUTATION_OCCURRED, (8,)),
    ]
    rows = timeline_rows(events)
    assert [row["category"] for row in rows] == ["decision", "birth", "mutation"]
    assert rows[-1]["rock_ids"] == [8]
    assert latest_event_rock_ids(events) == {"children": (7, 8), "mutations": (8,)}


def test_auto_run_stops_for_pause_or_terminal_status():
    assert auto_run_should_continue(SessionStatus.READY, True, False)
    assert not auto_run_should_continue(SessionStatus.PAUSED, True, False)
    assert not auto_run_should_continue(SessionStatus.READY, True, True)
    assert not auto_run_should_continue(SessionStatus.COMPLETED, True, False)
