from __future__ import annotations

from types import SimpleNamespace

from Rock_AI.runtime import AgentRuntimeManager
from Rock_Streamlit.sections import ai_observatory


def test_stale_streamlit_manager_discards_raw_sessions_and_ui_pointers(monkeypatch):
    stale = SimpleNamespace(sessions={"existing-session": object()})
    fake_streamlit = SimpleNamespace(
        session_state={
            ai_observatory.MANAGER_KEY: stale,
            ai_observatory.SESSION_ID_KEY: "existing-session",
            ai_observatory.LIVE_SESSION_ID_KEY: "existing-session",
            ai_observatory.LAST_RESULT_KEY: object(),
            ai_observatory.AUTO_RUN_KEY: True,
            ai_observatory.REPLAY_MODE_KEY: True,
        }
    )
    monkeypatch.setattr(ai_observatory, "st", fake_streamlit)

    manager = ai_observatory.get_runtime_manager()

    assert isinstance(manager, AgentRuntimeManager)
    assert manager.interface_version == AgentRuntimeManager.INTERFACE_VERSION
    assert manager.sessions == {}
    assert ai_observatory.SESSION_ID_KEY not in fake_streamlit.session_state
    assert ai_observatory.LIVE_SESSION_ID_KEY not in fake_streamlit.session_state
    assert ai_observatory.LAST_RESULT_KEY not in fake_streamlit.session_state
    assert fake_streamlit.session_state[ai_observatory.AUTO_RUN_KEY] is False
    assert fake_streamlit.session_state[ai_observatory.REPLAY_MODE_KEY] is False
