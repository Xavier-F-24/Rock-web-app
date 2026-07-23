"""Safe, tolerant reads of worker status and progress artifacts."""

from __future__ import annotations

import json
import os
import ctypes
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .training_job_status import TERMINAL_JOB_STATES, TrainingJobState, TrainingJobStatus
from .worker_heartbeat import HeartbeatHealth, classify_heartbeat


def atomic_write_json(path: str | Path, payload: dict) -> None:
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    try:
        for attempt in range(20):
            try:
                os.replace(temporary, destination)
                return
            except PermissionError:
                if attempt == 19: raise
                time.sleep(0.05)
    finally:
        temporary.unlink(missing_ok=True)


class TrainingProgressReader:
    def __init__(self, job_directory: str | Path): self.job_directory = Path(job_directory)
    def status(self) -> TrainingJobStatus:
        return TrainingJobStatus.from_dict(json.loads((self.job_directory / "status.json").read_text(encoding="utf-8")))
    def progress(self) -> list[dict]:
        path = self.job_directory / "progress.jsonl"
        if not path.exists(): return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try: rows.append(json.loads(line))
            except json.JSONDecodeError: continue
        return rows
    def console_tail(self, lines: int = 200) -> str:
        path = self.job_directory / "console.log"
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]) if path.exists() else ""
    def orphan_warning(self, stale_seconds: float = 120.0) -> str | None:
        status = self.status()
        if status.status in TERMINAL_JOB_STATES or not status.last_heartbeat_time: return None
        heartbeat = datetime.fromisoformat(status.last_heartbeat_time)
        age = (datetime.now(timezone.utc) - heartbeat).total_seconds()
        health = classify_heartbeat(
            age,
            process_alive=self._process_alive(status.process_id),
            failed=status.status is TrainingJobState.FAILED,
            slow_seconds=min(30.0, stale_seconds / 2),
            stagnant_seconds=stale_seconds,
        )
        if health is HeartbeatHealth.HEALTHY:
            return None
        label = "stale (stagnant)" if health is HeartbeatHealth.STAGNANT else health.value
        return f"Worker heartbeat is {label} (last update {age:.0f} seconds ago during {status.heartbeat_phase or 'unknown phase'})"

    def heartbeat_health(self, slow_seconds: float = 30.0, stagnant_seconds: float = 120.0) -> HeartbeatHealth:
        status = self.status()
        if status.status is TrainingJobState.FAILED:
            return HeartbeatHealth.FAILED
        if not status.last_heartbeat_time:
            return HeartbeatHealth.ORPHANED if status.status not in TERMINAL_JOB_STATES else HeartbeatHealth.HEALTHY
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(status.last_heartbeat_time)).total_seconds()
        return classify_heartbeat(
            age,
            process_alive=self._process_alive(status.process_id),
            slow_seconds=slow_seconds,
            stagnant_seconds=stagnant_seconds,
        )

    @staticmethod
    def _process_alive(process_id: int | None) -> bool:
        if not process_id:
            return True
        if os.name == "nt":
            return TrainingProgressReader._windows_process_alive(int(process_id))
        try:
            os.kill(int(process_id), 0)
            return True
        except (OSError, ValueError, SystemError):
            return False

    @staticmethod
    def _windows_process_alive(process_id: int) -> bool:
        process_query_limited_information = 0x1000
        still_active = 259
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
            kernel32.OpenProcess.restype = ctypes.c_void_p
            kernel32.GetExitCodeProcess.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32))
            kernel32.GetExitCodeProcess.restype = ctypes.c_int
            kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
            handle = kernel32.OpenProcess(process_query_limited_information, False, process_id)
            if not handle:
                # Access denied means the process may exist but is not inspectable.
                return ctypes.get_last_error() == 5
            try:
                exit_code = ctypes.c_uint32()
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                    return True
                return exit_code.value == still_active
            finally:
                kernel32.CloseHandle(handle)
        except (OSError, ValueError, TypeError, AttributeError):
            # Status rendering must never take down Streamlit over a process probe.
            return True
