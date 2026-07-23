"""Time-based worker heartbeat emission and phase-aware health classification."""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum


class HeartbeatHealth(str, Enum):
    HEALTHY = "healthy"
    SLOW = "slow"
    STAGNANT = "stagnant"
    ORPHANED = "orphaned"
    FAILED = "failed"


class HeartbeatPhase(str, Enum):
    STARTUP = "startup"
    GENOME_EVALUATION = "genome_evaluation"
    SCENARIO_EVALUATION = "scenario_evaluation"
    CANDIDATE_GENERATION = "candidate_generation"
    WORLD_EPISODE = "world_episode"
    CHECKPOINT_WRITING = "checkpoint_writing"
    GENERATION_FINALIZATION = "generation_finalization"


@dataclass
class TimedHeartbeat:
    callback: object | None
    interval_seconds: float = 5.0
    clock: object = time.monotonic
    _last_emitted: float = 0.0

    def pulse(self, phase: str | HeartbeatPhase, *, force: bool = False, **payload) -> bool:
        now = float(self.clock())
        if not force and now - self._last_emitted < self.interval_seconds:
            return False
        self._last_emitted = now
        if self.callback:
            phase_value = phase.value if isinstance(phase, HeartbeatPhase) else str(phase)
            self.callback(
                {
                    "event_type": "worker_heartbeat",
                    "phase": phase_value,
                    "health": HeartbeatHealth.HEALTHY.value,
                    "heartbeat_time": datetime.now(timezone.utc).isoformat(),
                    **payload,
                }
            )
        return True


def classify_heartbeat(
    age_seconds: float,
    *,
    process_alive: bool = True,
    failed: bool = False,
    slow_seconds: float = 30.0,
    stagnant_seconds: float = 120.0,
) -> HeartbeatHealth:
    if failed:
        return HeartbeatHealth.FAILED
    if not process_alive:
        return HeartbeatHealth.ORPHANED
    if age_seconds >= stagnant_seconds:
        return HeartbeatHealth.STAGNANT
    if age_seconds >= slow_seconds:
        return HeartbeatHealth.SLOW
    return HeartbeatHealth.HEALTHY


class BackgroundHeartbeat:
    """Emit heartbeats while the training thread is inside a long operation."""

    def __init__(self, callback, interval_seconds: float = 5.0):
        self.callback = callback
        self.interval_seconds = float(interval_seconds)
        self._phase = HeartbeatPhase.STARTUP.value
        self._payload: dict[str, object] = {"operation": "worker_started"}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def update(self, phase: str | HeartbeatPhase, **payload) -> None:
        with self._lock:
            self._phase = phase.value if isinstance(phase, HeartbeatPhase) else str(phase)
            self._payload = dict(payload)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="rock-ai-heartbeat", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 2))
        self._thread = None

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            with self._lock:
                phase = self._phase
                payload = dict(self._payload)
            if self.callback:
                self.callback(
                    {
                        "event_type": "worker_heartbeat",
                        "phase": phase,
                        "health": HeartbeatHealth.HEALTHY.value,
                        "heartbeat_time": datetime.now(timezone.utc).isoformat(),
                        **payload,
                    }
                )
