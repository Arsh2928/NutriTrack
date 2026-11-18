"""Cooldown-aware single-provider probe coordinator without automatic cross-provider switching."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional


class RecoveryManager:
    """Coordinates cooldown-aware recovery probes without duplicate background threads or UI-refresh spam."""

    def __init__(
        self,
        *,
        health_check_interval_seconds: float = 60.0,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self.health_check_interval_seconds = max(0.5, float(health_check_interval_seconds))
        self._clock = clock or time.time
        self._lock = threading.RLock()
        self._last_probe_at: dict[str, float] = {}
        self._probe_in_progress: bool = False

    def can_probe(
        self,
        provider: str,
        *,
        selected_mode: str,
        circuit_state: str,
        force: bool = False,
    ) -> bool:
        """Return True if a recovery probe is permitted for `provider` right now."""
        if selected_mode == "offline":
            return False
        with self._lock:
            if self._probe_in_progress:
                return False
            if force:
                return True
            if circuit_state != "half_open":
                return False
            now = self._clock()
            last = self._last_probe_at.get(provider)
            if last is not None and (now - last) < self.health_check_interval_seconds:
                return False
            return True

    def mark_probe_started(self, provider: str) -> None:
        with self._lock:
            self._probe_in_progress = True
            self._last_probe_at[provider] = self._clock()

    def mark_probe_finished(self) -> None:
        with self._lock:
            self._probe_in_progress = False

    def reset(self) -> None:
        with self._lock:
            self._last_probe_at.clear()
            self._probe_in_progress = False

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "health_check_interval_seconds": self.health_check_interval_seconds,
                "probe_in_progress": self._probe_in_progress,
                "last_probe_at": dict(self._last_probe_at),
            }
