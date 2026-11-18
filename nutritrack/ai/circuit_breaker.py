"""Independent thread-safe Circuit Breaker (CLOSED / OPEN / HALF_OPEN) for cloud AI providers."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional

from nutritrack.ai.failure_classifier import ClassifiedFailure

STATE_CLOSED = "closed"
STATE_OPEN = "open"
STATE_HALF_OPEN = "half_open"


class CircuitBreaker:
    """Per-provider circuit breaker with controllable clock and single-probe half-open recovery."""

    def __init__(
        self,
        provider: str,
        *,
        failure_threshold: int = 3,
        cooldown_seconds: float = 60.0,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self.provider = provider
        self.failure_threshold = max(1, int(failure_threshold))
        self.cooldown_seconds = max(0.5, float(cooldown_seconds))
        self._clock = clock or time.time
        self._lock = threading.RLock()

        self._state: str = STATE_CLOSED
        self._failure_count: int = 0
        self._opened_at: Optional[float] = None
        self._cooldown_until: Optional[float] = None
        self._probe_in_flight: bool = False
        self._last_failure_reason: Optional[str] = None
        self._last_failure_category: Optional[str] = None
        self._last_success_at: Optional[float] = None
        self._is_quota_exhausted: bool = False
        self._is_config_error: bool = False

    @property
    def state(self) -> str:
        with self._lock:
            self._check_cooldown_expiry_locked()
            return self._state

    def _check_cooldown_expiry_locked(self) -> None:
        if self._state == STATE_OPEN and not self._is_config_error:
            now = self._clock()
            if self._cooldown_until is not None and now >= self._cooldown_until:
                self._state = STATE_HALF_OPEN
                self._probe_in_flight = False

    def allow_request(self) -> tuple[bool, str]:
        """Return `(allowed, reason)`. In `HALF_OPEN`, permits only one probe at a time."""
        with self._lock:
            self._check_cooldown_expiry_locked()
            if self._is_config_error:
                return False, f"{self.provider} has a configuration error ({self._last_failure_category})."
            if self._state == STATE_CLOSED:
                return True, "closed"
            if self._state == STATE_HALF_OPEN:
                if self._probe_in_flight:
                    return False, "half_open_probe_in_flight"
                self._probe_in_flight = True
                return True, "half_open_probe"
            return False, "circuit_open"

    def record_success(self) -> dict[str, Any]:
        """Mark request successful; close circuit if recovering from OPEN/HALF_OPEN."""
        with self._lock:
            now = self._clock()
            prev_state = self._state
            was_recovering = prev_state in {STATE_OPEN, STATE_HALF_OPEN} or self._failure_count > 0
            self._state = STATE_CLOSED
            self._failure_count = 0
            self._opened_at = None
            self._cooldown_until = None
            self._probe_in_flight = False
            self._last_failure_reason = None
            self._last_failure_category = None
            self._is_quota_exhausted = False
            self._is_config_error = False
            self._last_success_at = now
            return {
                "previous_state": prev_state,
                "current_state": STATE_CLOSED,
                "recovered": was_recovering and prev_state != STATE_CLOSED,
            }

    def record_failure(self, failure: ClassifiedFailure, *, trip_immediately: bool = False) -> dict[str, Any]:
        """Record a classified failure and transition to OPEN when threshold or immediate trip is met."""
        with self._lock:
            now = self._clock()
            prev_state = self._state
            self._failure_count += 1
            self._probe_in_flight = False
            self._last_failure_reason = failure.sanitized_reason
            self._last_failure_category = failure.category
            self._is_quota_exhausted = failure.is_quota_exhausted
            self._is_config_error = failure.is_config_error

            should_open = (
                prev_state == STATE_HALF_OPEN
                or trip_immediately
                or failure.trip_circuit_immediately
                or self._failure_count >= self.failure_threshold
            )
            if should_open:
                self._state = STATE_OPEN
                self._opened_at = now
                override = failure.cooldown_override_seconds or failure.retry_after_seconds or 0.0
                cooldown = max(self.cooldown_seconds, float(override))
                self._cooldown_until = now + max(0.5, float(cooldown))
            return {
                "previous_state": prev_state,
                "current_state": self._state,
                "opened_now": prev_state != STATE_OPEN and self._state == STATE_OPEN,
                "failure_count": self._failure_count,
                "cooldown_until": self._cooldown_until,
                "category": failure.category,
            }

    def release_probe(self) -> None:
        """Release an in-flight HALF_OPEN probe flag if aborted before recording an outcome."""
        with self._lock:
            if self._state == STATE_HALF_OPEN:
                self._probe_in_flight = False

    def reset(self) -> None:
        """Manually reset the circuit breaker to CLOSED (e.g., on manual retry or config change)."""
        with self._lock:
            self._state = STATE_CLOSED
            self._failure_count = 0
            self._opened_at = None
            self._cooldown_until = None
            self._probe_in_flight = False
            self._last_failure_reason = None
            self._last_failure_category = None
            self._is_quota_exhausted = False
            self._is_config_error = False

    def snapshot(self) -> dict[str, Any]:
        """Return safe diagnostic snapshot of circuit breaker state."""
        with self._lock:
            self._check_cooldown_expiry_locked()
            now = self._clock()
            rem_cooldown = (
                max(0.0, round(self._cooldown_until - now, 1))
                if self._state == STATE_OPEN and self._cooldown_until is not None
                else 0.0
            )
            return {
                "provider": self.provider,
                "state": self._state,
                "failure_count": self._failure_count,
                "failure_threshold": self.failure_threshold,
                "cooldown_remaining_seconds": rem_cooldown,
                "probe_in_flight": self._probe_in_flight,
                "is_quota_exhausted": self._is_quota_exhausted,
                "is_config_error": self._is_config_error,
                "last_failure_category": self._last_failure_category,
                "last_failure_reason": self._last_failure_reason,
                "last_success_at": self._last_success_at,
            }
