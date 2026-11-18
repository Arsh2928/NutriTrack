"""Bounded exponential backoff retry policy with random jitter and Retry-After budget guards."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable, Optional

from nutritrack.ai.failure_classifier import ClassifiedFailure


@dataclass
class RetryDecision:
    """Decision on whether to retry the current provider or switch immediately."""

    should_retry: bool
    delay_seconds: float
    reason: str


class RetryPolicy:
    """Bounded exponential backoff with jitter that never blocks beyond `max_delay_seconds`."""

    def __init__(
        self,
        *,
        max_retries: int = 2,
        base_delay_seconds: float = 1.0,
        max_delay_seconds: float = 8.0,
        jitter_ratio: float = 0.25,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.max_retries = max(0, int(max_retries))
        self.base_delay_seconds = max(0.0, float(base_delay_seconds))
        self.max_delay_seconds = max(0.05, float(max_delay_seconds))
        self.jitter_ratio = max(0.0, min(1.0, float(jitter_ratio)))
        self._rng = rng or random.Random()

    def evaluate(self, attempt_index: int, failure: ClassifiedFailure) -> RetryDecision:
        """Return a `RetryDecision` for 0-based `attempt_index` (0 = first failure)."""
        if not failure.retryable or failure.is_quota_exhausted or failure.is_config_error:
            return RetryDecision(
                should_retry=False,
                delay_seconds=0.0,
                reason=f"Non-retryable category '{failure.category}'.",
            )
        if attempt_index >= self.max_retries:
            return RetryDecision(
                should_retry=False,
                delay_seconds=0.0,
                reason=f"Reached max_retries={self.max_retries}.",
            )

        # If provider gave an explicit Retry-After header/duration, honor it only if within request budget
        if failure.retry_after_seconds is not None:
            if failure.retry_after_seconds > self.max_delay_seconds:
                return RetryDecision(
                    should_retry=False,
                    delay_seconds=0.0,
                    reason=(
                        f"Provider requested Retry-After={failure.retry_after_seconds:g}s, "
                        f"exceeding max_delay_seconds={self.max_delay_seconds:g}s; switching provider immediately."
                    ),
                )
            jitter = self._rng.uniform(0.0, min(0.25, self.base_delay_seconds * 0.2)) if self.base_delay_seconds > 0 else 0.0
            delay = min(self.max_delay_seconds, failure.retry_after_seconds + jitter)
            return RetryDecision(
                should_retry=True,
                delay_seconds=round(delay, 3),
                reason=f"Honoring Retry-After ({delay:.2f}s).",
            )

        raw_exp = self.base_delay_seconds * (2 ** attempt_index)
        jitter = raw_exp * self.jitter_ratio * self._rng.random()
        delay = min(self.max_delay_seconds, raw_exp + jitter)
        return RetryDecision(
            should_retry=True,
            delay_seconds=round(delay, 3),
            reason=f"Exponential backoff retry #{attempt_index + 1} ({delay:.2f}s).",
        )
