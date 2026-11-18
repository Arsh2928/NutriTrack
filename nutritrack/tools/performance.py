"""Lightweight performance instrumentation for NutriTrack operations and endpoints."""

from __future__ import annotations

import statistics
import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import Any, Iterator

MAX_SAMPLES_PER_OP = 250


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return round(sorted_vals[0], 3)
    idx = min(len(sorted_vals) - 1, max(0, int(round((pct / 100.0) * (len(sorted_vals) - 1)))))
    return round(sorted_vals[idx], 3)


class PerformanceMonitor:
    """Thread-safe rolling latency collector tracking cold, warm, mean, median, and p95 timings."""

    def __init__(self, max_samples: int = MAX_SAMPLES_PER_OP) -> None:
        self._lock = threading.RLock()
        self._max_samples = max(10, int(max_samples))
        self._cold_ms: dict[str, float] = {}
        self._samples: dict[str, deque[float]] = {}

    def record(self, operation: str, elapsed_ms: float) -> None:
        val = round(max(0.0, float(elapsed_ms)), 3)
        with self._lock:
            if operation not in self._cold_ms:
                self._cold_ms[operation] = val
            buf = self._samples.setdefault(operation, deque(maxlen=self._max_samples))
            buf.append(val)

    @contextmanager
    def measure(self, operation: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.record(operation, (time.perf_counter() - t0) * 1000.0)

    def reset(self) -> None:
        with self._lock:
            self._cold_ms.clear()
            self._samples.clear()

    def stats_for(self, operation: str) -> dict[str, Any] | None:
        with self._lock:
            buf = self._samples.get(operation)
            if not buf:
                return None
            vals = list(buf)
            warm_vals = vals[1:] if len(vals) > 1 else vals
            sorted_all = sorted(vals)
            return {
                "count": len(vals),
                "cold_ms": self._cold_ms.get(operation, vals[0]),
                "warm_mean_ms": round(statistics.mean(warm_vals), 3),
                "mean_ms": round(statistics.mean(vals), 3),
                "median_ms": round(statistics.median(vals), 3),
                "p95_ms": _percentile(sorted_all, 95.0),
                "min_ms": round(sorted_all[0], 3),
                "max_ms": round(sorted_all[-1], 3),
            }

    def snapshot(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            ops = list(self._samples.keys())
        return {op: st for op in sorted(ops) if (st := self.stats_for(op)) is not None}


_GLOBAL_PERF_MONITOR = PerformanceMonitor()


def get_perf_monitor() -> PerformanceMonitor:
    return _GLOBAL_PERF_MONITOR
