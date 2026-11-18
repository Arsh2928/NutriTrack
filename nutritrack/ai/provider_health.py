"""Per-provider health state tracker for NutriTrack Cloud AI Routing."""

from __future__ import annotations

from datetime import datetime
import threading
from typing import Any, Optional

from nutritrack.ai.config import AIProviderConfig, is_model_supported
from nutritrack.ai.exceptions import sanitize_error_text
from nutritrack.ai.failure_classifier import ClassifiedFailure


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


class ProviderHealthTracker:
    """Thread-safe tracker for Gemini, Groq, and Data-only health and switch metadata."""

    def __init__(self, config: AIProviderConfig) -> None:
        self._lock = threading.RLock()
        self._config = config
        self._status: dict[str, str] = {
            "gemini": "unconfigured",
            "groq": "unconfigured",
            "data_only": "ready",
        }
        self._details: dict[str, dict[str, Any]] = {
            "gemini": {},
            "groq": {},
            "data_only": {"status": "ready"},
        }
        self.last_successful_provider: Optional[str] = None
        self.last_successful_request_at: Optional[str] = None
        self.last_switch_reason: Optional[str] = None
        self.recovery_attempting: bool = False
        self.sync_with_config(config)

    def sync_with_config(self, config: AIProviderConfig) -> None:
        """Initialize or refresh provider baseline status from configuration and mode."""
        with self._lock:
            self._config = config
            self._status["data_only"] = "ready"

            if config.selected_mode == "offline":
                self._status["gemini"] = "disabled_offline"
                self._status["groq"] = "disabled_offline"
                return

            # Gemini baseline
            if not config.gemini_configured:
                self._status["gemini"] = "unconfigured"
            else:
                ok_model, reason = is_model_supported("gemini", config.gemini_model)
                if not ok_model:
                    self._status["gemini"] = "config_error"
                    self._details["gemini"]["last_error_category"] = "unsupported_model"
                    self._details["gemini"]["last_error_reason"] = sanitize_error_text(reason)
                elif self._status.get("gemini") in {"unconfigured", "disabled_offline"}:
                    self._status["gemini"] = "available"

            # Groq baseline
            if not config.groq_configured:
                self._status["groq"] = "unconfigured"
            else:
                ok_model, reason = is_model_supported("groq", config.groq_model)
                if not ok_model:
                    self._status["groq"] = "config_error"
                    self._details["groq"]["last_error_category"] = "unsupported_model"
                    self._details["groq"]["last_error_reason"] = sanitize_error_text(reason)
                elif self._status.get("groq") in {"unconfigured", "disabled_offline"}:
                    self._status["groq"] = "available"

    def record_success(self, provider: str) -> None:
        with self._lock:
            ts = _now_iso()
            if provider in {"gemini", "groq"}:
                self._status[provider] = "healthy"
            elif provider == "data_only":
                self._status["data_only"] = "ready"
            info = self._details.setdefault(provider, {})
            info["last_checked_at"] = ts
            info["last_success_at"] = ts
            info["last_error_category"] = None
            info["last_error_reason"] = None
            self.last_successful_provider = provider
            self.last_successful_request_at = ts

    def record_failure(
        self,
        provider: str,
        failure: ClassifiedFailure,
        *,
        circuit_opened: bool = False,
    ) -> None:
        with self._lock:
            ts = _now_iso()
            if failure.is_config_error:
                status = "config_error"
            elif failure.is_quota_exhausted:
                status = "quota_exhausted"
            elif failure.is_rate_limited or failure.category in {"rate_limit", "rate_limit_temporary"}:
                status = "rate_limited"
            elif failure.is_network_error or failure.category in {"timeout", "connection_timeout", "response_timeout"}:
                status = "unreachable"
            elif circuit_opened:
                status = "circuit_open"
            elif failure.category == "server_error":
                status = "server_error"
            else:
                status = "circuit_open" if circuit_opened else "degraded"

            self._status[provider] = status
            info = self._details.setdefault(provider, {})
            info["last_checked_at"] = ts
            info["last_failure_at"] = ts
            info["last_error_category"] = failure.category
            info["last_error_reason"] = failure.sanitized_reason

    def set_switch_reason(self, reason: Optional[str]) -> None:
        with self._lock:
            self.last_switch_reason = sanitize_error_text(reason) if reason else None

    def set_recovery_attempting(self, flag: bool) -> None:
        with self._lock:
            self.recovery_attempting = bool(flag)

    def status_map(self) -> dict[str, str]:
        with self._lock:
            return dict(self._status)

    def details_map(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {k: dict(v) for k, v in self._details.items()}
