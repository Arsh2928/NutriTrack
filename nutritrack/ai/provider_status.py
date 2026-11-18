"""Structured provider status model and UI badge formatting for NutriTrack Cloud AI Routing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

DATA_ONLY_BANNER = (
    "OFFLINE MODE ACTIVE — Running locally using Python nutrition calculations and database only (no cloud AI calls)."
)


def compute_effective_label(
    *,
    selected_mode: str,
    active_provider: str,
    provider_status: dict[str, str],
    recovery_attempting: bool = False,
) -> str:
    """Compute the human-readable effective provider status label for strict 3-mode isolation."""
    if selected_mode == "offline":
        return "Offline Mode Active"
    if recovery_attempting:
        return f"Checking {selected_mode.capitalize()}"
    if selected_mode == "gemini":
        st = provider_status.get("gemini", "available")
        if st in {"config_error", "unconfigured"}:
            return "Gemini Configuration Error"
        if st == "quota_exhausted":
            return "Gemini Quota Exhausted"
        if st in {"rate_limited", "unreachable", "circuit_open", "server_error", "degraded"}:
            return "Gemini Unavailable"
        if st == "healthy":
            return "Gemini Connected"
        return "Gemini Ready"
    if selected_mode == "groq":
        st = provider_status.get("groq", "available")
        if st in {"config_error", "unconfigured"}:
            return "Groq Configuration Error"
        if st == "quota_exhausted":
            return "Groq Quota Exhausted"
        if st in {"rate_limited", "unreachable", "circuit_open", "server_error", "degraded"}:
            return "Groq Unavailable"
        if st == "healthy":
            return "Groq Connected"
        return "Groq Ready"
    return "Offline Mode Active"


@dataclass
class ProviderStatusSnapshot:
    """Complete, sanitized status snapshot returned to Flask and the Stitch UI."""

    selected_mode: str
    active_provider: str
    provider_status: dict[str, str]
    fallback_active: bool
    last_switch_reason: Optional[str] = None
    effective_label: str = "Data-only Mode"
    banner: Optional[str] = None
    last_successful_provider: Optional[str] = None
    last_successful_request_at: Optional[str] = None
    recovery_attempting: bool = False
    circuits: dict[str, Any] = field(default_factory=dict)
    models: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        is_offline = self.selected_mode == "offline"
        return {
            "selected_mode": self.selected_mode,
            "active_provider": self.active_provider,
            "provider_status": dict(self.provider_status),
            "fallback_active": False,
            "last_switch_reason": self.last_switch_reason,
            "effective_label": self.effective_label,
            "banner": DATA_ONLY_BANNER if is_offline else None,
            "data_only_active": is_offline,
            "last_successful_provider": self.last_successful_provider,
            "last_successful_request_at": self.last_successful_request_at,
            "recovery_attempting": bool(self.recovery_attempting),
            "circuits": dict(self.circuits),
            "models": dict(self.models),
        }
