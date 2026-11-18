"""Configuration loader, authoritative mode policy, and network isolation guard for NutriTrack AI."""

from __future__ import annotations

from contextlib import contextmanager
import contextvars
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Iterator, Optional

from dotenv import load_dotenv

from nutritrack.ai.exceptions import ProviderConfigError
from nutritrack.paths import PROJECT_ROOT, data_dir
from nutritrack.state.io import atomic_write_json, user_file_lock

VALID_AI_MODES = ("gemini", "groq", "offline")
DEFAULT_AI_MODE = "offline"

# Official current models supporting tool/function calling
DEFAULT_GEMINI_MODEL = "gemini-flash-lite-latest"
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Models known to be decommissioned or lacking tool-calling support
DEPRECATED_OR_UNSUPPORTED_MODELS = {
    "gemini-pro",
    "gemini-1.0-pro",
    "gemini-pro-vision",
    "text-bison-001",
    "chat-bison-001",
    "gemma-7b-it",
    "mixtral-8x7b-32768",
}

# Retired legacy .env model IDs that now return HTTP 404 upstream and map cleanly to current defaults
RETIRED_GEMINI_MODEL_ALIASES = {
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "models/gemini-2.5-flash",
    "models/gemini-2.0-flash",
}

RETIRED_GROQ_MODEL_ALIASES = {
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
    "llama3-8b-8192",
}


def normalize_provider_model(provider: str, raw_model: Optional[str]) -> str:
    """Resolve configured model name, upgrading retired 404 legacy .env defaults to current models."""
    prov = (provider or "").strip().lower()
    cleaned = (raw_model or "").strip()
    lower = cleaned.lower()
    if prov == "gemini":
        if not cleaned or lower in RETIRED_GEMINI_MODEL_ALIASES:
            return DEFAULT_GEMINI_MODEL
        return cleaned
    if prov == "groq":
        if not cleaned or lower in RETIRED_GROQ_MODEL_ALIASES:
            return DEFAULT_GROQ_MODEL
        return cleaned
    return cleaned

_MODE_LOCK = threading.RLock()
# Maps resolved base_dir string -> (mode, disk_sig_when_set)
_REGISTERED_MODES: dict[str, tuple[str, Optional[tuple[int, int]]]] = {}
_REQUEST_MODE_VAR: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "nutritrack_request_ai_mode", default=None
)
_TURN_DEADLINE_VAR: contextvars.ContextVar[Optional[float]] = contextvars.ContextVar(
    "nutritrack_turn_deadline", default=None
)
_TURN_CANCEL_EVENT_VAR: contextvars.ContextVar[Optional[threading.Event]] = contextvars.ContextVar(
    "nutritrack_turn_cancel", default=None
)


def load_env_file(*, force: bool = False) -> None:
    """Load `.env` from project root when not running inside isolated unit tests."""
    if not force and ("pytest" in sys.modules or "PYTEST_CURRENT_TEST" in os.environ):
        return
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)


def normalize_ai_mode(raw: Optional[str], *, default: str = DEFAULT_AI_MODE) -> str:
    """Normalize mode string strictly to one of `('gemini', 'groq', 'offline')`."""
    if raw is None:
        return default
    cleaned = str(raw).strip().lower()
    if cleaned == "gemini":
        return "gemini"
    if cleaned == "groq":
        return "groq"
    if cleaned == "offline":
        return "offline"
    raise ValueError(
        f"Invalid AI mode '{raw}'. Must be one of: gemini, groq, offline."
    )


def _parse_int(name: str, default: int, low: int, high: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        val = int(raw)
        return max(low, min(high, val))
    except ValueError:
        return default


def _parse_float(name: str, default: float, low: float, high: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        val = float(raw)
        return max(low, min(high, val))
    except ValueError:
        return default


def _persisted_mode_path(base_dir: Optional[Path] = None) -> Path:
    root = Path(base_dir) if base_dir else data_dir()
    return root / "ai_mode_settings.json"


def _mode_disk_sig(path: Path) -> Optional[tuple[int, int]]:
    try:
        st = path.stat()
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def _quarantine_corrupt_mode_file(path: Path) -> None:
    from datetime import datetime
    import uuid

    stamp = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    corrupt_path = path.with_name(f"{path.name}.corrupt-{stamp}")
    try:
        os.replace(path, corrupt_path)
    except OSError:
        pass


def load_persisted_mode(base_dir: Optional[Path] = None) -> Optional[str]:
    """Read user-selected mode persisted from the web UI, quarantining corrupt/legacy files."""
    path = _persisted_mode_path(base_dir)
    if not path.exists():
        return None
    lock_path = path.with_name(f"{path.name}.lock")
    with user_file_lock(lock_path):
        if not path.exists():
            return None
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(doc, dict) or not isinstance(doc.get("selected_mode"), str) or not doc["selected_mode"].strip():
                _quarantine_corrupt_mode_file(path)
                return None
            return normalize_ai_mode(doc["selected_mode"])
        except Exception:
            _quarantine_corrupt_mode_file(path)
            return None


def save_persisted_mode(mode: str, base_dir: Optional[Path] = None) -> str:
    """Persist user-selected mode (`gemini`, `groq`, `offline`) atomically under cross-process lock."""
    normalized = normalize_ai_mode(mode)
    path = _persisted_mode_path(base_dir)
    lock_path = path.with_name(f"{path.name}.lock")
    with user_file_lock(lock_path):
        atomic_write_json(path, {"selected_mode": normalized})
        sig = _mode_disk_sig(path)
    key = str((Path(base_dir) if base_dir else data_dir()).resolve())
    with _MODE_LOCK:
        _REGISTERED_MODES[key] = (normalized, sig)
    return normalized


def register_active_ai_mode(mode: str, base_dir: Optional[Path] = None) -> str:
    """Register the authoritative active AI mode for `base_dir` in-process."""
    normalized = normalize_ai_mode(mode)
    path = _persisted_mode_path(base_dir)
    key = str((Path(base_dir) if base_dir else data_dir()).resolve())
    with _MODE_LOCK:
        _REGISTERED_MODES[key] = (normalized, _mode_disk_sig(path))
    return normalized


def clear_registered_ai_mode(base_dir: Optional[Path] = None) -> None:
    """Clear cached active mode registration (used during runtime reset in tests)."""
    with _MODE_LOCK:
        if base_dir is None:
            _REGISTERED_MODES.clear()
        else:
            key = str(Path(base_dir).resolve())
            _REGISTERED_MODES.pop(key, None)


@contextmanager
def ai_mode_scope(mode: str) -> Iterator[str]:
    """Bind an immutable per-request AI mode snapshot for the current execution context."""
    normalized = normalize_ai_mode(mode)
    token = _REQUEST_MODE_VAR.set(normalized)
    try:
        yield normalized
    finally:
        _REQUEST_MODE_VAR.reset(token)


@contextmanager
def turn_execution_scope(
    timeout_seconds: float,
    cancel_event: Optional[threading.Event] = None,
) -> Iterator[threading.Event]:
    """Bind a per-turn monotonic deadline and cooperative cancellation event."""
    ev = cancel_event or threading.Event()
    deadline = time.monotonic() + max(0.1, float(timeout_seconds))
    tok_d = _TURN_DEADLINE_VAR.set(deadline)
    tok_c = _TURN_CANCEL_EVENT_VAR.set(ev)
    try:
        yield ev
    finally:
        _TURN_CANCEL_EVENT_VAR.reset(tok_c)
        _TURN_DEADLINE_VAR.reset(tok_d)


def is_turn_cancelled() -> bool:
    """True when the current turn was cancelled or exceeded its deadline."""
    ev = _TURN_CANCEL_EVENT_VAR.get()
    if ev is not None and ev.is_set():
        return True
    deadline = _TURN_DEADLINE_VAR.get()
    return deadline is not None and time.monotonic() >= deadline


def remaining_turn_timeout(default: float) -> float:
    """Return min(default, remaining_deadline) bounded above 0.05s when a turn deadline is active."""
    deadline = _TURN_DEADLINE_VAR.get()
    if deadline is None:
        return float(default)
    rem = deadline - time.monotonic()
    return max(0.05, min(float(default), rem))


def get_active_ai_mode(base_dir: Optional[Path] = None) -> str:
    """Return the authoritative currently selected mode (`gemini`, `groq`, or `offline`).

    Never infers a mode from API keys or legacy `automatic`/`online` values.
    """
    scoped = _REQUEST_MODE_VAR.get()
    if scoped is not None:
        return scoped

    root = Path(base_dir) if base_dir else data_dir()
    key = str(root.resolve())
    path = _persisted_mode_path(base_dir)
    disk_sig = _mode_disk_sig(path)

    with _MODE_LOCK:
        reg = _REGISTERED_MODES.get(key)
        if reg is not None:
            reg_mode, reg_sig = reg
            if disk_sig == reg_sig:
                return reg_mode

    persisted = load_persisted_mode(base_dir)
    if persisted is not None:
        with _MODE_LOCK:
            _REGISTERED_MODES[key] = (persisted, _mode_disk_sig(path))
        return persisted

    env_raw = os.environ.get("NUTRITRACK_AI_MODE", "").strip()
    if env_raw:
        try:
            return normalize_ai_mode(env_raw, default=DEFAULT_AI_MODE)
        except ValueError:
            return DEFAULT_AI_MODE

    return DEFAULT_AI_MODE


def is_offline_mode(base_dir: Optional[Path] = None) -> bool:
    """True when the authoritative selected mode is `offline`."""
    return get_active_ai_mode(base_dir) == "offline"


def assert_network_allowed(
    purpose: str,
    *,
    provider: Optional[str] = None,
    base_dir: Optional[Path] = None,
) -> str:
    """Centralized network-access policy guard enforcing strict 3-mode AI & Offline isolation."""
    mode = get_active_ai_mode(base_dir)
    if mode == "offline":
        raise ProviderConfigError(
            f"External network access for '{purpose}' is prohibited in Offline mode.",
            provider="data_only",
            category="offline_network_blocked",
            is_config_error=True,
        )
    if purpose == "ai":
        prov = (provider or "").strip().lower()
        if prov not in ("gemini", "groq") or prov != mode:
            raise ProviderConfigError(
                f"AI provider '{prov or 'unknown'}' is blocked while selected mode is '{mode}'.",
                provider=prov or mode,
                category="provider_isolation_blocked",
                is_config_error=True,
            )
    return mode


def is_model_supported(provider: str, model_name: Optional[str]) -> tuple[bool, Optional[str]]:
    """Validate whether a configured model ID is supported for tool-calling."""
    if not model_name or not str(model_name).strip():
        return False, f"No model configured for {provider}."
    cleaned = str(model_name).strip().lower()
    if cleaned in DEPRECATED_OR_UNSUPPORTED_MODELS or "unsupported" in cleaned or "invalid-model" in cleaned:
        return (
            False,
            f"Model '{model_name}' for {provider} is deprecated or does not support required tool calling.",
        )
    return True, None


@dataclass
class AIProviderConfig:
    """Validated runtime configuration for Gemini, Groq, retries, circuit breakers, and recovery."""

    selected_mode: str = DEFAULT_AI_MODE
    gemini_api_key: str = ""
    gemini_model: str = DEFAULT_GEMINI_MODEL
    gemini_base_url: str = DEFAULT_GEMINI_BASE_URL
    groq_api_key: str = ""
    groq_model: str = DEFAULT_GROQ_MODEL
    groq_base_url: str = DEFAULT_GROQ_BASE_URL
    max_retries: int = 2
    request_timeout_seconds: float = 20.0
    retry_base_delay_seconds: float = 1.0
    retry_max_delay_seconds: float = 8.0
    circuit_failure_threshold: int = 3
    circuit_cooldown_seconds: float = 60.0
    health_check_interval_seconds: float = 60.0

    @classmethod
    def from_env(cls, *, persisted_dir: Optional[Path] = None) -> "AIProviderConfig":
        """Load config without ever inferring provider choice from API keys or legacy modes."""
        load_env_file()
        env_mode_raw = os.environ.get("NUTRITRACK_AI_MODE", "").strip()
        persisted = load_persisted_mode(persisted_dir)
        if persisted is not None:
            mode = persisted
        elif env_mode_raw:
            try:
                mode = normalize_ai_mode(env_mode_raw, default=DEFAULT_AI_MODE)
            except ValueError:
                mode = DEFAULT_AI_MODE
        else:
            mode = DEFAULT_AI_MODE

        gemini_key = (
            os.environ.get("GEMINI_API_KEY", "").strip()
            or os.environ.get("GOOGLE_API_KEY", "").strip()
        )
        gemini_model = normalize_provider_model("gemini", os.environ.get("GEMINI_MODEL", ""))
        gemini_base_url = os.environ.get("GEMINI_BASE_URL", "").strip() or DEFAULT_GEMINI_BASE_URL

        groq_key = os.environ.get("GROQ_API_KEY", "").strip()
        groq_model = normalize_provider_model("groq", os.environ.get("GROQ_MODEL", ""))
        groq_base_url = os.environ.get("GROQ_BASE_URL", "").strip() or DEFAULT_GROQ_BASE_URL

        return cls(
            selected_mode=mode,
            gemini_api_key=gemini_key,
            gemini_model=gemini_model,
            gemini_base_url=gemini_base_url,
            groq_api_key=groq_key,
            groq_model=groq_model,
            groq_base_url=groq_base_url,
            max_retries=_parse_int("AI_MAX_RETRIES", 2, 0, 5),
            request_timeout_seconds=_parse_float("AI_REQUEST_TIMEOUT_SECONDS", 20.0, 1.0, 120.0),
            retry_base_delay_seconds=_parse_float("AI_RETRY_BASE_DELAY_SECONDS", 1.0, 0.0, 30.0),
            retry_max_delay_seconds=_parse_float("AI_RETRY_MAX_DELAY_SECONDS", 8.0, 0.05, 60.0),
            circuit_failure_threshold=_parse_int("AI_CIRCUIT_FAILURE_THRESHOLD", 3, 1, 20),
            circuit_cooldown_seconds=_parse_float("AI_CIRCUIT_COOLDOWN_SECONDS", 60.0, 0.5, 3600.0),
            health_check_interval_seconds=_parse_float("AI_HEALTH_CHECK_INTERVAL_SECONDS", 60.0, 1.0, 3600.0),
        )

    @property
    def gemini_configured(self) -> bool:
        return bool(self.gemini_api_key and self.gemini_api_key.strip())

    @property
    def groq_configured(self) -> bool:
        return bool(self.groq_api_key and self.groq_api_key.strip())

    def to_public_dict(self) -> dict[str, Any]:
        """Return safe configuration metadata with zero secrets or API keys."""
        gem_ok, gem_err = is_model_supported("gemini", self.gemini_model)
        groq_ok, groq_err = is_model_supported("groq", self.groq_model)
        return {
            "selected_mode": self.selected_mode,
            "gemini": {
                "configured": self.gemini_configured,
                "model": self.gemini_model,
                "model_supported": gem_ok,
                "model_error": gem_err,
            },
            "groq": {
                "configured": self.groq_configured,
                "model": self.groq_model,
                "model_supported": groq_ok,
                "model_error": groq_err,
            },
            "retry": {
                "max_retries": self.max_retries,
                "request_timeout_seconds": self.request_timeout_seconds,
                "base_delay_seconds": self.retry_base_delay_seconds,
                "max_delay_seconds": self.retry_max_delay_seconds,
            },
            "circuit_breaker": {
                "failure_threshold": self.circuit_failure_threshold,
                "cooldown_seconds": self.circuit_cooldown_seconds,
                "health_check_interval_seconds": self.health_check_interval_seconds,
            },
        }
