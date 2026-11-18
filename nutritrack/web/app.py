"""Flask server and REST APIs for NutriTrack (CA2 multi-agent backend + preserved CA1 legacy endpoints)."""

from __future__ import annotations

from collections import OrderedDict
import concurrent.futures
import contextvars
import copy
from datetime import date, datetime, timedelta
import hmac
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any, Optional

from flask import Flask, jsonify, make_response, request, send_from_directory
from werkzeug.exceptions import HTTPException

from nutritrack.agents import NutriTrackOrchestrator, detect_llm_config
from nutritrack.ai.config import ai_mode_scope, clear_registered_ai_mode, turn_execution_scope
from nutritrack.ai.exceptions import ProviderConfigError, sanitize_error_text
from nutritrack.ca1 import agent as rule_agent
from nutritrack.ca1 import daily_store, food_database, tools as ca1_tools
from nutritrack.core.nutrition import coerce_diary_portion, resolve_item
from nutritrack.core.quantity import parse_item_text
from nutritrack.paths import data_dir, logs_dir
from nutritrack.state import (
    DurableOperationStore,
    OperationStoreCorruptError,
    RunLog,
    StateError,
    StateManager,
    compute_payload_hash,
)
from nutritrack.state.io import LockRegistry, atomic_write_json, user_file_lock
from nutritrack.tools import analysis_tools, intake_tools, new_context, recommendation_tools
from nutritrack.tools.fast_log import fast_log_meal, fast_remove_meal, route_chat_fast
from nutritrack.tools.performance import get_perf_monitor
from nutritrack.web.auth import AuthError, AuthManager, is_auth_required

_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app = Flask(__name__, static_folder=_STATIC_DIR, static_url_path="")

_ca1_lock = threading.RLock()
_ca2_lock = threading.RLock()
_user_lock_registry = LockRegistry()
_cache_lock = threading.Lock()
_insights_cache: OrderedDict[tuple, dict[str, Any]] = OrderedDict()
_user_data_versions: OrderedDict[str, int] = OrderedDict()
_TURN_MAX_WORKERS = 16
_TURN_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=_TURN_MAX_WORKERS, thread_name_prefix="nt-turn")
_TURN_INFLIGHT_SEM = threading.BoundedSemaphore(_TURN_MAX_WORKERS)
_MAX_CACHE_ENTRIES = 128
_MAX_USER_VERSION_ENTRIES = 512
_MAX_JSON_BYTES = 262_144
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_USER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_MEAL_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_WATER_MIN_LITERS = 0.0
_WATER_MAX_LITERS = 15.0


def _user_lock(user_id: str):
    """Return a bounded, ref-counted reentrant lock context manager for `user_id`."""
    return _user_lock_registry.lock(user_id)


def _get_data_version(user_id: str) -> int:
    with _cache_lock:
        return int(_user_data_versions.get(user_id, 0))


def _bump_data_version(user_id: str) -> int:
    with _cache_lock:
        nxt = int(_user_data_versions.get(user_id, 0)) + 1
        _user_data_versions[user_id] = nxt
        _user_data_versions.move_to_end(user_id)
        while len(_user_data_versions) > _MAX_USER_VERSION_ENTRIES:
            _user_data_versions.popitem(last=False)
        stale_keys = [k for k in _insights_cache if k and k[0] == user_id]
        for k in stale_keys:
            _insights_cache.pop(k, None)
        return nxt


def _turn_timeout_seconds() -> float:
    raw = os.environ.get("NUTRITRACK_TURN_TIMEOUT_SECONDS", "25").strip()
    try:
        return max(1.0, float(raw))
    except (TypeError, ValueError):
        return 25.0


@app.after_request
def _add_security_headers(response):
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    if request.path.startswith("/api/"):
        response.headers.setdefault("Cache-Control", "no-store")
    return response


def _state_analytics_sig(st: Any) -> str:
    """Fast structural digest of persisted nutrition days & targets for cache invalidation."""
    doc = getattr(st, "_doc", None)
    if not isinstance(doc, dict):
        return ""
    return compute_payload_hash("analytics_sig", {"days": doc.get("days"), "targets": doc.get("targets")})


@app.errorhandler(OperationStoreCorruptError)
def _handle_corrupt_op_store(exc: OperationStoreCorruptError):
    return (
        jsonify(
            {
                "status": "failed",
                "error_code": "operation_store_corrupt",
                "error_message": sanitize_error_text(exc),
            }
        ),
        500,
    )


@app.errorhandler(404)
def _handle_404(exc):
    if request.path.startswith("/api/"):
        return jsonify({"status": "failed", "error_code": "not_found", "error_message": "Endpoint not found."}), 404
    return exc


@app.errorhandler(405)
def _handle_405(exc):
    if request.path.startswith("/api/"):
        return jsonify({"status": "failed", "error_code": "method_not_allowed", "error_message": "Method not allowed for this endpoint."}), 405
    return exc


@app.errorhandler(Exception)
def _handle_unexpected(exc):
    if isinstance(exc, HTTPException):
        if request.path.startswith("/api/"):
            return jsonify({"status": "failed", "error_code": "http_error", "error_message": sanitize_error_text(exc.description or str(exc))}), exc.code or 400
        return exc
    if request.path.startswith("/api/"):
        return jsonify({"status": "failed", "error_code": "internal_error", "error_message": sanitize_error_text(exc)}), 500
    return "Internal Server Error", 500


def _reject_non_finite_json(token: str) -> Any:
    raise ValueError(f"Non-finite JSON constant not allowed: {token}")


def _parse_json_dict(*, allow_empty: bool = False) -> tuple[Optional[dict[str, Any]], Optional[tuple[Any, int]]]:
    """Parse JSON request body strictly into a dict; validate size, Content-Type, and finite JSON."""
    if request.content_length is not None and request.content_length > _MAX_JSON_BYTES:
        return None, (
            jsonify({"status": "failed", "error_code": "payload_too_large", "error_message": "Request body exceeds 256 KiB limit."}),
            413,
        )
    raw_bytes = request.get_data(cache=True) or b""
    if len(raw_bytes) > _MAX_JSON_BYTES:
        return None, (
            jsonify({"status": "failed", "error_code": "payload_too_large", "error_message": "Request body exceeds 256 KiB limit."}),
            413,
        )
    if not raw_bytes.strip():
        if allow_empty:
            return {}, None
        return None, (
            jsonify({"status": "failed", "error_code": "invalid_json", "error_message": "Request body must be a JSON object."}),
            400,
        )
    mimetype = (request.mimetype or "").strip().lower()
    if mimetype and mimetype not in ("application/json", "text/json") and not mimetype.endswith("+json"):
        return None, (
            jsonify({"status": "failed", "error_code": "unsupported_media_type", "error_message": "Content-Type must be application/json."}),
            415,
        )
    try:
        parsed = json.loads(raw_bytes.decode("utf-8"), parse_constant=_reject_non_finite_json)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None, (
            jsonify({"status": "failed", "error_code": "invalid_json", "error_message": "Request body must be valid JSON without NaN/Infinity."}),
            400,
        )
    if not isinstance(parsed, dict):
        return None, (
            jsonify({"status": "failed", "error_code": "invalid_json", "error_message": "Request body must be a valid JSON object."}),
            400,
        )
    return parsed, None


def _resolve_user_id(raw: Any) -> tuple[Optional[str], Optional[tuple[Any, int]]]:
    if raw is None:
        return "demo_user", None
    if not isinstance(raw, str):
        return None, (
            jsonify({"status": "failed", "error_code": "invalid_user_id", "error_message": "user_id must be a string."}),
            400,
        )
    cleaned = raw.strip()
    if not cleaned:
        return "demo_user", None
    if not _USER_ID_RE.match(cleaned):
        return None, (
            jsonify({"status": "failed", "error_code": "invalid_user_id", "error_message": "user_id must match [A-Za-z0-9_-]{1,64}."}),
            400,
        )
    return cleaned, None


def _extract_request_id(data: dict[str, Any]) -> Optional[str]:
    raw = data.get("request_id") or request.headers.get("X-Request-Id")
    if isinstance(raw, str):
        cleaned = raw.strip()
        if cleaned and _REQUEST_ID_RE.match(cleaned):
            return cleaned
    return None


def _file_sig(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def _finite_float(raw: Any, label: str = "Value") -> float:
    """Parse `raw` as a finite float (rejecting booleans, NaN, and Inf)."""
    if isinstance(raw, bool):
        raise ValueError(f"{label} must be a finite number.")
    try:
        val = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number.") from exc
    if not math.isfinite(val):
        raise ValueError(f"{label} must be a finite number.")
    return val

# ---------------------------------------------------------------------------
# Preserved CA1 Legacy Runtime (rule-based deterministic agent only; zero hidden LLMs)
# ---------------------------------------------------------------------------

ca1_tools.reset_session()
AGENT_MODE = "rule_based"
agent = rule_agent.MealCaloricAgent()

# ---------------------------------------------------------------------------
# CA2 Multi-Agent Runtime (thread-safe, env-path-aware for tests and CLI)
# ---------------------------------------------------------------------------

_cached_paths: Optional[tuple[str, str]] = None
_state_manager: Optional[StateManager] = None
_run_log: Optional[RunLog] = None
_orchestrator: Optional[NutriTrackOrchestrator] = None
_auth_manager: Optional[AuthManager] = None
_op_store: Optional[DurableOperationStore] = None


def reset_ca2_runtime() -> None:
    """Clear cached CA2 runtime objects so updated environment paths take effect immediately."""
    global _cached_paths, _state_manager, _run_log, _orchestrator, _auth_manager, _op_store
    with _ca2_lock:
        old_paths = _cached_paths
        _cached_paths = None
        _state_manager = None
        _run_log = None
        _orchestrator = None
        _auth_manager = None
        _op_store = None
    clear_registered_ai_mode(old_paths[0] if old_paths else None)
    clear_registered_ai_mode(data_dir())
    with _cache_lock:
        _insights_cache.clear()
        _user_data_versions.clear()


def set_ca2_orchestrator(orchestrator: Optional[NutriTrackOrchestrator]) -> None:
    """Inject a custom `NutriTrackOrchestrator` (e.g., with mocked AI providers in tests)."""
    global _cached_paths, _state_manager, _run_log, _orchestrator, _auth_manager, _op_store
    with _ca2_lock:
        if orchestrator is None:
            reset_ca2_runtime()
            return
        _orchestrator = orchestrator
        _state_manager = orchestrator.state_manager
        _run_log = orchestrator.run_log
        _cached_paths = (str(data_dir()), str(logs_dir()))
        _auth_manager = AuthManager(base_dir=_cached_paths[0])
        _op_store = DurableOperationStore(base_dir=_cached_paths[0])


def _get_full_runtime() -> tuple[StateManager, RunLog, NutriTrackOrchestrator, AuthManager, DurableOperationStore]:
    """Return full runtime tuple including AuthManager and DurableOperationStore."""
    global _cached_paths, _state_manager, _run_log, _orchestrator, _auth_manager, _op_store
    with _ca2_lock:
        current_paths = (str(data_dir()), str(logs_dir()))
        if _orchestrator is None or _cached_paths != current_paths:
            ca1_default = data_dir() / "daily_log.json"
            ca1_arg: Any = ca1_default if ca1_default.exists() else None
            _state_manager = StateManager(base_dir=current_paths[0], ca1_log_path=ca1_arg)
            _run_log = RunLog(path=logs_dir() / "agent_runs.jsonl")
            _orchestrator = NutriTrackOrchestrator(
                state_manager=_state_manager,
                run_log=_run_log,
            )
            _auth_manager = AuthManager(base_dir=current_paths[0])
            _op_store = DurableOperationStore(base_dir=current_paths[0])
            _cached_paths = current_paths
        if _auth_manager is None:
            _auth_manager = AuthManager(base_dir=current_paths[0])
        if _op_store is None:
            _op_store = DurableOperationStore(base_dir=current_paths[0])
        assert _state_manager is not None and _run_log is not None and _orchestrator is not None
        return _state_manager, _run_log, _orchestrator, _auth_manager, _op_store


def _get_ca2_runtime() -> tuple[StateManager, RunLog, NutriTrackOrchestrator]:
    """Return `(StateManager, RunLog, NutriTrackOrchestrator)` matching current configured paths."""
    sm, rl, orch, _, _ = _get_full_runtime()
    return sm, rl, orch


def _extract_auth_token() -> tuple[Optional[str], str]:
    """Return `(token, source)` where source is `'bearer'`, `'cookie'`, or `'none'`."""
    auth_hdr = request.headers.get("Authorization", "")
    if isinstance(auth_hdr, str) and auth_hdr.strip():
        parts = auth_hdr.strip().split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip(), "bearer"
        return "__malformed__", "bearer"
    cookie_tok = request.cookies.get("nutritrack_session")
    if cookie_tok:
        return cookie_tok.strip(), "cookie"
    return None, "none"


def _authorize_request(
    requested_user_id: Optional[Any] = None,
    *,
    allow_user_override: bool = False,
) -> tuple[Optional[str], Optional[dict[str, Any]], Optional[tuple[Any, int]]]:
    """Enforce session auth, CSRF protection, and per-user ownership on protected routes."""
    _, _, _, auth_mgr, _ = _get_full_runtime()
    token, tok_source = _extract_auth_token()
    session_data = None
    if token is not None:
        if token == "__malformed__":
            return None, None, (
                jsonify({"status": "failed", "error_code": "unauthorized", "error_message": "Invalid Authorization header."}),
                401,
            )
        session_data = auth_mgr.verify_session(token)
        if session_data is None:
            return None, None, (
                jsonify({"status": "failed", "error_code": "unauthorized", "error_message": "Session is invalid or expired. Please sign in."}),
                401,
            )

    auth_enforced = (
        is_auth_required(
            app_require_auth=bool(app.config.get("REQUIRE_AUTH", False)),
            remote_addr=request.remote_addr,
            forwarded_for=request.headers.get("X-Forwarded-For"),
            bound_host=os.environ.get("HOST"),
        )
        or auth_mgr.has_users()
    )

    if session_data is None:
        if auth_enforced:
            return None, None, (
                jsonify({"status": "failed", "error_code": "unauthorized", "error_message": "Authentication required."}),
                401,
            )
        uid, u_err = _resolve_user_id(requested_user_id)
        return uid, None, u_err

    if tok_source == "cookie" and request.method in ("POST", "PUT", "PATCH", "DELETE"):
        csrf_hdr = (request.headers.get("X-CSRF-Token") or "").strip()
        expected_csrf = str(session_data.get("csrf_token") or "")
        if not csrf_hdr or not expected_csrf or not hmac.compare_digest(csrf_hdr, expected_csrf):
            return None, None, (
                jsonify({"status": "failed", "error_code": "csrf_failed", "error_message": "Missing or invalid CSRF token."}),
                403,
            )

    authed_user = str(session_data["user_id"])
    if requested_user_id is not None:
        if not isinstance(requested_user_id, str):
            return None, None, (
                jsonify({"status": "failed", "error_code": "invalid_user_id", "error_message": "user_id must be a string."}),
                400,
            )
        if requested_user_id.strip():
            uid, u_err = _resolve_user_id(requested_user_id)
            if u_err is not None:
                return None, None, u_err
            if uid != authed_user and not allow_user_override:
                return None, None, (
                    jsonify(
                        {
                            "status": "failed",
                            "error_code": "forbidden",
                            "error_message": "You do not have permission to access another user's resources.",
                        }
                    ),
                    403,
                )

    return authed_user, session_data, None


def _run_orchestrator_turn(
    orch: NutriTrackOrchestrator,
    message: str,
    *,
    user_id: str,
    day: Optional[str],
    selected_mode: str,
) -> dict[str, Any]:
    """Execute `orch.run_turn` under an immutable mode scope, bounded admission, and server-side deadline."""
    timeout_s = _turn_timeout_seconds()
    if not _TURN_INFLIGHT_SEM.acquire(timeout=0.25):
        return {
            "status": "failed",
            "error_code": "server_busy",
            "error_message": "Server turn executor is at capacity. Please retry shortly.",
            "reply": "Server is currently handling too many AI requests. Please retry in a moment.",
            "activated_specialists": [],
            "results": {},
        }

    cancel_event = threading.Event()
    submitted = False

    def _invoke() -> dict[str, Any]:
        try:
            with turn_execution_scope(timeout_s, cancel_event), ai_mode_scope(selected_mode):
                return orch.run_turn(message, user_id=user_id, date=day)
        finally:
            _TURN_INFLIGHT_SEM.release()

    try:
        ctx = contextvars.copy_context()
        future = _TURN_EXECUTOR.submit(ctx.run, _invoke)
        submitted = True
        return future.result(timeout=timeout_s)
    except concurrent.futures.TimeoutError:
        cancel_event.set()
        future.cancel()
        return {
            "status": "failed",
            "error_code": "turn_timeout",
            "error_message": f"Request exceeded server timeout ({timeout_s:g}s).",
            "reply": "That request timed out on the server. Please check status or retry.",
            "activated_specialists": [],
            "results": {},
        }
    except Exception:
        if not submitted:
            _TURN_INFLIGHT_SEM.release()
        raise


def _build_legacy_summary() -> dict[str, Any]:
    return ca1_tools.get_memory().summary()


def _safe_user_id(raw: Optional[str]) -> str:
    cleaned = (raw or "demo_user").strip()
    if not re.match(r"^[A-Za-z0-9_-]{1,64}$", cleaned):
        return "demo_user"
    return cleaned or "demo_user"


_DEFAULT_PROFILE: dict[str, Any] = {
    "name": "NutriTrack User",
    "email": "",
    "role": "Member",
    "weight_kg": 74.0,
    "target_weight_kg": 72.0,
    "height_cm": 178.0,
    "health_goal": "Weight Maintenance & Lean Muscle",
    "meals_per_day": "3 Meals + 1 Snack",
    "carbs_target": 250.0,
    "fat_target": 75.0,
    "water_target_liters": 2.5,
    "ai_meal_suggestions": True,
    "weekly_summary_enabled": True,
    "diet_types": ["non-vegetarian", "vegetarian", "eggetarian"],
    "water_logs": {},
}

_DEMO_PROFILE_OVERRIDES: dict[str, str] = {
    "name": "Arshdeep Singh",
    "email": "arshdeep@nutritrack.app",
    "role": "Premium Member",
}

_ALL_DIET_TYPES = ("non-vegetarian", "vegetarian", "eggetarian")


_PROFILE_NUMERIC_BOUNDS: dict[str, tuple[float, float]] = {
    "weight_kg": (20.0, 400.0),
    "target_weight_kg": (20.0, 400.0),
    "height_cm": (80.0, 260.0),
    "carbs_target": (1.0, 1500.0),
    "fat_target": (1.0, 500.0),
    "water_target_liters": (0.25, 15.0),
}
_VALID_DIET_TOKENS = {
    "any",
    "all",
    "everything",
    "non-veg",
    "non_vegetarian",
    "nonvegetarian",
    "non-vegetarian",
    "vegetarian",
    "eggetarian",
}


def _resolve_diet_selection(
    diet_raw: Any,
    diet_types_raw: Any = None,
    *,
    strict: bool = False,
) -> tuple[str, list[str]]:
    """Resolve single or multi-select diet choices into `(state_diet, diet_types_list)`."""
    raw_items: list[str] = []
    source = diet_types_raw if diet_types_raw is not None else diet_raw
    if isinstance(source, (list, tuple, set)):
        for x in source:
            if not isinstance(x, str):
                raise ValueError("diet_types entries must be strings.")
            if x.strip():
                raw_items.append(x.strip().lower())
    elif isinstance(source, str) and source.strip():
        raw_items = [x.strip().lower() for x in source.split(",") if x.strip()]
    elif source is not None and not (isinstance(source, str) and not source.strip()):
        raise ValueError("diet/diet_types must be a string or list of strings.")

    if strict:
        for item in raw_items:
            if item not in _VALID_DIET_TOKENS:
                raise ValueError(f"Unsupported diet selection '{item}'. Choose non-vegetarian, vegetarian, or eggetarian.")

    selected: list[str] = []
    for item in raw_items:
        if item in {"any", "all", "everything"}:
            for d in _ALL_DIET_TYPES:
                if d not in selected:
                    selected.append(d)
        elif item in {"non-veg", "non_vegetarian", "nonvegetarian", "non-vegetarian"}:
            if "non-vegetarian" not in selected:
                selected.append("non-vegetarian")
        elif item in _ALL_DIET_TYPES and item not in selected:
            selected.append(item)

    if not selected:
        if isinstance(diet_raw, str) and diet_raw.strip().lower() in {"vegetarian", "eggetarian"}:
            d_clean = diet_raw.strip().lower()
            selected = [d_clean] if d_clean == "vegetarian" else ["vegetarian", "eggetarian"]
        elif isinstance(diet_raw, str) and diet_raw.strip() and diet_raw.strip().lower() not in {"any", "all"}:
            return diet_raw.strip().lower(), list(_ALL_DIET_TYPES)
        else:
            selected = list(_ALL_DIET_TYPES)

    if "non-vegetarian" in selected:
        effective_diet = "any"
    elif "eggetarian" in selected:
        effective_diet = "eggetarian"
    elif "vegetarian" in selected:
        effective_diet = "vegetarian"
    else:
        effective_diet = "any"

    return effective_diet, selected


def _profile_path(user_id: str) -> Path:
    safe = _safe_user_id(user_id)
    return data_dir() / "profiles" / f"{safe}.json"


def _parse_bool_strict(raw: Any, label: str) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        low = raw.strip().lower()
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no"):
            return False
    raise ValueError(f"{label} must be a boolean.")


def _validate_water_logs(logs_raw: Any) -> dict[str, float]:
    if not isinstance(logs_raw, dict):
        raise ValueError("water_logs must be an object mapping YYYY-MM-DD dates to liter amounts.")
    cleaned: dict[str, float] = {}
    for k, v in logs_raw.items():
        if not isinstance(k, str):
            raise ValueError("water_logs keys must be ISO YYYY-MM-DD date strings.")
        day_k = k.strip()
        if not _ISO_DATE_RE.match(day_k):
            raise ValueError(f"Invalid water_logs date '{k}'; expected YYYY-MM-DD.")
        try:
            date.fromisoformat(day_k)
        except ValueError as exc:
            raise ValueError(f"Invalid calendar date '{day_k}' in water_logs.") from exc
        liters = _finite_float(v, f"water_logs[{day_k}]")
        if liters < _WATER_MIN_LITERS or liters > _WATER_MAX_LITERS:
            raise ValueError(
                f"water_logs[{day_k}] must be between {_WATER_MIN_LITERS:g} and {_WATER_MAX_LITERS:g} liters."
            )
        cleaned[day_k] = round(liters, 2)
    return cleaned


def _validate_profile_updates(updates: dict[str, Any]) -> dict[str, Any]:
    """Validate all profile fields strictly before writing anything to disk."""
    validated: dict[str, Any] = {}
    str_fields = ("name", "email", "role", "health_goal", "meals_per_day")
    bool_fields = ("ai_meal_suggestions", "weekly_summary_enabled")

    for sf in str_fields:
        if sf in updates and updates[sf] is not None:
            raw = updates[sf]
            if not isinstance(raw, str):
                raise ValueError(f"{sf} must be a string.")
            val = raw.strip()
            if val or sf == "email":
                validated[sf] = val[:100]

    for nf, (lo, hi) in _PROFILE_NUMERIC_BOUNDS.items():
        if nf in updates and updates[nf] is not None and str(updates[nf]).strip() != "":
            val = _finite_float(updates[nf], nf)
            if val < lo or val > hi:
                raise ValueError(f"{nf} must be between {lo:g} and {hi:g}.")
            validated[nf] = round(val, 2)

    for bf in bool_fields:
        if bf in updates and updates[bf] is not None:
            validated[bf] = _parse_bool_strict(updates[bf], bf)

    if "diet_types" in updates or "diet" in updates:
        _, resolved_types = _resolve_diet_selection(updates.get("diet"), updates.get("diet_types"), strict=True)
        validated["diet_types"] = resolved_types

    if "water_logs" in updates and updates["water_logs"] is not None:
        validated["water_logs"] = _validate_water_logs(updates["water_logs"])

    return validated


def _sanitize_profile_doc(user_id: str, loaded: Any, active_date: Optional[str] = None) -> dict[str, Any]:
    doc = copy.deepcopy(_DEFAULT_PROFILE)
    if user_id == "demo_user":
        doc.update(_DEMO_PROFILE_OVERRIDES)
    if isinstance(loaded, dict):
        for k, v in loaded.items():
            if k in _DEFAULT_PROFILE:
                doc[k] = v
    raw_logs = doc.get("water_logs")
    clean_logs: dict[str, float] = {}
    if isinstance(raw_logs, dict):
        for k, v in raw_logs.items():
            if not isinstance(k, str) or not _ISO_DATE_RE.match(k.strip()):
                continue
            try:
                val = _finite_float(v, "water")
            except ValueError:
                continue
            if _WATER_MIN_LITERS <= val <= _WATER_MAX_LITERS:
                clean_logs[k.strip()] = round(val, 2)
    doc["water_logs"] = clean_logs
    day_key = active_date or date.today().isoformat()
    water_today = float(clean_logs.get(day_key, 0.0) or 0.0)
    doc["water_today_liters"] = round(water_today, 2)
    doc["active_date"] = day_key
    return doc


def _load_profile(user_id: str, active_date: Optional[str] = None) -> dict[str, Any]:
    path = _profile_path(user_id)
    lock_path = path.with_name(f"{path.name}.lock")
    with _user_lock(user_id), user_file_lock(lock_path):
        loaded: Any = None
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                loaded = None
        return _sanitize_profile_doc(user_id, loaded, active_date=active_date)


def _save_profile(user_id: str, updates: dict[str, Any], active_date: Optional[str] = None) -> dict[str, Any]:
    clean = _validate_profile_updates(updates)
    path = _profile_path(user_id)
    lock_path = path.with_name(f"{path.name}.lock")
    with _user_lock(user_id), user_file_lock(lock_path):
        loaded: Any = None
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                loaded = None
        doc = _sanitize_profile_doc(user_id, loaded, active_date=active_date)
        doc.pop("water_today_liters", None)
        doc.pop("active_date", None)
        water_updates = clean.pop("water_logs", None)
        doc.update(clean)
        if water_updates is not None:
            doc["water_logs"].update(water_updates)
        atomic_write_json(path, doc)
        return _sanitize_profile_doc(user_id, doc, active_date=active_date)


def _normalize_meal_type(raw: Optional[str], *, strict: bool = False) -> Optional[str]:
    if raw is None:
        return None
    if not isinstance(raw, str):
        if strict:
            raise ValueError("meal_type must be a string.")
        return None
    cleaned = raw.strip().lower()
    if not cleaned:
        return None
    mapping = {
        "breakfast": "breakfast",
        "brunch": "brunch",
        "lunch": "lunch",
        "dinner": "dinner",
        "supper": "dinner",
        "snack": "snack",
        "snacks": "snack",
    }
    res = mapping.get(cleaned)
    if res is None and strict:
        raise ValueError("meal_type must be one of: breakfast, brunch, lunch, dinner, snack.")
    return res


# ---------------------------------------------------------------------------
# UI Entry Point + Authentication + Preserved CA1 API Endpoints
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return send_from_directory(_STATIC_DIR, "index.html")


@app.route("/api/nutritrack/auth/register", methods=["POST"])
def nt_auth_register():
    data, err = _parse_json_dict()
    if err is not None:
        return err
    assert data is not None
    user_id = data.get("user_id") or ""
    password = data.get("password") or ""
    _, _, _, auth_mgr, _ = _get_full_runtime()
    try:
        sess = auth_mgr.register(user_id, password)
    except AuthError as exc:
        return jsonify({"status": "failed", "error_code": exc.error_code, "error_message": str(exc)}), exc.status_code
    resp = make_response(
        jsonify(
            {
                "status": "success",
                "user_id": sess["user_id"],
                "csrf_token": sess["csrf_token"],
                "token": sess["token"],
                "expires_at": sess["expires_at"],
            }
        )
    )
    resp.set_cookie(
        "nutritrack_session",
        sess["token"],
        httponly=True,
        samesite="Lax",
        secure=request.is_secure,
        path="/",
    )
    return resp, 201


@app.route("/api/nutritrack/auth/login", methods=["POST"])
def nt_auth_login():
    data, err = _parse_json_dict()
    if err is not None:
        return err
    assert data is not None
    user_id = data.get("user_id") or ""
    password = data.get("password") or ""
    client_ip = (request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or request.remote_addr or "local")
    _, _, _, auth_mgr, _ = _get_full_runtime()
    try:
        sess = auth_mgr.authenticate(user_id, password, client_ip=client_ip)
    except AuthError as exc:
        resp = make_response(
            jsonify({"status": "failed", "error_code": exc.error_code, "error_message": str(exc)}),
            exc.status_code,
        )
        if exc.retry_after is not None:
            resp.headers["Retry-After"] = str(exc.retry_after)
        return resp
    resp = make_response(
        jsonify(
            {
                "status": "success",
                "user_id": sess["user_id"],
                "csrf_token": sess["csrf_token"],
                "token": sess["token"],
                "expires_at": sess["expires_at"],
            }
        )
    )
    resp.set_cookie(
        "nutritrack_session",
        sess["token"],
        httponly=True,
        samesite="Lax",
        secure=request.is_secure,
        path="/",
    )
    return resp


@app.route("/api/nutritrack/auth/logout", methods=["POST"])
def nt_auth_logout():
    _, _, _, auth_mgr, _ = _get_full_runtime()
    token, tok_source = _extract_auth_token()
    if token and token != "__malformed__":
        session_data = auth_mgr.verify_session(token)
        if session_data is not None and tok_source == "cookie":
            csrf_hdr = (request.headers.get("X-CSRF-Token") or "").strip()
            expected = str(session_data.get("csrf_token") or "")
            if not csrf_hdr or not expected or not hmac.compare_digest(csrf_hdr, expected):
                return jsonify({"status": "failed", "error_code": "csrf_failed", "error_message": "Missing or invalid CSRF token."}), 403
        auth_mgr.revoke_session(token)
    resp = make_response(jsonify({"status": "success"}))
    resp.delete_cookie("nutritrack_session", path="/")
    return resp


@app.route("/api/nutritrack/auth/me", methods=["GET"])
def nt_auth_me():
    _, _, _, auth_mgr, _ = _get_full_runtime()
    token, _ = _extract_auth_token()
    sess = auth_mgr.verify_session(token) if token and token != "__malformed__" else None
    auth_enforced = (
        is_auth_required(
            app_require_auth=bool(app.config.get("REQUIRE_AUTH", False)),
            remote_addr=request.remote_addr,
            forwarded_for=request.headers.get("X-Forwarded-For"),
            bound_host=os.environ.get("HOST"),
        )
        or auth_mgr.has_users()
    )
    if sess is not None:
        return jsonify(
            {
                "status": "success",
                "authenticated": True,
                "auth_required": auth_enforced,
                "user_id": sess["user_id"],
                "csrf_token": sess["csrf_token"],
                "expires_at": sess["expires_at"],
            }
        )
    return jsonify(
        {
            "status": "success",
            "authenticated": False,
            "auth_required": auth_enforced,
            "user_id": None if auth_enforced else "demo_user",
        }
    )


@app.route("/api/nutritrack/operations/<request_id>", methods=["GET"])
def nt_operation_status(request_id: str):
    if not _REQUEST_ID_RE.match(request_id):
        return jsonify({"status": "failed", "error_code": "invalid_input", "error_message": "Invalid request_id format."}), 400
    user_id, _, auth_err = _authorize_request(request.args.get("user_id"))
    if auth_err is not None:
        return auth_err
    assert user_id is not None
    _, _, _, _, op_store = _get_full_runtime()
    rec = op_store.get(user_id, request_id)
    if rec is None:
        return jsonify({"status": "failed", "error_code": "not_found", "error_message": f"Operation '{request_id}' not found."}), 404
    return jsonify(
        {
            "status": "success",
            "request_id": request_id,
            "user_id": user_id,
            "operation_status": rec.get("status"),
            "status_code": rec.get("status_code"),
            "response": rec.get("response"),
            "error_message": rec.get("error_message"),
            "updated_at": rec.get("updated_at"),
        }
    )


@app.route("/api/chat", methods=["POST"])
def legacy_chat():
    """Preserved deterministic CA1 endpoint (no GitHub Models or hidden LLMs)."""
    data, err = _parse_json_dict()
    if err is not None:
        resp_obj, code = err
        if code in (413, 415):
            return resp_obj, code
        return jsonify({"error": "JSON object body required."}), 400
    assert data is not None
    _, _, auth_err = _authorize_request(data.get("user_id"))
    if auth_err is not None:
        return auth_err

    req_mode = data.get("mode") or data.get("provider")
    if isinstance(req_mode, str) and req_mode.strip():
        low_mode = req_mode.strip().lower()
        if low_mode in ("github", "github_models", "openai", "automatic", "auto", "online"):
            return (
                jsonify(
                    {
                        "status": "failed",
                        "error": f"Legacy mode '{low_mode}' is unsupported. Choose 'gemini', 'groq', or 'offline' explicitly.",
                        "error_code": "unsupported_mode",
                    }
                ),
                400,
            )

    raw_msg = data.get("message")
    if not isinstance(raw_msg, str):
        return jsonify({"error": "Message cannot be empty."}), 400
    message = raw_msg.strip()
    if not message:
        return jsonify({"error": "Message cannot be empty."}), 400

    with _ca1_lock, ai_mode_scope("offline"):
        try:
            result = agent.chat_capture(message)
        except Exception as exc:
            fallback = rule_agent.MealCaloricAgent()
            result = fallback.chat_capture(message)
            result["trace"] = f"[Server Fallback] {sanitize_error_text(exc)}\n" + result.get("trace", "")

        daily_store.sync_after_change(ca1_tools.get_memory())
        return jsonify(
            {
                "response": result["response"],
                "trace": result["trace"],
                "summary": _build_legacy_summary(),
                "agent_mode": AGENT_MODE,
            }
        )


@app.route("/api/summary", methods=["GET"])
def legacy_summary():
    _, _, auth_err = _authorize_request(request.args.get("user_id"))
    if auth_err is not None:
        return auth_err
    with _ca1_lock:
        return jsonify(_build_legacy_summary())


@app.route("/api/history", methods=["GET"])
def legacy_history():
    _, _, auth_err = _authorize_request(request.args.get("user_id"))
    if auth_err is not None:
        return auth_err
    with _ca1_lock:
        return jsonify(daily_store.get_history())


@app.route("/api/mode", methods=["GET"])
def legacy_mode():
    return jsonify({"agent_mode": AGENT_MODE, "model": None})


# ---------------------------------------------------------------------------
# CA2 Multi-Agent NutriTrack API Endpoints (/api/nutritrack/*)
# ---------------------------------------------------------------------------


@app.route("/api/nutritrack/mode", methods=["GET"])
def nt_mode():
    _, _, auth_err = _authorize_request()
    if auth_err is not None:
        return auth_err
    _, _, orch = _get_ca2_runtime()
    info = orch.get_mode_info()
    info.pop("api_key", None)
    return jsonify(info)


@app.route("/api/nutritrack/ai/status", methods=["GET"])
def nt_ai_status():
    _, _, auth_err = _authorize_request()
    if auth_err is not None:
        return auth_err
    with _ca2_lock:
        _, _, orch = _get_ca2_runtime()
        return jsonify(orch.ai_manager.get_status())


@app.route("/api/nutritrack/ai/mode", methods=["GET", "POST"])
def nt_ai_mode():
    _, _, auth_err = _authorize_request()
    if auth_err is not None:
        return auth_err
    with _ca2_lock:
        _, _, orch = _get_ca2_runtime()
        if request.method == "GET":
            st = orch.ai_manager.get_status()
            return jsonify(
                {
                    "status": "success",
                    "selected_mode": st["selected_mode"],
                    "active_provider": st["active_provider"],
                    "effective_label": st["effective_label"],
                    "fallback_active": st["fallback_active"],
                    "banner": st["banner"],
                    "ai_status": st,
                }
            )
        data, err = _parse_json_dict()
        if err is not None:
            return err
        assert data is not None
        raw_mode = data.get("mode")
        if not raw_mode or not isinstance(raw_mode, str):
            return (
                jsonify(
                    {
                        "status": "failed",
                        "error_code": "invalid_mode",
                        "error_message": "Provide a valid mode: 'gemini', 'groq', or 'offline'.",
                    }
                ),
                400,
            )
        try:
            updated = orch.ai_manager.set_mode(raw_mode, persist=True)
        except ValueError as exc:
            return (
                jsonify(
                    {
                        "status": "failed",
                        "error_code": "invalid_mode",
                        "error_message": str(exc),
                    }
                ),
                400,
            )
        except (ProviderConfigError, OSError) as exc:
            return (
                jsonify(
                    {
                        "status": "failed",
                        "error_code": "config_persist_failed",
                        "error_message": sanitize_error_text(exc),
                    }
                ),
                500,
            )
        return jsonify(
            {
                "status": "success",
                "selected_mode": updated["selected_mode"],
                "active_provider": updated["active_provider"],
                "effective_label": updated["effective_label"],
                "fallback_active": updated["fallback_active"],
                "banner": updated["banner"],
                "ai_status": updated,
            }
        )


@app.route("/api/nutritrack/ai/providers", methods=["GET"])
def nt_ai_providers():
    _, _, auth_err = _authorize_request()
    if auth_err is not None:
        return auth_err
    with _ca2_lock:
        _, _, orch = _get_ca2_runtime()
        return jsonify(orch.ai_manager.get_providers_diagnostics())


@app.route("/api/nutritrack/ai/retry", methods=["POST"])
def nt_ai_retry():
    _, _, auth_err = _authorize_request()
    if auth_err is not None:
        return auth_err
    with _ca2_lock:
        _, _, orch = _get_ca2_runtime()
        updated = orch.ai_manager.retry_providers(force=True)
        return jsonify(
            {
                "status": "success",
                "selected_mode": updated["selected_mode"],
                "active_provider": updated["active_provider"],
                "effective_label": updated["effective_label"],
                "fallback_active": updated["fallback_active"],
                "ai_status": updated,
            }
        )


@app.route("/api/nutritrack/users", methods=["GET"])
def nt_users():
    _, sess, auth_err = _authorize_request()
    if auth_err is not None:
        return auth_err
    with _ca2_lock:
        sm, _, _ = _get_ca2_runtime()
        if sess is not None:
            authed_user = str(sess["user_id"])
            sm.get(authed_user)
            return jsonify({"users": [authed_user], "default_user": authed_user})
        users = sm.list_users()
        if "demo_user" not in users:
            sm.get("demo_user")
            users = sm.list_users()
        return jsonify({"users": users, "default_user": "demo_user"})


@app.route("/api/nutritrack/state", methods=["GET"])
def nt_state():
    user_id, _, auth_err = _authorize_request(request.args.get("user_id"))
    if auth_err is not None:
        return auth_err
    assert user_id is not None
    day = (request.args.get("date") or "").strip() or None
    t0 = time.perf_counter()
    with _user_lock(user_id):
        sm, rl, orch = _get_ca2_runtime()
        st = sm.get(user_id)
        try:
            snap = st.snapshot(day=day, history_days=7)
        except StateError as exc:
            return jsonify({"status": "failed", "error_code": "invalid_input", "error_message": str(exc)}), 400
        ctx = new_context(sm, rl, user_id=user_id, task="dashboard_state")
        macro_split = analysis_tools.calculate_macro_split(ctx, date=day)
        recent_runs = rl.read(limit=25, user_id=user_id, exclude_user_ids=("system",))
        profile = _load_profile(user_id, active_date=snap["date"])
        mode_info = orch.get_mode_info()
    get_perf_monitor().record("get_state", (time.perf_counter() - t0) * 1000.0)
    return jsonify(
        {
            "user_id": user_id,
            "state": snap,
            "profile": profile,
            "macro_split": macro_split,
            "workflow": st.get_workflow(),
            "agent_results": st.get_agent_results(),
            "open_questions": st.get_open_questions(),
            "recent_runs": recent_runs,
            "mode_info": mode_info,
            "ai_status": mode_info.get("ai_status") or orch.ai_manager.get_status(),
        }
    )


@app.route("/api/nutritrack/chat", methods=["POST"])
def nt_chat():
    data, err = _parse_json_dict()
    if err is not None:
        return err
    assert data is not None
    user_id, _, auth_err = _authorize_request(data.get("user_id"))
    if auth_err is not None:
        return auth_err
    assert user_id is not None

    raw_msg = data.get("message")
    if not isinstance(raw_msg, str):
        return jsonify({"status": "failed", "error_code": "empty_message", "error_message": "Message must be a non-empty string."}), 400
    message = raw_msg.strip()
    if not message:
        return jsonify({"status": "failed", "error_code": "empty_message", "error_message": "Message cannot be empty."}), 400
    if len(message) > 1000:
        return jsonify({"status": "failed", "error_code": "input_too_long", "error_message": "Message exceeds 1000 character limit."}), 400

    day_raw = data.get("date")
    if day_raw is not None and not isinstance(day_raw, str):
        return jsonify({"status": "failed", "error_code": "invalid_input", "error_message": "date must be a string."}), 400
    day = (day_raw or "").strip() or None

    raw_req_id = data.get("request_id")
    if raw_req_id is not None and (not isinstance(raw_req_id, str) or (raw_req_id.strip() and not _REQUEST_ID_RE.match(raw_req_id.strip()))):
        return jsonify({"status": "failed", "error_code": "invalid_input", "error_message": "Invalid request_id format."}), 400
    request_id = _extract_request_id(data)

    sm, rl, orch, _, op_store = _get_full_runtime()
    selected_mode = orch.ai_manager.selected_mode

    if day is not None:
        try:
            sm.get(user_id).resolve_date(day)
        except StateError as exc:
            return jsonify({"status": "failed", "error_code": "invalid_input", "error_message": str(exc)}), 400

    if request_id:
        p_hash = compute_payload_hash("chat", {"user_id": user_id, "message": message, "date": day})
        op_action, op_rec = op_store.begin(user_id, request_id, endpoint="chat", payload_hash=p_hash)
        if op_action == "conflict":
            return (
                jsonify(
                    {
                        "status": "failed",
                        "error_code": "idempotency_conflict",
                        "error_message": "request_id was already used for a different request payload.",
                    }
                ),
                409,
            )
        if op_action == "in_progress":
            settled = op_store.wait_if_in_progress(user_id, request_id, timeout=3.0)
            if settled and settled.get("status") == "succeeded" and isinstance(settled.get("response"), dict):
                replay = copy.deepcopy(settled["response"])
                replay["duplicate"] = True
                return jsonify(replay), int(settled.get("status_code") or 200)
            return (
                jsonify(
                    {
                        "status": "pending",
                        "error_code": "operation_in_progress",
                        "error_message": "An identical request with this request_id is currently processing.",
                        "request_id": request_id,
                    }
                ),
                409,
            )
        if op_action == "replay" and isinstance(op_rec.get("response"), dict):
            replay = copy.deepcopy(op_rec["response"])
            replay["duplicate"] = True
            return jsonify(replay), int(op_rec.get("status_code") or 200)

    t0 = time.perf_counter()
    completed_recorded = False
    try:
        with _user_lock(user_id), ai_mode_scope(selected_mode):
            sig_before = _state_analytics_sig(sm.get(user_id))
            ai_st = orch.ai_manager.get_status()
            mode_info = orch.get_mode_info()
            mode_info.pop("api_key", None)
            result: Optional[dict[str, Any]] = None
            if getattr(orch, "_explicit_client", None) is None:
                result = route_chat_fast(
                    sm,
                    rl,
                    message=message,
                    user_id=user_id,
                    date=day,
                    request_id=request_id,
                    ai_status=ai_st,
                    mode_info=mode_info,
                )
            if result is None:
                result = _run_orchestrator_turn(orch, message, user_id=user_id, day=day, selected_mode=selected_mode)
            sig_after = _state_analytics_sig(sm.get(user_id))

        err_code = result.get("error_code")
        if result.get("status") == "failed":
            if err_code in ("llm_unavailable", "provider_error", "server_busy"):
                status_code = 503
            elif err_code == "turn_timeout":
                status_code = 504
            elif err_code in ("invalid_date", "ambiguous_date", "invalid_input"):
                status_code = 400
            else:
                status_code = 400
        else:
            status_code = 200

        if sig_after != sig_before or result.get("added") or result.get("removed"):
            _bump_data_version(user_id)

        if request_id:
            if status_code == 200 and result.get("status") != "failed":
                op_store.complete(
                    user_id,
                    request_id,
                    status="succeeded",
                    status_code=status_code,
                    response=result,
                )
            else:
                op_store.complete(
                    user_id,
                    request_id,
                    status="failed",
                    status_code=status_code,
                    error_message=result.get("error_message") or result.get("response"),
                )
            completed_recorded = True

        get_perf_monitor().record("post_chat", (time.perf_counter() - t0) * 1000.0)
        return jsonify(result), status_code
    except Exception as exc:
        if request_id and not completed_recorded:
            op_store.complete(
                user_id,
                request_id,
                status="failed",
                status_code=500,
                error_message=sanitize_error_text(exc),
            )
        raise


@app.route("/api/nutritrack/goals", methods=["POST"])
def nt_goals():
    data, err = _parse_json_dict()
    if err is not None:
        return err
    assert data is not None
    user_id, _, auth_err = _authorize_request(data.get("user_id"))
    if auth_err is not None:
        return auth_err
    assert user_id is not None
    day_raw = data.get("date")
    if day_raw is not None and not isinstance(day_raw, str):
        return jsonify({"status": "failed", "error_code": "invalid_input", "error_message": "date must be a string."}), 400
    day = (day_raw or "").strip() or None
    cal_raw = data.get("calories")
    prot_raw = data.get("protein")

    parts: list[str] = []
    try:
        if cal_raw is not None and str(cal_raw).strip() != "":
            cal_val = _finite_float(cal_raw, "calories")
            if cal_val <= 0:
                return jsonify({"status": "failed", "error_code": "invalid_input", "error_message": "Calorie target must be positive."}), 400
            parts.append(f"calorie goal to {cal_val:g} kcal")
        if prot_raw is not None and str(prot_raw).strip() != "":
            prot_val = _finite_float(prot_raw, "protein")
            if prot_val <= 0:
                return jsonify({"status": "failed", "error_code": "invalid_input", "error_message": "Protein target must be positive."}), 400
            parts.append(f"protein goal to {prot_val:g} g")
    except ValueError:
        return jsonify({"status": "failed", "error_code": "invalid_input", "error_message": "Calorie and protein goals must be finite numbers."}), 400

    if not parts:
        return jsonify({"status": "failed", "error_code": "invalid_input", "error_message": "Provide a calorie and/or protein target."}), 400

    prompt = "Set my daily " + " and ".join(parts) + "."
    with _user_lock(user_id):
        _, _, orch = _get_ca2_runtime()
        selected_mode = orch.ai_manager.selected_mode
        result = _run_orchestrator_turn(orch, prompt, user_id=user_id, day=day, selected_mode=selected_mode)
        if result.get("status") == "success":
            _bump_data_version(user_id)
    if result.get("status") == "failed":
        code = 503 if result.get("error_code") == "llm_unavailable" else 400
        return jsonify(result), code
    return jsonify(result)


@app.route("/api/nutritrack/preferences", methods=["POST"])
def nt_preferences():
    data, err = _parse_json_dict()
    if err is not None:
        return err
    assert data is not None
    user_id, _, auth_err = _authorize_request(data.get("user_id"))
    if auth_err is not None:
        return auth_err
    assert user_id is not None
    diet_raw = data.get("diet")
    diet_types_raw = data.get("diet_types")
    avoid = data.get("avoid")
    prefer = data.get("prefer")

    if isinstance(avoid, str):
        avoid = [x.strip() for x in avoid.split(",") if x.strip()]
    elif avoid is not None and not isinstance(avoid, list):
        return jsonify({"status": "failed", "error_code": "invalid_input", "error_message": "avoid must be a string or list."}), 400
    if isinstance(prefer, str):
        prefer = [x.strip() for x in prefer.split(",") if x.strip()]
    elif prefer is not None and not isinstance(prefer, list):
        return jsonify({"status": "failed", "error_code": "invalid_input", "error_message": "prefer must be a string or list."}), 400

    diet_val = None
    selected_types = None
    if diet_raw is not None or diet_types_raw is not None:
        try:
            diet_val, selected_types = _resolve_diet_selection(diet_raw, diet_types_raw)
        except ValueError as exc:
            return jsonify({"status": "failed", "error_code": "invalid_input", "error_message": str(exc)}), 400

    with _user_lock(user_id):
        sm, rl, _ = _get_ca2_runtime()
        st = sm.get(user_id)
        try:
            updated = st.set_preferences(diet=diet_val, avoid=avoid, prefer=prefer)
        except Exception as exc:
            return jsonify({"status": "failed", "error_code": "invalid_input", "error_message": str(exc)}), 400
        if selected_types is not None:
            _save_profile(user_id, {"diet_types": selected_types})
            updated["diet_types"] = selected_types
        _bump_data_version(user_id)
        rl.append(
            run_id=f"ui-{user_id}",
            user_id=user_id,
            agent="Nutrition State",
            task="Update food preferences",
            tool="set_preferences",
            status="success",
            output_summary=f"diet={updated['diet']}, diet_types={selected_types}, avoid={updated['avoid']}, prefer={updated['prefer']}",
        )
        return jsonify({"status": "success", "preferences": updated, "state": st.snapshot()})


@app.route("/api/nutritrack/recommendations", methods=["POST"])
def nt_recommendations():
    data, err = _parse_json_dict(allow_empty=True)
    if err is not None:
        return err
    assert data is not None
    user_id, _, auth_err = _authorize_request(data.get("user_id"))
    if auth_err is not None:
        return auth_err
    assert user_id is not None
    day_raw = data.get("date")
    if day_raw is not None and not isinstance(day_raw, str):
        return jsonify({"status": "failed", "error_code": "invalid_input", "error_message": "date must be a string."}), 400
    day = (day_raw or "").strip() or None
    diet = (data.get("diet") or "").strip().lower() if isinstance(data.get("diet"), str) else ""
    avoid = (data.get("avoid") or "").strip() if isinstance(data.get("avoid"), str) else ""
    filter_tag = (data.get("filter") or "all").strip().lower()

    if filter_tag == "vegetarian" and not diet:
        diet = "vegetarian"

    prompt = "What meal should I eat next to hit my remaining calorie and protein target?"
    extras: list[str] = []
    if diet and diet != "any":
        extras.append(f"Make it {diet}.")
    if avoid:
        extras.append(f"Avoid {avoid}.")
    if extras:
        prompt = f"{prompt} {' '.join(extras)}"

    with _user_lock(user_id):
        sm, rl, orch = _get_ca2_runtime()
        selected_mode = orch.ai_manager.selected_mode
        result = _run_orchestrator_turn(orch, prompt, user_id=user_id, day=day, selected_mode=selected_mode)
        st = sm.get(user_id)
        if result.get("status") == "success" and "meal_recommendation" in (result.get("activated_specialists") or []):
            pool_cands = st.pool.get("candidates") or {}
            ranked_ids = list(st.pool.get("ranked") or [])
            if ranked_ids and pool_cands:
                ranked_objs = [pool_cands[cid] for cid in ranked_ids if cid in pool_cands]
                if filter_tag == "high-protein":
                    ranked_objs.sort(
                        key=lambda c: (
                            -float((c.get("estimate") or {}).get("protein", 0) or 0),
                            -float((c.get("rank_info") or {}).get("score", 0) or 0),
                        )
                    )
                elif filter_tag == "quick-prep":
                    ranked_objs.sort(
                        key=lambda c: (
                            len(c.get("items") or []),
                            -float((c.get("rank_info") or {}).get("score", 0) or 0),
                        )
                    )

                ctx = new_context(sm, rl, user_id=user_id, task="validate_alternatives")
                for cand in ranked_objs[:4]:
                    cid = cand.get("id")
                    if cid:
                        recommendation_tools.validate_recommendation_fit(ctx, candidate_id=cid)
                current_recs = {r["id"]: r for r in st.snapshot(day=day)["recommendations"] if "id" in r}
                ordered_recs = [current_recs[c["id"]] for c in ranked_objs[:4] if c.get("id") in current_recs]
                if ordered_recs:
                    st.set_recommendations(ordered_recs, day=day)
                result["state"] = st.snapshot(day=day)
        if result.get("status") == "success":
            _bump_data_version(user_id)

    result["filter"] = filter_tag
    if result.get("status") == "failed":
        code = 503 if result.get("error_code") == "llm_unavailable" else (504 if result.get("error_code") == "turn_timeout" else 400)
        return jsonify(result), code
    return jsonify(result)


@app.route("/api/nutritrack/insights", methods=["GET"])
def nt_insights():
    user_id, _, auth_err = _authorize_request(request.args.get("user_id"))
    if auth_err is not None:
        return auth_err
    assert user_id is not None
    start_date = (request.args.get("start_date") or "").strip() or None
    end_date = (request.args.get("end_date") or "").strip() or None
    include_today = request.args.get("include_today", "true").lower() not in {"0", "false", "no"}
    days_raw = request.args.get("days", "7")
    try:
        days = int(days_raw)
    except (TypeError, ValueError):
        return jsonify({"status": "failed", "error_code": "invalid_input", "error_message": "days must be an integer."}), 400
    if days < 1 or days > 92:
        return jsonify({"status": "failed", "error_code": "invalid_input", "error_message": "days must be between 1 and 92."}), 400

    today_iso = date.today().isoformat()
    tz_name = time.tzname[0] if time.tzname else "UTC"

    t0 = time.perf_counter()
    with _user_lock(user_id):
        sm, _, _ = _get_ca2_runtime()
        st_sig = _file_sig(sm.path_for(user_id))
        prof_sig = _file_sig(_profile_path(user_id))
        data_ver = _get_data_version(user_id)
        cache_key = (user_id, start_date, end_date, days, include_today, today_iso, tz_name, st_sig, prof_sig, data_ver)
        with _cache_lock:
            cached = _insights_cache.get(cache_key)
            if cached is not None:
                _insights_cache.move_to_end(cache_key)
        if cached is not None:
            get_perf_monitor().record("get_insights", (time.perf_counter() - t0) * 1000.0)
            hit_payload = dict(cached)
            hit_payload["cache"] = {"cache_hit": True, "data_version": data_ver}
            return jsonify(hit_payload)

        st = sm.get(user_id)
        bundle = analysis_tools.compute_insights_bundle(
            st,
            end_date=end_date,
            days=days,
            start_date=start_date,
            include_today=include_today,
        )
        trends = bundle["trends"]
        if trends.get("status") == "failed":
            return jsonify(trends), 400
        period_end = (trends.get("period") or {}).get("end") or trends.get("end_date")
        profile = _load_profile(user_id, active_date=period_end)
        payload = {
            "status": "success",
            "user_id": user_id,
            "trends": trends,
            "macro_split": bundle["macro_split"],
            "daily_totals": bundle["daily_totals"],
            "profile": profile,
            "cache": {"cache_hit": False, "data_version": data_ver},
        }
        with _cache_lock:
            _insights_cache[cache_key] = payload
            _insights_cache.move_to_end(cache_key)
            while len(_insights_cache) > _MAX_CACHE_ENTRIES:
                _insights_cache.popitem(last=False)
    get_perf_monitor().record("get_insights", (time.perf_counter() - t0) * 1000.0)
    return jsonify(payload)


@app.route("/api/nutritrack/foods", methods=["GET"])
def nt_foods():
    user_id, _, auth_err = _authorize_request(request.args.get("user_id"))
    if auth_err is not None:
        return auth_err
    assert user_id is not None
    q = (request.args.get("q") or "").strip().lower()
    db = food_database.get_food_db(user_id=user_id)
    items: list[dict[str, Any]] = []
    for name in sorted(db.keys()):
        if q and q not in name.lower():
            continue
        entry = db[name]
        items.append(
            {
                "name": name,
                "calories": entry.get("calories", 0),
                "protein": entry.get("protein", 0),
                "carbs": entry.get("carbs", 0),
                "fat": entry.get("fat", 0),
                "fiber": entry.get("fiber", 0),
                "serving_size": entry.get("serving_size", "1 serving"),
                "unit": entry.get("unit", "piece"),
                "base_amount": entry.get("base_amount", 1),
                "custom": bool(entry.get("custom", False)),
            }
        )
    return jsonify({"status": "success", "user_id": user_id, "foods": items, "count": len(items)})


@app.route("/api/nutritrack/foods/preview", methods=["GET"])
def nt_food_preview():
    user_id, _, auth_err = _authorize_request(request.args.get("user_id"))
    if auth_err is not None:
        return auth_err
    assert user_id is not None
    food = (request.args.get("food") or "").strip()
    portion = (request.args.get("portion") or "").strip()
    qty_arg = (request.args.get("quantity") or "").strip()
    unit_arg = (request.args.get("unit") or "").strip()
    if not portion and qty_arg:
        portion = f"{qty_arg} {unit_arg}".strip()
    if not food:
        return jsonify({"status": "empty", "ok": False, "message": "Enter a food name to preview nutrition."})
    _, _, orch = _get_ca2_runtime()
    with ai_mode_scope(orch.ai_manager.selected_mode):
        coerced_portion, conversion_note = coerce_diary_portion(food, portion, user_id=user_id)
        combined = f"{coerced_portion} {food}".strip() if coerced_portion else food
        parsed = parse_item_text(combined)
        target_food = parsed.food_text or food
        res = resolve_item(
            target_food,
            parsed.value,
            parsed.unit,
            parsed.household_unit,
            register_external=False,
            user_id=user_id,
        )
    if not res.get("ok"):
        return jsonify(
            {
                "status": "unresolved",
                "ok": False,
                "food": food,
                "reason": res.get("reason"),
                "message": res.get("message"),
                "question": res.get("question"),
                "did_you_mean": res.get("did_you_mean", []),
            }
        )
    notes = list(res.get("notes", []))
    if conversion_note:
        notes.insert(0, conversion_note)
    return jsonify(
        {
            "status": "success",
            "ok": True,
            "food": res["food"],
            "quantity_str": res["quantity_str"],
            "serving_size": res["serving_size"],
            "calories": res["calories"],
            "protein": res["protein"],
            "carbs": res["carbs"],
            "fat": res["fat"],
            "fiber": res.get("fiber", 0),
            "provenance": res.get("provenance"),
            "external": bool(res.get("external", False)),
            "external_source": res.get("external_source"),
            "notes": notes,
        }
    )


@app.route("/api/nutritrack/meals", methods=["POST"])
@app.route("/api/nutritrack/meals/fast", methods=["POST"])
def nt_add_meal():
    data, err = _parse_json_dict()
    if err is not None:
        return err
    assert data is not None
    user_id, _, auth_err = _authorize_request(data.get("user_id"))
    if auth_err is not None:
        return auth_err
    assert user_id is not None
    day_raw = data.get("date")
    if day_raw is not None and not isinstance(day_raw, str):
        return jsonify({"status": "failed", "error_code": "invalid_input", "error_message": "date must be a string."}), 400
    day = (day_raw or "").strip() or None
    try:
        meal_type = _normalize_meal_type(data.get("meal_type"), strict=True)
    except ValueError as exc:
        return jsonify({"status": "failed", "error_code": "invalid_input", "error_message": str(exc)}), 400
    items_text = (data.get("items_text") or "").strip() if isinstance(data.get("items_text"), str) else ""
    food = (data.get("food") or "").strip() if isinstance(data.get("food"), str) else ""
    portion = (data.get("portion") or "").strip() if isinstance(data.get("portion"), str) else ""
    qty_raw = data.get("quantity")
    unit_raw = (data.get("unit") or "").strip() if isinstance(data.get("unit"), str) else ""
    if not portion and qty_raw is not None and str(qty_raw).strip() != "":
        portion = f"{qty_raw} {unit_raw}".strip()

    sm, rl, orch, _, op_store = _get_full_runtime()
    selected_mode = orch.ai_manager.selected_mode

    if not items_text:
        if not food:
            return jsonify({"status": "failed", "error_code": "invalid_input", "error_message": "Food name is required."}), 400
        coerced_portion, _ = coerce_diary_portion(food, portion, user_id=user_id)
        items_text = f"{coerced_portion} {food}".strip() if coerced_portion else food

    suffix = f" for {meal_type}" if meal_type else ""
    prompt = f"I ate {items_text}{suffix}"
    request_id = _extract_request_id(data)

    if request_id:
        p_hash = compute_payload_hash("add_meal", {"user_id": user_id, "items_text": items_text, "date": day, "meal_type": meal_type})
        op_action, op_rec = op_store.begin(user_id, request_id, endpoint="meals", payload_hash=p_hash)
        if op_action == "conflict":
            return jsonify({"status": "failed", "error_code": "idempotency_conflict", "error_message": "request_id used with different meal payload."}), 409
        if op_action == "in_progress":
            settled = op_store.wait_if_in_progress(user_id, request_id, timeout=3.0)
            if settled and settled.get("status") == "succeeded" and isinstance(settled.get("response"), dict):
                replay = copy.deepcopy(settled["response"])
                replay["duplicate"] = True
                return jsonify(replay), int(settled.get("status_code") or 200)
            return jsonify({"status": "pending", "error_code": "operation_in_progress", "error_message": "Meal request still processing."}), 409
        if op_action == "replay" and isinstance(op_rec.get("response"), dict):
            replay = copy.deepcopy(op_rec["response"])
            replay["duplicate"] = True
            return jsonify(replay), int(op_rec.get("status_code") or 200)

    completed_recorded = False
    try:
        with _user_lock(user_id), ai_mode_scope(selected_mode):
            sig_before = _state_analytics_sig(sm.get(user_id))
            if getattr(orch, "_explicit_client", None) is not None:
                result = orch.run_turn(prompt, user_id=user_id, date=day)
            else:
                result = fast_log_meal(
                    sm,
                    rl,
                    message=items_text,
                    user_id=user_id,
                    date=day,
                    meal_type=meal_type,
                    request_id=request_id,
                    ai_status=orch.ai_manager.get_status(),
                )
            sig_after = _state_analytics_sig(sm.get(user_id))
        status_code = 400 if result.get("status") == "failed" else 200
        if sig_after != sig_before or result.get("added"):
            _bump_data_version(user_id)
        if request_id:
            if status_code == 200 and result.get("status") != "failed":
                op_store.complete(user_id, request_id, status="succeeded", status_code=status_code, response=result)
            else:
                op_store.complete(user_id, request_id, status="failed", status_code=status_code, error_message=result.get("error_message"))
            completed_recorded = True
        return jsonify(result), status_code
    except Exception as exc:
        if request_id and not completed_recorded:
            op_store.complete(user_id, request_id, status="failed", status_code=500, error_message=sanitize_error_text(exc))
        raise


@app.route("/api/nutritrack/meals/<meal_id>", methods=["PUT"])
def nt_edit_meal(meal_id: str):
    if not _MEAL_ID_RE.match(meal_id):
        return jsonify({"status": "failed", "error_code": "invalid_input", "error_message": "Invalid meal_id."}), 400
    data, err = _parse_json_dict()
    if err is not None:
        return err
    assert data is not None
    user_id, _, auth_err = _authorize_request(data.get("user_id"))
    if auth_err is not None:
        return auth_err
    assert user_id is not None
    day_raw = data.get("date")
    if day_raw is not None and not isinstance(day_raw, str):
        return jsonify({"status": "failed", "error_code": "invalid_input", "error_message": "date must be a string."}), 400
    day = (day_raw or "").strip() or None
    food = (data.get("food") or "").strip() if isinstance(data.get("food"), str) else ""
    portion = (data.get("portion") or "").strip() if isinstance(data.get("portion"), str) else ""
    qty_raw = data.get("quantity")
    unit_raw = (data.get("unit") or "").strip() if isinstance(data.get("unit"), str) else ""
    if not portion and qty_raw is not None and str(qty_raw).strip() != "":
        portion = f"{qty_raw} {unit_raw}".strip()
    try:
        meal_type = _normalize_meal_type(data.get("meal_type"), strict=True)
    except ValueError as exc:
        return jsonify({"status": "failed", "error_code": "invalid_input", "error_message": str(exc)}), 400

    if not food:
        return jsonify({"status": "failed", "error_code": "invalid_input", "error_message": "Food name is required."}), 400

    sm, rl, orch = _get_ca2_runtime()
    selected_mode = orch.ai_manager.selected_mode
    with _user_lock(user_id), ai_mode_scope(selected_mode):
        coerced_portion, _ = coerce_diary_portion(food, portion, user_id=user_id)
        combined = f"{coerced_portion} {food}".strip() if coerced_portion else food
        parsed = parse_item_text(combined)
        check = resolve_item(parsed.food_text or food, parsed.value, parsed.unit, parsed.household_unit, user_id=user_id)
        if not check.get("ok"):
            return jsonify(
                {
                    "status": "failed",
                    "error_code": check.get("reason", "unresolved_food"),
                    "error_message": check.get("message") or f"Could not resolve '{combined}'.",
                    "question": check.get("question"),
                }
            ), 400

        st = sm.get(user_id)
        try:
            existing_meals = st.get_day_meals(day)
        except StateError as exc:
            return jsonify({"status": "failed", "error_code": "invalid_input", "error_message": str(exc)}), 400
        target_entry = next((m for m in existing_meals if m.get("meal_id") == meal_id), None)
        if target_entry is None:
            return jsonify({"status": "failed", "error_code": "not_found", "error_message": f"Meal '{meal_id}' not found."}), 404
        effective_meal_type = meal_type or target_entry.get("meal_type")
        removed = st.remove_meal(meal_id=meal_id)
        if not removed.get("removed"):
            return jsonify({"status": "failed", "error_code": "not_found", "error_message": f"Meal '{meal_id}' not found."}), 404
        target_day = (removed.get("dates") or [day])[0]
        before_ids = {m.get("meal_id") for m in st.get_day_meals(target_day)}
        if getattr(orch, "_explicit_client", None) is not None:
            suffix = f" for {effective_meal_type}" if effective_meal_type else ""
            result = orch.run_turn(f"I ate {combined}{suffix}", user_id=user_id, date=day)
        else:
            result = fast_log_meal(
                sm,
                rl,
                message=combined,
                user_id=user_id,
                date=day,
                meal_type=effective_meal_type,
                ai_status=orch.ai_manager.get_status(),
            )
        after_ids = {m.get("meal_id") for m in st.get_day_meals(target_day)}
        if result.get("status") != "success" or after_ids == before_ids:
            restore_entry = {
                k: target_entry[k]
                for k in ("food", "quantity_str", "calories", "protein", "carbs", "fat", "fiber", "external", "external_source")
                if k in target_entry and target_entry[k] is not None
            }
            st.add_meals(
                [restore_entry],
                meal_type=target_entry.get("meal_type"),
                day=target_day,
                source=target_entry.get("source") or "edit_rollback",
            )
            result["rolled_back"] = True
            result["state"] = st.snapshot(day=day)
            status_code = 503 if result.get("error_code") == "llm_unavailable" else 400
            return jsonify(result), status_code
        _bump_data_version(user_id)
        return jsonify(result)


@app.route("/api/nutritrack/meals/<meal_id>", methods=["DELETE"])
def nt_delete_meal(meal_id: str):
    if not _MEAL_ID_RE.match(meal_id):
        return jsonify({"status": "failed", "error_code": "invalid_input", "error_message": "Invalid meal_id."}), 400
    user_id, _, auth_err = _authorize_request(request.args.get("user_id"))
    if auth_err is not None:
        return auth_err
    assert user_id is not None
    day = (request.args.get("date") or "").strip() or None
    sm, rl, orch = _get_ca2_runtime()
    selected_mode = orch.ai_manager.selected_mode
    with _user_lock(user_id), ai_mode_scope(selected_mode):
        st = sm.get(user_id)
        if not st.has_meal(meal_id):
            return jsonify({"status": "failed", "error_code": "not_found", "error_message": f"Meal '{meal_id}' not found."}), 404
        if getattr(orch, "_explicit_client", None) is not None:
            result = orch.run_turn(f"remove meal {meal_id}", user_id=user_id, date=day)
        else:
            result = fast_remove_meal(sm, rl, meal_id=meal_id, user_id=user_id, date=day, ai_status=orch.ai_manager.get_status())
        if result.get("status") == "failed" or st.has_meal(meal_id):
            status_code = 503 if result.get("error_code") == "llm_unavailable" else 400
            return jsonify(result), status_code
        _bump_data_version(user_id)
        return jsonify(result)


@app.route("/api/nutritrack/profile", methods=["GET", "POST"])
def nt_profile():
    if request.method == "GET":
        user_id, _, auth_err = _authorize_request(request.args.get("user_id"))
        if auth_err is not None:
            return auth_err
        assert user_id is not None
        day = (request.args.get("date") or "").strip() or None
        with _user_lock(user_id):
            sm, _, _ = _get_ca2_runtime()
            st = sm.get(user_id)
            try:
                resolved_day = st.resolve_date(day)
            except StateError as exc:
                return jsonify({"status": "failed", "error_code": "invalid_input", "error_message": str(exc)}), 400
            return jsonify({"status": "success", "user_id": user_id, "profile": _load_profile(user_id, active_date=resolved_day)})

    data, err = _parse_json_dict()
    if err is not None:
        return err
    assert data is not None
    user_id, _, auth_err = _authorize_request(data.get("user_id"))
    if auth_err is not None:
        return auth_err
    assert user_id is not None
    day_raw = data.get("date")
    if day_raw is not None and not isinstance(day_raw, str):
        return jsonify({"status": "failed", "error_code": "invalid_input", "error_message": "date must be a string."}), 400
    day = (day_raw or "").strip() or None

    with _user_lock(user_id):
        sm, rl, _ = _get_ca2_runtime()
        st = sm.get(user_id)
        try:
            resolved_day = st.resolve_date(day)
        except StateError as exc:
            return jsonify({"status": "failed", "error_code": "invalid_input", "error_message": str(exc)}), 400

        # Stage 1: Validate ALL inputs (profile + targets + preferences) BEFORE persisting either file
        try:
            _validate_profile_updates(data)
            cal_val: Optional[float] = None
            prot_val: Optional[float] = None
            cal_raw = data.get("calories")
            prot_raw = data.get("protein")
            if cal_raw is not None and str(cal_raw).strip() != "":
                cal_val = _finite_float(str(cal_raw).replace(",", ""), "calories")
                if not (500.0 <= cal_val <= 15000.0):
                    raise StateError("Daily calorie target must be between 500 and 15000 kcal.")
            if prot_raw is not None and str(prot_raw).strip() != "":
                prot_val = _finite_float(str(prot_raw).replace(",", ""), "protein")
                if not (10.0 <= prot_val <= 600.0):
                    raise StateError("Daily protein target must be between 10 and 600 g.")

            update_prefs = "diet" in data or "diet_types" in data or "avoid" in data or "prefer" in data
            diet_val = None
            avoid_val = data.get("avoid")
            prefer_val = data.get("prefer")
            if update_prefs:
                if "diet" in data or "diet_types" in data:
                    diet_val, _ = _resolve_diet_selection(data.get("diet"), data.get("diet_types"), strict=True)
                if isinstance(avoid_val, str):
                    avoid_val = [x.strip() for x in avoid_val.split(",") if x.strip()]
                elif avoid_val is not None and not isinstance(avoid_val, list):
                    raise ValueError("avoid must be a string or list.")
                if isinstance(prefer_val, str):
                    prefer_val = [x.strip() for x in prefer_val.split(",") if x.strip()]
                elif prefer_val is not None and not isinstance(prefer_val, list):
                    raise ValueError("prefer must be a string or list.")
        except (TypeError, ValueError, StateError) as exc:
            return jsonify({"status": "failed", "error_code": "invalid_input", "error_message": str(exc)}), 400

        # Stage 2: Save profile with automatic rollback if subsequent state mutation fails
        prof_path = _profile_path(user_id)
        prof_lock_path = prof_path.with_name(f"{prof_path.name}.lock")
        prev_prof_bytes: Optional[bytes] = None
        with user_file_lock(prof_lock_path):
            if prof_path.exists():
                prev_prof_bytes = prof_path.read_bytes()

        try:
            updated_profile = _save_profile(user_id, data, active_date=resolved_day)
            goal_parts: list[str] = []
            if cal_val is not None:
                st.set_target("calories", cal_val, source="user")
                goal_parts.append(f"calories={cal_val:g} kcal")
            if prot_val is not None:
                st.set_target("protein", prot_val, source="user")
                goal_parts.append(f"protein={prot_val:g} g")
            if update_prefs:
                st.set_preferences(diet=diet_val, avoid=avoid_val, prefer=prefer_val)
        except (TypeError, ValueError, StateError) as exc:
            with _user_lock(user_id), user_file_lock(prof_lock_path):
                if prev_prof_bytes is not None:
                    atomic_write_json(prof_path, json.loads(prev_prof_bytes.decode("utf-8")))
                elif prof_path.exists():
                    prof_path.unlink(missing_ok=True)
            return jsonify({"status": "failed", "error_code": "invalid_input", "error_message": str(exc) or "Invalid profile settings."}), 400

        _bump_data_version(user_id)
        rl.append(
            run_id=f"ui-profile-{user_id}",
            user_id=user_id,
            agent="Goal and Budget Agent" if goal_parts else "Nutrition State",
            task="Update profile, targets & dietary preferences",
            tool="save_profile",
            status="success",
            output_summary=(
                f"Updated settings for {updated_profile['name']}: "
                f"{', '.join(goal_parts) if goal_parts else 'profile updated'}, "
                f"diet_types={updated_profile.get('diet_types')}, water={updated_profile['water_target_liters']}L"
            ),
        )
        return jsonify(
            {
                "status": "success",
                "user_id": user_id,
                "profile": updated_profile,
                "state": st.snapshot(day=day),
                "turn": None,
            }
        )


@app.route("/api/nutritrack/water", methods=["POST"])
def nt_water():
    data, err = _parse_json_dict(allow_empty=True)
    if err is not None:
        return err
    assert data is not None
    user_id, _, auth_err = _authorize_request(data.get("user_id"))
    if auth_err is not None:
        return auth_err
    assert user_id is not None
    day_raw = data.get("date")
    if day_raw is not None and not isinstance(day_raw, str):
        return jsonify({"status": "failed", "error_code": "invalid_input", "error_message": "date must be a string."}), 400
    day = (day_raw or "").strip() or None
    request_id = _extract_request_id(data)

    sm, rl, _, _, op_store = _get_full_runtime()
    if request_id:
        p_hash = compute_payload_hash("water", {"user_id": user_id, "date": day, "liters": data.get("liters"), "delta_liters": data.get("delta_liters")})
        op_action, op_rec = op_store.begin(user_id, request_id, endpoint="water", payload_hash=p_hash)
        if op_action == "conflict":
            return jsonify({"status": "failed", "error_code": "idempotency_conflict", "error_message": "request_id conflict on water update."}), 409
        if op_action == "in_progress":
            settled = op_store.wait_if_in_progress(user_id, request_id, timeout=3.0)
            if settled and settled.get("status") == "succeeded" and isinstance(settled.get("response"), dict):
                replay = copy.deepcopy(settled["response"])
                replay["duplicate"] = True
                return jsonify(replay), int(settled.get("status_code") or 200)
            return jsonify({"status": "pending", "error_code": "operation_in_progress", "error_message": "Water update is currently processing.", "request_id": request_id}), 409
        if op_action == "replay" and isinstance(op_rec.get("response"), dict):
            replay = copy.deepcopy(op_rec["response"])
            replay["duplicate"] = True
            return jsonify(replay), int(op_rec.get("status_code") or 200)

    with _user_lock(user_id):
        st = sm.get(user_id)
        try:
            resolved_day = st.resolve_date(day)
        except StateError as exc:
            if request_id:
                op_store.complete(user_id, request_id, status="failed", status_code=400, error_message=str(exc))
            return jsonify({"status": "failed", "error_code": "invalid_input", "error_message": str(exc)}), 400
        prof = _load_profile(user_id, active_date=resolved_day)
        current = float(prof.get("water_today_liters", 0.0) or 0.0)
        if "liters" in data and data["liters"] is not None:
            try:
                new_val = _finite_float(data["liters"], "liters")
            except ValueError as exc:
                if request_id:
                    op_store.complete(user_id, request_id, status="failed", status_code=400, error_message=str(exc))
                return jsonify({"status": "failed", "error_code": "invalid_input", "error_message": str(exc)}), 400
            if new_val < _WATER_MIN_LITERS or new_val > _WATER_MAX_LITERS:
                msg = f"liters must be between {_WATER_MIN_LITERS:g} and {_WATER_MAX_LITERS:g}."
                if request_id:
                    op_store.complete(user_id, request_id, status="failed", status_code=400, error_message=msg)
                return jsonify({"status": "failed", "error_code": "invalid_input", "error_message": msg}), 400
        else:
            try:
                delta = _finite_float(data.get("delta_liters", 0.25), "delta_liters")
            except ValueError as exc:
                if request_id:
                    op_store.complete(user_id, request_id, status="failed", status_code=400, error_message=str(exc))
                return jsonify({"status": "failed", "error_code": "invalid_input", "error_message": str(exc)}), 400
            if abs(delta) > _WATER_MAX_LITERS:
                msg = f"delta_liters magnitude cannot exceed {_WATER_MAX_LITERS:g} liters."
                if request_id:
                    op_store.complete(user_id, request_id, status="failed", status_code=400, error_message=msg)
                return jsonify({"status": "failed", "error_code": "invalid_input", "error_message": msg}), 400
            new_val = current + delta
            if new_val < _WATER_MIN_LITERS or new_val > _WATER_MAX_LITERS:
                msg = f"Resulting daily water intake ({new_val:g} L) must remain between {_WATER_MIN_LITERS:g} and {_WATER_MAX_LITERS:g} liters."
                if request_id:
                    op_store.complete(user_id, request_id, status="failed", status_code=400, error_message=msg)
                return jsonify({"status": "failed", "error_code": "invalid_input", "error_message": msg}), 400

        new_val = round(new_val, 2)
        logs = dict(prof.get("water_logs") or {})
        logs[resolved_day] = new_val
        updated = _save_profile(user_id, {"water_logs": logs}, active_date=resolved_day)
        _bump_data_version(user_id)
        rl.append(
            run_id=f"ui-water-{user_id}",
            user_id=user_id,
            agent="Nutrition State",
            task=f"Update water intake ({resolved_day})",
            tool="log_water",
            status="success",
            output_summary=f"Water intake on {resolved_day}: {new_val} L / {updated['water_target_liters']} L",
        )
        payload = {
            "status": "success",
            "date": resolved_day,
            "water_today_liters": new_val,
            "water_target_liters": updated["water_target_liters"],
            "profile": updated,
        }
        if request_id:
            op_store.complete(user_id, request_id, status="succeeded", status_code=200, response=payload)
        return jsonify(payload)


@app.route("/api/nutritrack/custom-food", methods=["POST"])
def nt_custom_food():
    data, err = _parse_json_dict()
    if err is not None:
        return err
    assert data is not None
    user_id, _, auth_err = _authorize_request(data.get("user_id"))
    if auth_err is not None:
        return auth_err
    assert user_id is not None
    name = (data.get("name") or "").strip() if isinstance(data.get("name"), str) else ""
    if not name:
        return jsonify({"status": "failed", "error_code": "invalid_input", "error_message": "Food name is required."}), 400
    try:
        calories = _finite_float(data["calories"], "calories")
        protein = _finite_float(data["protein"], "protein")
        carbs = _finite_float(data.get("carbs", 0) or 0, "carbs")
        fat = _finite_float(data.get("fat", 0) or 0, "fat")
        fiber = _finite_float(data.get("fiber", 0) or 0, "fiber")
    except (KeyError, ValueError):
        return jsonify(
            {"status": "failed", "error_code": "invalid_input", "error_message": "Calories and protein are required finite numeric values."}
        ), 400
    if any(x < 0 for x in (calories, protein, carbs, fat, fiber)):
        return jsonify({"status": "failed", "error_code": "invalid_input", "error_message": "Nutrition values cannot be negative."}), 400

    raw_serving = data.get("serving_size")
    if raw_serving is not None and not isinstance(raw_serving, str):
        return jsonify({"status": "failed", "error_code": "invalid_input", "error_message": "serving_size must be a string."}), 400
    serving_size = (raw_serving or "1 serving").strip() or "1 serving"

    raw_unit = data.get("unit")
    if raw_unit is not None and not isinstance(raw_unit, str):
        return jsonify({"status": "failed", "error_code": "invalid_input", "error_message": "unit must be a string."}), 400
    unit = (raw_unit or "piece").strip() or "piece"

    try:
        base_amount = _finite_float(data.get("base_amount", 1) or 1, "base_amount")
        replace_flag = _parse_bool_strict(data["replace"], "replace") if data.get("replace") is not None else False
        log_now = _parse_bool_strict(data["log_immediately"], "log_immediately") if data.get("log_immediately") is not None else False
        log_qty = base_amount
        if log_now and data.get("log_quantity") is not None:
            log_qty = _finite_float(data.get("log_quantity"), "log_quantity")
            if log_qty <= 0:
                raise ValueError("log_quantity must be positive.")
        log_date_raw = data.get("date")
        if log_date_raw is not None and not isinstance(log_date_raw, str):
            raise ValueError("date must be a string.")
        log_date = (log_date_raw or "").strip() or None
    except ValueError as exc:
        return jsonify({"status": "failed", "error_code": "invalid_input", "error_message": str(exc)}), 400
    if base_amount <= 0:
        return jsonify({"status": "failed", "error_code": "invalid_input", "error_message": "base_amount must be positive."}), 400

    with _user_lock(user_id):
        sm, rl, orch = _get_ca2_runtime()
        if log_date is not None:
            try:
                log_date = sm.get(user_id).resolve_date(log_date)
            except StateError as exc:
                return jsonify({"status": "failed", "error_code": "invalid_input", "error_message": str(exc)}), 400
        selected_mode = orch.ai_manager.selected_mode
        ctx = new_context(sm, rl, user_id=user_id, task=f"register_custom_food:{name}")
        with ai_mode_scope(selected_mode):
            res = intake_tools.register_custom_food(
                ctx,
                name=name,
                calories=calories,
                protein=protein,
                carbs=carbs,
                fat=fat,
                fiber=fiber,
                serving_size=serving_size,
                unit=unit,
                base_amount=base_amount,
                replace=replace_flag,
            )
            if res.get("status") == "success":
                _bump_data_version(user_id)
            if res.get("status") == "success" and log_now:
                if getattr(orch, "_explicit_client", None) is not None:
                    follow_up = orch.run_turn(f"I ate {log_qty:g} {unit} {name}", user_id=user_id, date=log_date)
                else:
                    follow_up = fast_log_meal(
                        sm,
                        rl,
                        message=f"{log_qty:g} {unit} {name}",
                        user_id=user_id,
                        date=log_date,
                        ai_status=orch.ai_manager.get_status(),
                    )
                res["logged_turn"] = follow_up
        if res.get("status") == "success":
            status_code = 200
        elif res.get("error_code") == "food_exists":
            status_code = 409
        else:
            status_code = 400
        return jsonify(res), status_code


@app.route("/api/nutritrack/custom-food/<food_name>", methods=["DELETE"])
def nt_delete_custom_food(food_name: str):
    user_id, _, auth_err = _authorize_request(request.args.get("user_id"))
    if auth_err is not None:
        return auth_err
    assert user_id is not None
    cleaned = (food_name or "").strip().lower()
    if not cleaned:
        return jsonify({"status": "failed", "error_code": "invalid_input", "error_message": "Food name is required."}), 400
    with _user_lock(user_id):
        deleted = food_database.delete_custom_food(cleaned, user_id=user_id)
        if not deleted:
            return (
                jsonify(
                    {
                        "status": "failed",
                        "error_code": "not_found",
                        "error_message": f"Custom food '{cleaned}' does not exist for user '{user_id}'.",
                    }
                ),
                404,
            )
        _bump_data_version(user_id)
        return jsonify({"status": "success", "user_id": user_id, "deleted": cleaned})


@app.route("/api/nutritrack/perf", methods=["GET"])
def nt_perf():
    _, _, auth_err = _authorize_request()
    if auth_err is not None:
        return auth_err
    return jsonify({"status": "success", "metrics": get_perf_monitor().snapshot()})


@app.route("/api/nutritrack/runs", methods=["GET"])
def nt_runs():
    raw_user = request.args.get("user_id")
    user_id, sess, auth_err = _authorize_request(raw_user)
    if auth_err is not None:
        return auth_err
    assert user_id is not None
    include_system = request.args.get("include_system", "0") == "1"
    limit_raw = request.args.get("limit", "40")
    try:
        limit = max(1, min(200, int(limit_raw)))
    except (TypeError, ValueError):
        limit = 40
    # When authenticated, strictly scope runs to the logged-in user; in local dev mode with no user_id filter, allow all
    filter_uid = str(sess["user_id"]) if sess is not None else ((raw_user or "").strip() or None)
    with _ca2_lock:
        _, rl, _ = _get_ca2_runtime()
        entries = rl.read(
            limit=limit,
            user_id=filter_uid,
            exclude_user_ids=None if include_system else ("system",),
        )
        return jsonify({"entries": entries, "count": len(entries)})


def main() -> None:
    daily_store.load_or_rollover(ca1_tools.get_memory())
    _get_ca2_runtime()
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    print(f"[NutriTrack] Serving on http://{host}:{port} | LLM config: {detect_llm_config()['mode']}")
    sys.stdout.flush()
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
