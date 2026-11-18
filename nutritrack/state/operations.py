"""Durable per-user operation lifecycle tracking for idempotent retries and timeout recovery."""

from __future__ import annotations

import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Optional
import uuid

from ..paths import data_dir
from .io import atomic_write_json, user_file_lock
from .nutrition_state import USER_ID_RE, StateError, StateUpdateError

OPERATION_STATUSES = ("pending", "processing", "succeeded", "failed")
MAX_OPERATIONS_PER_USER = 200

_ACTIVE_LEASES_LOCK = threading.Lock()
# Maps (resolved_ops_dir, user_id, request_id) -> lease_token for live in-process workers
_PROCESS_ACTIVE_LEASES: dict[tuple[str, str, str], str] = {}


class OperationStoreCorruptError(StateUpdateError):
    """Raised when a persisted operation store file is corrupted or unreadable."""


def _is_pid_alive(pid: Any) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    if pid == os.getpid():
        return True
    try:
        import psutil

        return bool(psutil.pid_exists(pid))
    except Exception:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def compute_payload_hash(endpoint: str, payload: dict[str, Any]) -> str:
    """Compute a deterministic SHA-256 fingerprint for an operation request (excluding volatile request_id)."""
    clean = {k: v for k, v in (payload or {}).items() if k != "request_id"}
    raw = json.dumps({"endpoint": endpoint, "payload": clean}, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class DurableOperationStore:
    """Cross-process durable idempotency & operation status store (`data/operations/<user_id>.json`)."""

    def __init__(self, base_dir: Optional[Path | str] = None, stale_seconds: float = 90.0) -> None:
        self._base = Path(base_dir) if base_dir else None
        self._stale_seconds = max(5.0, float(stale_seconds))
        self._events_lock = threading.Lock()
        self._events: dict[tuple[str, str], threading.Event] = {}

    def _ops_dir(self) -> Path:
        return (self._base or data_dir()) / "operations"

    def _path_for(self, user_id: str) -> Path:
        if not isinstance(user_id, str) or not USER_ID_RE.match(user_id):
            raise StateError("user_id must be 1-64 characters: letters, digits, '_' or '-'.")
        return self._ops_dir() / f"{user_id}.json"

    def _read_ops_locked(self, path: Path) -> dict[str, dict[str, Any]]:
        if not path.exists():
            return {}
        try:
            raw_text = path.read_text(encoding="utf-8")
            doc = json.loads(raw_text)
        except Exception as exc:
            raise OperationStoreCorruptError(
                f"Operation store '{path.name}' is corrupted ({exc}). Refusing to overwrite duplicate-protection history."
            ) from exc
        if not isinstance(doc, dict) or not isinstance(doc.get("operations"), dict):
            raise OperationStoreCorruptError(
                f"Operation store '{path.name}' has invalid structure. Refusing to overwrite duplicate-protection history."
            )
        return doc["operations"]

    def _write_ops_locked(self, path: Path, ops: dict[str, dict[str, Any]]) -> None:
        if len(ops) > MAX_OPERATIONS_PER_USER:
            evictable = [
                k for k, v in ops.items()
                if not isinstance(v, dict) or v.get("status") not in ("pending", "processing")
            ]
            for oldest_key in evictable:
                if len(ops) <= MAX_OPERATIONS_PER_USER:
                    break
                ops.pop(oldest_key, None)
        atomic_write_json(path, {"operations": ops})

    def _lease_key(self, user_id: str, request_id: str) -> tuple[str, str, str]:
        return (str(self._ops_dir().resolve()), user_id, request_id)

    def _set_active_lease(self, user_id: str, request_id: str, lease_token: str) -> None:
        with _ACTIVE_LEASES_LOCK:
            _PROCESS_ACTIVE_LEASES[self._lease_key(user_id, request_id)] = lease_token

    def _clear_active_lease(self, user_id: str, request_id: str, lease_token: Optional[str] = None) -> None:
        key = self._lease_key(user_id, request_id)
        with _ACTIVE_LEASES_LOCK:
            if lease_token is None or _PROCESS_ACTIVE_LEASES.get(key) == lease_token:
                _PROCESS_ACTIVE_LEASES.pop(key, None)

    def _has_active_lease(self, user_id: str, request_id: str, lease_token: Optional[str] = None) -> bool:
        key = self._lease_key(user_id, request_id)
        with _ACTIVE_LEASES_LOCK:
            active_tok = _PROCESS_ACTIVE_LEASES.get(key)
            if active_tok is None:
                return False
            return lease_token is None or active_tok == lease_token

    def _register_event(self, user_id: str, request_id: str) -> None:
        with self._events_lock:
            self._events[(user_id, request_id)] = threading.Event()

    def _signal_event(self, user_id: str, request_id: str) -> None:
        with self._events_lock:
            ev = self._events.pop((user_id, request_id), None)
        if ev is not None:
            ev.set()

    def begin(
        self,
        user_id: str,
        request_id: str,
        *,
        endpoint: str,
        payload_hash: str,
    ) -> tuple[str, dict[str, Any]]:
        """Begin or inspect a durable operation.

        Returns `(action, record)` where `action` is one of:
          - `"execute"`: caller should execute the operation and then call `complete(...)`
          - `"replay"`: already succeeded; return stored response without re-executing
          - `"in_progress"`: currently processing concurrently (or lease still held by live worker)
          - `"conflict"`: request_id already used for a different payload
        """
        path = self._path_for(user_id)
        lock_path = path.with_name(f"{path.name}.lock")
        now_ts = time.time()
        now_iso = datetime.now().isoformat(timespec="seconds")

        with user_file_lock(lock_path):
            ops = self._read_ops_locked(path)
            existing = ops.get(request_id)
            if isinstance(existing, dict):
                if existing.get("payload_hash") != payload_hash:
                    return "conflict", copy.deepcopy(existing)
                status = existing.get("status")
                if status == "succeeded":
                    return "replay", copy.deepcopy(existing)
                if status in ("pending", "processing"):
                    age = now_ts - float(existing.get("updated_ts") or 0.0)
                    lease_exp = float(existing.get("lease_expires_ts") or (float(existing.get("updated_ts") or 0.0) + self._stale_seconds))
                    owner_pid = existing.get("owner_pid")
                    stored_token = existing.get("lease_token")
                    worker_alive = self._has_active_lease(user_id, request_id, stored_token) or (
                        isinstance(owner_pid, int) and owner_pid != os.getpid() and _is_pid_alive(owner_pid)
                    )
                    if age < self._stale_seconds or now_ts < lease_exp or worker_alive:
                        return "in_progress", copy.deepcopy(existing)
                # Failed or confirmed dead-worker stale in-progress: transition cleanly to processing for safe retry
                lease_token = uuid.uuid4().hex
                existing["status"] = "processing"
                existing["error_message"] = None
                existing["updated_at"] = now_iso
                existing["updated_ts"] = now_ts
                existing["lease_expires_ts"] = now_ts + self._stale_seconds
                existing["owner_pid"] = os.getpid()
                existing["lease_token"] = lease_token
                existing["attempts"] = int(existing.get("attempts") or 1) + 1
                self._write_ops_locked(path, ops)
                self._set_active_lease(user_id, request_id, lease_token)
                self._register_event(user_id, request_id)
                return "execute", copy.deepcopy(existing)

            lease_token = uuid.uuid4().hex
            rec: dict[str, Any] = {
                "request_id": request_id,
                "user_id": user_id,
                "endpoint": endpoint,
                "payload_hash": payload_hash,
                "status": "processing",
                "status_code": None,
                "response": None,
                "error_message": None,
                "attempts": 1,
                "owner_pid": os.getpid(),
                "lease_token": lease_token,
                "lease_expires_ts": now_ts + self._stale_seconds,
                "created_at": now_iso,
                "updated_at": now_iso,
                "updated_ts": now_ts,
            }
            ops[request_id] = rec
            self._write_ops_locked(path, ops)
            self._set_active_lease(user_id, request_id, lease_token)
            self._register_event(user_id, request_id)
            return "execute", copy.deepcopy(rec)

    def complete(
        self,
        user_id: str,
        request_id: str,
        *,
        status: str,
        status_code: int,
        response: Optional[dict[str, Any]] = None,
        error_message: Optional[str] = None,
        lease_token: Optional[str] = None,
    ) -> dict[str, Any]:
        """Mark an operation as `succeeded` or `failed` and persist its outcome."""
        if status not in ("succeeded", "failed"):
            raise ValueError(f"Operation completion status must be 'succeeded' or 'failed', got {status!r}")
        path = self._path_for(user_id)
        lock_path = path.with_name(f"{path.name}.lock")
        now_ts = time.time()
        now_iso = datetime.now().isoformat(timespec="seconds")

        try:
            with user_file_lock(lock_path):
                ops = self._read_ops_locked(path)
                rec = ops.get(request_id)
                if not isinstance(rec, dict):
                    rec = {
                        "request_id": request_id,
                        "user_id": user_id,
                        "endpoint": "",
                        "payload_hash": "",
                        "attempts": 1,
                        "created_at": now_iso,
                    }
                    ops[request_id] = rec
                rec["status"] = status
                rec["status_code"] = int(status_code)
                rec["response"] = copy.deepcopy(response) if status == "succeeded" else None
                rec["error_message"] = error_message
                rec["updated_at"] = now_iso
                rec["updated_ts"] = now_ts
                self._write_ops_locked(path, ops)
                return copy.deepcopy(rec)
        finally:
            self._clear_active_lease(user_id, request_id, lease_token)
            self._signal_event(user_id, request_id)

    def wait_if_in_progress(self, user_id: str, request_id: str, timeout: float = 5.0) -> Optional[dict[str, Any]]:
        """Wait briefly if this process is currently executing `(user_id, request_id)` and return updated state."""
        with self._events_lock:
            ev = self._events.get((user_id, request_id))
        if ev is not None:
            ev.wait(timeout=max(0.05, float(timeout)))
        return self.get(user_id, request_id)

    def get(self, user_id: str, request_id: str) -> Optional[dict[str, Any]]:
        """Fetch persisted status of `(user_id, request_id)` or None."""
        path = self._path_for(user_id)
        lock_path = path.with_name(f"{path.name}.lock")
        with user_file_lock(lock_path):
            ops = self._read_ops_locked(path)
            rec = ops.get(request_id)
            return copy.deepcopy(rec) if isinstance(rec, dict) else None
