"""
run_log.py
----------
Append-only execution log: logs/agent_runs.jsonl  (one JSON object per line).

Guarantees
  * APPEND-ONLY: the file is only ever opened in append mode; existing lines are
    never rewritten, truncated or reordered.
  * NO SECRETS: API keys, tokens, passwords and "Bearer ..." strings are masked
    before anything is written, and every free-text field is length-limited.
  * A broken log never breaks a tool: append() returns False and reports the
    problem through `logging`; it does not raise.

Entry fields: timestamp, run_id, user_id, agent, task, tool, input_summary,
output_summary, status ("success" | "failed"), error (null | {code, message}),
duration_ms.
"""
from collections import OrderedDict
import json
import logging
import os
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path

from ..paths import logs_dir
from .io import user_file_lock

log = logging.getLogger(__name__)

STATUSES = ("success", "failed")
MAX_FIELD_CHARS = 400
_SECRET_KEY_PARTS = ("api_key", "apikey", "token", "password", "passwd", "secret",
                     "authorization", "bearer", "credential")
SECRET_VALUE_PATTERNS = (
    re.compile(r"AIza[A-Za-z0-9_\-]{20,}"),
    re.compile(r"gsk_[A-Za-z0-9_\-]{16,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{10,}"),
    re.compile(r"(?i)(?:[?&](?:api[_-]?key|key|token|access_token)=)[^\s&'\"<>#]+"),
    re.compile(r"(?i)(?:api[_-]?key|authorization|x-goog-api-key|x-api-key)['\"]?\s*[:=]\s*['\"]?[^\s,'\";}\)]+['\"]?"),
)
_SECRET_VALUE_PATTERNS = SECRET_VALUE_PATTERNS
MASK = "***"


def new_run_id(now=None):
    """Unique without coordination: timestamp + 10 random hex characters.
    (4 characters were NOT enough: a test of 500 ids in one second collided, as the
    birthday paradox predicts. 10 characters make a collision astronomically unlikely.)"""
    now = now or datetime.now()
    return f"run-{now:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:10]}"


def redact(value):
    """Return a copy of `value` with secrets masked (by key name and by shape)."""
    if isinstance(value, dict):
        return {k: (MASK if any(p in str(k).lower() for p in _SECRET_KEY_PARTS) else redact(v))
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, str):
        for pattern in SECRET_VALUE_PATTERNS:
            value = pattern.sub(MASK, value)
    return value


def sanitize_error_text(value, max_len=320):
    """Mask API keys, URL key params, tokens, and auth headers in error messages."""
    if value is None:
        return ""
    text = str(value)
    for pattern in SECRET_VALUE_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    text = " ".join(text.split())
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def summarize(value, limit=MAX_FIELD_CHARS):
    """Short, secret-free text for a log field."""
    if value is None:
        return ""
    safe = redact(value)
    text = safe if isinstance(safe, str) else json.dumps(safe, ensure_ascii=False, default=str, separators=(",", ":"))
    for pattern in _SECRET_VALUE_PATTERNS:          # re-check the serialised text too
        text = pattern.sub(MASK, text)
    if len(text) > limit:
        text = text[:limit] + f"...(+{len(text) - limit} chars)"
    return text


MAX_LOG_BYTES = 5 * 1024 * 1024  # 5 MiB size-based rotation threshold
MAX_ROTATED_FILES = 3
_MAX_CACHED_RUNS = 64


class RunLog:
    def __init__(self, path=None, clock=None, max_bytes=None):
        self.path = Path(path) if path else logs_dir() / "agent_runs.jsonl"
        self._lock_path = self.path.with_name(f"{self.path.name}.lock")
        self._clock = clock or datetime.now
        self._lock = threading.Lock()
        self._max_bytes = max(256, int(max_bytes)) if max_bytes is not None else int(
            os.environ.get("NUTRITRACK_MAX_LOG_BYTES", str(MAX_LOG_BYTES))
        )
        self.failures = 0
        self.skipped_lines = 0          # malformed lines seen by the last read()
        self._cache_sig = None
        self._run_cache: OrderedDict[str, list[dict]] = OrderedDict()
        self._tail_cache: dict[tuple, list[dict]] = {}

    def _stat_sig(self):
        try:
            st = self.path.stat()
            return (st.st_mtime_ns, st.st_size)
        except OSError:
            return None

    def _invalidate_caches_locked(self) -> None:
        self._cache_sig = None
        self._run_cache.clear()
        self._tail_cache.clear()

    def _rotate_if_needed_locked(self) -> bool:
        try:
            if self.path.exists() and self.path.stat().st_size >= self._max_bytes:
                for idx in range(MAX_ROTATED_FILES, 1, -1):
                    src = self.path.with_name(f"{self.path.name}.{idx - 1}")
                    dst = self.path.with_name(f"{self.path.name}.{idx}")
                    if src.exists():
                        os.replace(src, dst)
                rotated = self.path.with_name(f"{self.path.name}.1")
                os.replace(self.path, rotated)
                return True
        except OSError:
            pass
        return False

    def append(self, *, run_id, agent, task, tool, status, input_summary="", output_summary="",
               error=None, user_id=None, duration_ms=None, execution_mode=None):
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {status!r}")
        entry = {
            "timestamp": self._clock().isoformat(timespec="milliseconds"),
            "run_id": str(run_id),
            "user_id": user_id,
            "agent": summarize(agent, 80),
            "task": summarize(task, 200),
            "tool": summarize(tool, 80),
            "input_summary": summarize(input_summary),
            "output_summary": summarize(output_summary),
            "status": status,
            "error": None if error is None else {
                "code": summarize((error or {}).get("code", "error"), 60),
                "message": summarize((error or {}).get("message", ""), 300)},
            "duration_ms": None if duration_ms is None else round(float(duration_ms), 1),
        }
        if execution_mode is not None:
            entry["execution_mode"] = str(execution_mode)
        line = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
        do_fsync = os.environ.get("NUTRITRACK_LOG_FSYNC", "").strip() == "1"
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with user_file_lock(self._lock_path):
                    pre_sig = self._stat_sig()
                    rotated = self._rotate_if_needed_locked()
                    with open(self.path, "a", encoding="utf-8") as fh:   # "a": can never overwrite
                        fh.write(line + "\n")
                        fh.flush()
                        if do_fsync:
                            os.fsync(fh.fileno())
                    post_sig = self._stat_sig()
                    if not rotated and pre_sig is not None and pre_sig == self._cache_sig:
                        rid = entry["run_id"]
                        if rid in self._run_cache:
                            self._run_cache[rid].append(dict(entry))
                            self._run_cache.move_to_end(rid)
                        self._tail_cache.clear()
                        self._cache_sig = post_sig
                    else:
                        self._invalidate_caches_locked()
            return True
        except Exception as exc:
            self.failures += 1
            log.error("Could not append to the run log %s: %s", self.path, exc)
            return False

    def _read_tail_entries(self, limit: int, *, user_id=None, excluded=None, run_id=None) -> tuple[list[dict], int]:
        """Read last `limit` matching entries newest-first across active + rotated logs, then reverse."""
        collected: list[dict] = []
        skipped = 0
        chunk_size = 8192
        candidates = [self.path] + [
            self.path.with_name(f"{self.path.name}.{i}") for i in range(1, MAX_ROTATED_FILES + 1)
        ]
        for target in candidates:
            if len(collected) >= limit:
                break
            if not target.exists():
                continue
            with open(target, "rb") as fh:
                fh.seek(0, os.SEEK_END)
                pos = fh.tell()
                carry = b""
                while pos > 0 and len(collected) < limit:
                    read_len = min(chunk_size, pos)
                    pos -= read_len
                    fh.seek(pos)
                    buf = fh.read(read_len) + carry
                    parts = buf.split(b"\n")
                    carry = parts[0] if pos > 0 else b""
                    lines = parts[1:] if pos > 0 else parts
                    for raw_bytes in reversed(lines):
                        raw = raw_bytes.decode("utf-8", errors="replace").strip()
                        if not raw:
                            continue
                        try:
                            entry = json.loads(raw)
                            if not isinstance(entry, dict):
                                raise ValueError("not an object")
                        except Exception:
                            skipped += 1
                            continue
                        if run_id is not None and entry.get("run_id") != run_id:
                            continue
                        if user_id is not None and entry.get("user_id") != user_id:
                            continue
                        if excluded and entry.get("user_id") in excluded:
                            continue
                        collected.append(entry)
                        if len(collected) >= limit:
                            break
        collected.reverse()
        return collected, skipped

    def read(self, limit=None, run_id=None, user_id=None, exclude_user_ids=None):
        """Entries in file order (oldest first). Uses fast EOF tail scan when bounded by `limit`."""
        if limit is not None and int(limit) == 0:
            with self._lock:
                self.skipped_lines = 0
            return []
        excluded = frozenset(exclude_user_ids) if exclude_user_ids else frozenset()
        with self._lock:
            if not self.path.exists():
                self.skipped_lines = 0
                return []
            with user_file_lock(self._lock_path):
                sig = self._stat_sig()
                if sig is None:
                    self.skipped_lines = 0
                    return []
                if sig != self._cache_sig:
                    self._run_cache.clear()
                    self._tail_cache.clear()
                    self._cache_sig = sig

                if run_id is not None and user_id is None and not excluded and limit is None:
                    cached = self._run_cache.get(run_id)
                    if cached is not None:
                        self.skipped_lines = 0
                        return [dict(e) for e in cached]

                if limit is not None and int(limit) > 0 and run_id is None:
                    tkey = (int(limit), user_id, excluded)
                    cached_tail = self._tail_cache.get(tkey)
                    if cached_tail is not None:
                        self.skipped_lines = 0
                        return [dict(e) for e in cached_tail]
                    collected, skipped = self._read_tail_entries(int(limit), user_id=user_id, excluded=excluded)
                    self.skipped_lines = skipped
                    if len(self._tail_cache) > 32:
                        self._tail_cache.clear()
                    self._tail_cache[tkey] = [dict(e) for e in collected]
                    return collected

                entries = []
                skipped = 0
                with open(self.path, "r", encoding="utf-8") as fh:
                    for raw in fh:
                        raw = raw.strip()
                        if not raw:
                            continue
                        try:
                            entry = json.loads(raw)
                            if not isinstance(entry, dict):
                                raise ValueError("not an object")
                        except Exception:
                            skipped += 1
                            continue
                        if run_id is not None and entry.get("run_id") != run_id:
                            continue
                        if user_id is not None and entry.get("user_id") != user_id:
                            continue
                        if excluded and entry.get("user_id") in excluded:
                            continue
                        entries.append(entry)
                self.skipped_lines = skipped
                if run_id is not None and user_id is None and not excluded and limit is None:
                    self._run_cache[run_id] = [dict(e) for e in entries]
                    if len(self._run_cache) > _MAX_CACHED_RUNS:
                        self._run_cache.popitem(last=False)
                return entries[-int(limit):] if limit else entries
