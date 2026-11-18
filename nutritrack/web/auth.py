"""Server-side authentication, session management, CSRF protection, and user authorization."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import secrets
import time
from typing import Any, Optional

from nutritrack.paths import data_dir
from nutritrack.state.io import atomic_write_json, user_file_lock

SESSION_COOKIE_NAME = "nutritrack_session"
SESSION_TTL_SECONDS = 24 * 60 * 60
_USER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_PBKDF2_ITERS = 200_000
_LOOPBACK_ADDRS = {"127.0.0.1", "::1", "localhost"}
_LOGIN_WINDOW_SECONDS = 60.0
_LOGIN_MAX_FAILURES_PER_USER = 5
_LOGIN_MAX_FAILURES_PER_IP = 25


class AuthError(Exception):
    """Structured authentication or authorization failure mapped to HTTP 400/401/403/409/429/500."""

    def __init__(
        self,
        status_code: int,
        error_code: str,
        message: str,
        *,
        retry_after: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = int(status_code)
        self.error_code = error_code
        self.code = error_code
        self.message = message
        self.retry_after = retry_after


def _hash_password(password: str, salt_hex: Optional[str] = None) -> tuple[str, str]:
    salt = bytes.fromhex(salt_hex) if salt_hex else os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERS)
    return salt.hex(), digest.hex()


def _verify_password(password: str, salt_hex: str, expected_hex: str) -> bool:
    try:
        _, got_hex = _hash_password(password, salt_hex)
        return hmac.compare_digest(got_hex, expected_hex)
    except Exception:
        return False


class AuthManager:
    """File-backed, cross-process safe authentication and session store scoped to `data_dir`."""

    def __init__(self, base_dir: Optional[Path | str] = None, session_ttl: float = SESSION_TTL_SECONDS) -> None:
        self._base = Path(base_dir) if base_dir else None
        self._ttl = max(60.0, float(session_ttl))

    def _auth_dir(self) -> Path:
        return (self._base or data_dir()) / "auth"

    def _users_path(self) -> Path:
        return self._auth_dir() / "users.json"

    def _sessions_path(self) -> Path:
        return self._auth_dir() / "sessions.json"

    def _secret_path(self) -> Path:
        return self._auth_dir() / ".session_secret"

    def _rate_limits_path(self) -> Path:
        return self._auth_dir() / "login_attempts.json"

    def _get_signing_key(self) -> bytes:
        env_secret = os.environ.get("NUTRITRACK_SESSION_SECRET", "").strip()
        if env_secret:
            return hashlib.sha256(env_secret.encode("utf-8")).digest()
        sec_path = self._secret_path()
        lock_path = sec_path.with_name(f"{sec_path.name}.lock")
        try:
            with user_file_lock(lock_path):
                if sec_path.exists():
                    raw = sec_path.read_text(encoding="utf-8").strip()
                    key_bytes = bytes.fromhex(raw[:64]) if len(raw) >= 64 else b""
                    if len(key_bytes) != 32:
                        raise AuthError(500, "auth_storage_error", "Session signing key file is corrupted.")
                    return key_bytes
                key = secrets.token_bytes(32)
                sec_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = sec_path.with_name(f"{sec_path.name}.tmp.{os.getpid()}")
                tmp_path.write_text(key.hex(), encoding="utf-8")
                if os.name != "nt":
                    try:
                        os.chmod(tmp_path, 0o600)
                    except OSError:
                        pass
                os.replace(tmp_path, sec_path)
                if os.name != "nt":
                    try:
                        os.chmod(sec_path, 0o600)
                    except OSError:
                        pass
                return key
        except AuthError:
            raise
        except Exception as exc:
            raise AuthError(500, "auth_storage_error", "Session signing key storage is unavailable.") from exc

    def _sign_sid(self, sid: str) -> str:
        sig = hmac.new(self._get_signing_key(), sid.encode("utf-8"), hashlib.sha256).hexdigest()
        return f"{sid}.{sig}"

    def _verify_token_signature(self, token: str) -> Optional[str]:
        if not isinstance(token, str) or "." not in token:
            return None
        sid, sig = token.split(".", 1)
        if not sid or not sig:
            return None
        expected = hmac.new(self._get_signing_key(), sid.encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        return sid

    def _read_json_dict(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def has_users(self) -> bool:
        path = self._users_path()
        if not path.exists():
            return False
        lock_path = path.with_name(f"{path.name}.lock")
        with user_file_lock(lock_path):
            return bool(self._read_json_dict(path))

    def register(self, user_id: str, password: str) -> dict[str, Any]:
        if not isinstance(user_id, str) or not _USER_ID_RE.match(user_id.strip()):
            raise AuthError(400, "invalid_user_id", "user_id must match [A-Za-z0-9_-]{1,64}.")
        clean_id = user_id.strip()
        if not isinstance(password, str) or len(password) < 6 or len(password) > 200:
            raise AuthError(400, "weak_password", "Password must be 6-200 characters.")

        path = self._users_path()
        lock_path = path.with_name(f"{path.name}.lock")
        salt_hex, pw_hash = _hash_password(password)
        with user_file_lock(lock_path):
            users = self._read_json_dict(path)
            if clean_id in users:
                raise AuthError(409, "user_exists", f"User '{clean_id}' is already registered.")
            users[clean_id] = {
                "user_id": clean_id,
                "salt": salt_hex,
                "pw_hash": pw_hash,
                "created_at": time.time(),
            }
            atomic_write_json(path, users)
        return self.create_session(clean_id)

    def _check_login_rate_limit(self, client_key: str, user_key: str, now: float) -> None:
        path = self._rate_limits_path()
        lock_path = path.with_name(f"{path.name}.lock")
        with user_file_lock(lock_path):
            doc = self._read_json_dict(path)
            dirty = False
            cutoff = now - _LOGIN_WINDOW_SECONDS
            for bucket in (f"u:{client_key}:{user_key}", f"ip:{client_key}"):
                raw_ts = doc.get(bucket)
                if not isinstance(raw_ts, list):
                    continue
                valid_ts = [float(t) for t in raw_ts if isinstance(t, (int, float)) and float(t) > cutoff]
                if len(valid_ts) != len(raw_ts):
                    doc[bucket] = valid_ts
                    dirty = True
                limit = _LOGIN_MAX_FAILURES_PER_USER if bucket.startswith("u:") else _LOGIN_MAX_FAILURES_PER_IP
                if len(valid_ts) >= limit:
                    oldest = min(valid_ts)
                    retry_after = max(1, int(math.ceil((oldest + _LOGIN_WINDOW_SECONDS) - now)))
                    if dirty:
                        atomic_write_json(path, doc)
                    raise AuthError(
                        429,
                        "rate_limited",
                        f"Too many failed login attempts. Try again in {retry_after}s.",
                        retry_after=retry_after,
                    )
            if dirty:
                atomic_write_json(path, doc)

    def _record_login_failure(self, client_key: str, user_key: str, now: float) -> None:
        path = self._rate_limits_path()
        lock_path = path.with_name(f"{path.name}.lock")
        cutoff = now - _LOGIN_WINDOW_SECONDS
        with user_file_lock(lock_path):
            doc = self._read_json_dict(path)
            cleaned_doc: dict[str, list[float]] = {}
            for k, v in doc.items():
                if isinstance(v, list):
                    valid = [float(t) for t in v if isinstance(t, (int, float)) and float(t) > cutoff]
                    if valid:
                        cleaned_doc[k] = valid[-50:]
            for bucket in (f"u:{client_key}:{user_key}", f"ip:{client_key}"):
                lst = cleaned_doc.get(bucket, [])
                lst.append(now)
                cleaned_doc[bucket] = lst[-50:]
            atomic_write_json(path, cleaned_doc)

    def _clear_login_failures(self, client_key: str, user_key: str) -> None:
        path = self._rate_limits_path()
        if not path.exists():
            return
        lock_path = path.with_name(f"{path.name}.lock")
        with user_file_lock(lock_path):
            doc = self._read_json_dict(path)
            removed = False
            if f"u:{client_key}:{user_key}" in doc:
                doc.pop(f"u:{client_key}:{user_key}", None)
                removed = True
            if removed:
                atomic_write_json(path, doc)

    def authenticate(self, user_id: str, password: str, *, client_ip: str = "") -> dict[str, Any]:
        if not isinstance(user_id, str) or not _USER_ID_RE.match(user_id.strip()):
            raise AuthError(400, "invalid_user_id", "user_id must match [A-Za-z0-9_-]{1,64}.")
        if not isinstance(password, str) or not password:
            raise AuthError(400, "invalid_credentials", "Password is required.")
        clean_id = user_id.strip()
        client_key = (client_ip or "local").strip() or "local"
        now = time.time()
        self._check_login_rate_limit(client_key, clean_id, now)
        path = self._users_path()
        lock_path = path.with_name(f"{path.name}.lock")
        with user_file_lock(lock_path):
            users = self._read_json_dict(path)
            rec = users.get(clean_id)
        if not isinstance(rec, dict) or not _verify_password(
            password, str(rec.get("salt", "")), str(rec.get("pw_hash", ""))
        ):
            self._record_login_failure(client_key, clean_id, now)
            raise AuthError(401, "invalid_credentials", "Invalid user_id or password.")
        self._clear_login_failures(client_key, clean_id)
        return self.create_session(clean_id)

    def create_session(self, user_id: str, *, ttl_seconds: Optional[float] = None) -> dict[str, Any]:
        clean_id = user_id.strip()
        if not _USER_ID_RE.match(clean_id):
            raise AuthError(400, "invalid_user_id", "Invalid user_id.")
        sid = secrets.token_urlsafe(24)
        csrf_token = secrets.token_urlsafe(24)
        now = time.time()
        ttl = self._ttl if ttl_seconds is None else max(1.0, float(ttl_seconds))
        expires_at = now + ttl
        rec = {
            "sid": sid,
            "user_id": clean_id,
            "csrf_token": csrf_token,
            "created_at": now,
            "expires_at": expires_at,
        }
        path = self._sessions_path()
        lock_path = path.with_name(f"{path.name}.lock")
        with user_file_lock(lock_path):
            sessions = self._read_json_dict(path)
            # Prune expired sessions to prevent unbounded session store growth
            sessions = {
                k: v for k, v in sessions.items()
                if isinstance(v, dict) and float(v.get("expires_at") or 0) > now
            }
            sessions[sid] = rec
            atomic_write_json(path, sessions)
        return {
            "token": self._sign_sid(sid),
            "user_id": clean_id,
            "csrf_token": csrf_token,
            "expires_at": expires_at,
        }

    def verify_session(self, token: Optional[str]) -> Optional[dict[str, Any]]:
        if not token:
            return None
        sid = self._verify_token_signature(token)
        if sid is None:
            return None
        path = self._sessions_path()
        lock_path = path.with_name(f"{path.name}.lock")
        now = time.time()
        with user_file_lock(lock_path):
            sessions = self._read_json_dict(path)
            rec = sessions.get(sid)
            if not isinstance(rec, dict):
                return None
            if float(rec.get("expires_at") or 0) <= now:
                sessions.pop(sid, None)
                try:
                    atomic_write_json(path, sessions)
                except OSError:
                    pass
                return None
            return dict(rec)

    def revoke_session(self, token: Optional[str]) -> bool:
        if not token:
            return False
        sid = self._verify_token_signature(token)
        if sid is None:
            return False
        path = self._sessions_path()
        lock_path = path.with_name(f"{path.name}.lock")
        with user_file_lock(lock_path):
            sessions = self._read_json_dict(path)
            existed = sid in sessions
            if existed:
                sessions.pop(sid, None)
                atomic_write_json(path, sessions)
            return existed


def is_auth_required(
    *,
    app_require_auth: bool = False,
    remote_addr: Optional[str] = None,
    forwarded_for: Optional[str] = None,
    bound_host: Optional[str] = None,
) -> bool:
    """Enforce authentication whenever configured OR when exposed beyond localhost."""
    if app_require_auth:
        return True
    if os.environ.get("NUTRITRACK_REQUIRE_AUTH", "").strip().lower() in ("1", "true", "yes"):
        return True
    host_val = (bound_host or os.environ.get("HOST", "127.0.0.1")).strip().lower()
    if host_val and host_val not in _LOOPBACK_ADDRS:
        return True
    if forwarded_for and forwarded_for.strip():
        first_ip = forwarded_for.split(",")[0].strip().lower()
        if first_ip and first_ip not in _LOOPBACK_ADDRS:
            return True
    if remote_addr and remote_addr.strip().lower() not in _LOOPBACK_ADDRS:
        return True
    return False
