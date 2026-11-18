"""Atomic JSON writing, bounded reference-counted thread locks, and cross-process per-user file locking."""
from contextlib import contextmanager
import json
import os
import tempfile
import threading
import time
from pathlib import Path


class LockRegistry:
    """Thread-safe keyed RLock registry with reference-counted cleanup to avoid unbounded memory growth."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._entries: dict[str, list] = {}  # key -> [RLock, ref_count]

    def _acquire_ref(self, key: str) -> threading.RLock:
        with self._guard:
            entry = self._entries.get(key)
            if entry is None:
                entry = [threading.RLock(), 0]
                self._entries[key] = entry
            entry[1] += 1
            return entry[0]

    def _release_ref(self, key: str) -> None:
        with self._guard:
            entry = self._entries.get(key)
            if entry is None:
                return
            entry[1] -= 1
            if entry[1] <= 0:
                self._entries.pop(key, None)

    @contextmanager
    def lock(self, key: str):
        rlock = self._acquire_ref(key)
        try:
            rlock.acquire()
            try:
                yield
            finally:
                rlock.release()
        finally:
            self._release_ref(key)

    def active_count(self) -> int:
        with self._guard:
            return len(self._entries)


_THREAD_LOCKS = LockRegistry()
_LOCK_DEPTH_LOCAL = threading.local()


def get_lock_registry() -> LockRegistry:
    """Return the shared state lock registry (exposed for diagnostics/tests)."""
    return _THREAD_LOCKS


@contextmanager
def user_file_lock(lock_path, timeout_seconds: float = 10.0):
    """Reentrant cross-thread and cross-process advisory lock using an external `.lock` file."""
    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    key = str(path.resolve())
    with _THREAD_LOCKS.lock(key):
        depths = getattr(_LOCK_DEPTH_LOCAL, "depths", None)
        if depths is None:
            depths = {}
            _LOCK_DEPTH_LOCAL.depths = depths
        depth = depths.get(key, 0)
        if depth > 0:
            depths[key] = depth + 1
            try:
                yield
            finally:
                depths[key] = depth
            return

        fh = open(path, "a+b")
        locked = False
        try:
            deadline = time.monotonic() + max(0.5, float(timeout_seconds))
            if os.name == "nt":
                import msvcrt

                while True:
                    try:
                        if os.fstat(fh.fileno()).st_size == 0:
                            fh.seek(0)
                            fh.write(b"\0")
                            fh.flush()
                        fh.seek(0)
                        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                        locked = True
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError(f"Timed out waiting for file lock {path}")
                        time.sleep(0.005)
            else:
                import fcntl

                while True:
                    try:
                        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        locked = True
                        break
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError(f"Timed out waiting for file lock {path}")
                        time.sleep(0.005)
            depths[key] = 1
            yield
        finally:
            depths.pop(key, None)
            if locked:
                try:
                    if os.name == "nt":
                        import msvcrt

                        fh.seek(0)
                        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
            fh.close()


def atomic_write_json(path, data):
    """Write `data` as JSON so that a crash or full disk can never leave a
    half-written file: write a temp file in the SAME folder, flush it to disk,
    then os.replace() it over the target (atomic on every supported OS).
    Raises on failure and leaves the previous file untouched."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False, allow_nan=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise
