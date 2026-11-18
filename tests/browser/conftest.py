"""Shared Microsoft Edge Playwright fixtures and isolated server/stub infrastructure for NutriTrack browser QA."""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
import http.server
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Iterator, Optional
import urllib.request

from playwright.sync_api import Browser, BrowserContext, Page, sync_playwright
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class StubLLMServer:
    """Configurable local OpenAI-compatible HTTP server simulating Gemini/Groq responses & error codes."""

    def __init__(self, *, name: str = "stub", default_reply: str = "Verified AI response.") -> None:
        self.name = name
        self.default_reply = default_reply
        self.calls: list[dict[str, Any]] = []
        self._behaviors: deque[dict[str, Any]] = deque()
        self._lock = threading.Lock()
        self.port = find_free_port()
        self.base_url = f"http://127.0.0.1:{self.port}/v1"
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:
                return

            def _read_json(self) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length > 0 else b"{}"
                try:
                    return json.loads(raw.decode("utf-8"))
                except Exception:
                    return {}

            def do_GET(self) -> None:
                payload = json.dumps({"object": "list", "data": [{"id": "stub-model"}]}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self) -> None:
                body = self._read_json()
                with owner._lock:
                    owner.calls.append(
                        {
                            "path": self.path,
                            "model": body.get("model"),
                            "messages_count": len(body.get("messages") or []),
                        }
                    )
                    behavior = owner._behaviors.popleft() if owner._behaviors else {}

                delay = float(behavior.get("sleep") or 0.0)
                if delay > 0:
                    time.sleep(delay)

                status = int(behavior.get("status") or 200)
                headers = behavior.get("headers") or {}
                if status >= 400:
                    err_obj = behavior.get("body") or {
                        "error": {"message": f"{owner.name} simulated HTTP {status}", "code": status}
                    }
                    payload = json.dumps(err_obj).encode("utf-8")
                    try:
                        self.send_response(status)
                        self.send_header("Content-Type", "application/json")
                        for hk, hv in headers.items():
                            self.send_header(str(hk), str(hv))
                        self.send_header("Content-Length", str(len(payload)))
                        self.end_headers()
                        self.wfile.write(payload)
                    except OSError:
                        pass
                    return

                reply_text = str(behavior.get("reply") or owner.default_reply)
                model_name = str(body.get("model") or "stub-model")
                resp_obj = {
                    "id": f"chatcmpl-{owner.name}-{int(time.time() * 1000)}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": reply_text},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 10, "total_tokens": 22},
                }
                payload = json.dumps(resp_obj).encode("utf-8")
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except OSError:
                    pass

        self._httpd = http.server.ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def enqueue(self, *items: dict[str, Any]) -> None:
        with self._lock:
            self._behaviors.extend(items)

    def clear(self) -> None:
        with self._lock:
            self.calls.clear()
            self._behaviors.clear()

    def close(self) -> None:
        try:
            self._httpd.shutdown()
            self._httpd.server_close()
        except Exception:
            pass


@dataclass
class NutriTrackServer:
    """Isolated live Flask server process with restart capability and temp storage."""

    root_dir: Path
    port: int
    env: dict[str, str]
    proc: Optional[subprocess.Popen[Any]] = field(default=None, init=False)
    _out_fh: Any = field(default=None, init=False)
    _err_fh: Any = field(default=None, init=False)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, *, timeout_s: float = 15.0) -> None:
        self.stop()
        log_out = self.root_dir / "server.out.log"
        log_err = self.root_dir / "server.err.log"
        self._out_fh = open(log_out, "ab")
        self._err_fh = open(log_err, "ab")
        proc_env = os.environ.copy()
        proc_env.update(self.env)
        proc_env["HOST"] = "127.0.0.1"
        proc_env["PORT"] = str(self.port)
        proc_env["FLASK_DEBUG"] = "0"
        proc_env["PYTHONUNBUFFERED"] = "1"
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "nutritrack.web.app"],
            cwd=str(PROJECT_ROOT),
            env=proc_env,
            stdout=self._out_fh,
            stderr=self._err_fh,
        )
        deadline = time.monotonic() + timeout_s
        last_err = ""
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                err_txt = log_err.read_text(encoding="utf-8", errors="replace") if log_err.exists() else ""
                raise RuntimeError(f"NutriTrack test server exited early ({self.proc.returncode}): {err_txt[-600:]}")
            try:
                with urllib.request.urlopen(f"{self.base_url}/api/nutritrack/auth/me", timeout=1.5) as resp:
                    if resp.status == 200:
                        return
            except Exception as exc:
                last_err = str(exc)
            time.sleep(0.12)
        raise TimeoutError(f"Server on {self.base_url} did not become ready within {timeout_s}s: {last_err}")

    def stop(self) -> None:
        if self.proc is not None:
            try:
                if self.proc.poll() is None:
                    self.proc.terminate()
                    try:
                        self.proc.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        self.proc.kill()
                        self.proc.wait(timeout=3.0)
            except Exception:
                pass
            self.proc = None
        for fh_attr in ("_out_fh", "_err_fh"):
            fh = getattr(self, fh_attr, None)
            if fh is not None:
                try:
                    fh.close()
                except Exception:
                    pass
                setattr(self, fh_attr, None)

    def restart(self) -> None:
        self.stop()
        self.start()


def make_isolated_server(
    *,
    ai_mode: str = "offline",
    gemini_url: Optional[str] = None,
    groq_url: Optional[str] = None,
    gemini_key: str = "",
    groq_key: str = "",
    extra_env: Optional[dict[str, str]] = None,
) -> tuple[tempfile.TemporaryDirectory[str], NutriTrackServer]:
    tmp = tempfile.TemporaryDirectory(prefix="nt_browser_qa_", ignore_cleanup_errors=True)
    root = Path(tmp.name)
    data_dir = root / "data"
    logs_dir = root / "logs"
    data_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    env = {
        "NUTRITRACK_DATA_DIR": str(data_dir),
        "NUTRITRACK_LOG_DIR": str(logs_dir),
        "NUTRITRACK_CUSTOM_FOODS": str(data_dir / "custom_foods.json"),
        "NUTRITRACK_DAILY_STORE": str(data_dir / "daily_store.json"),
        "NUTRITRACK_CA1_LOG": str(logs_dir / "ca1.jsonl"),
        "NUTRITRACK_AI_MODE": ai_mode,
        "GEMINI_API_KEY": gemini_key,
        "GROQ_API_KEY": groq_key,
        "OPENAI_API_KEY": "",
        "GITHUB_TOKEN": "",
        "AI_RETRY_BASE_DELAY_SECONDS": "0.05",
        "AI_RETRY_MAX_DELAY_SECONDS": "0.1",
        "AI_MAX_RETRIES": "1",
    }
    if gemini_url:
        env["GEMINI_BASE_URL"] = gemini_url
    if groq_url:
        env["GROQ_BASE_URL"] = groq_url
    if extra_env:
        env.update(extra_env)
    srv = NutriTrackServer(root_dir=root, port=find_free_port(), env=env)
    srv.start()
    return tmp, srv


@pytest.fixture(scope="session")
def edge_browser() -> Iterator[Browser]:
    """Session-scoped Microsoft Edge Playwright browser instance."""
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="msedge", headless=True)
        yield browser
        browser.close()


@pytest.fixture()
def isolated_server() -> Iterator[NutriTrackServer]:
    """Fresh isolated NutriTrack server per test in Offline mode with dedicated temp data/log dirs."""
    tmp, srv = make_isolated_server(ai_mode="offline")
    try:
        yield srv
    finally:
        srv.stop()
        tmp.cleanup()


@pytest.fixture()
def edge_context(edge_browser: Browser) -> Iterator[BrowserContext]:
    ctx = edge_browser.new_context(viewport={"width": 1366, "height": 768})
    yield ctx
    ctx.close()


@pytest.fixture()
def app_page(edge_context: BrowserContext, isolated_server: NutriTrackServer) -> Iterator[Page]:
    """Open NutriTrack in Edge and assert zero uncaught page errors across the test."""
    page = edge_context.new_page()
    page_errors: list[str] = []
    page.on("pageerror", lambda err: page_errors.append(str(err)))
    page.goto(isolated_server.base_url, wait_until="domcontentloaded")
    page.wait_for_function("() => document.querySelector('#header-mode-pill') !== null")
    page.wait_for_timeout(250)
    yield page
    assert not page_errors, f"Uncaught JavaScript page errors: {page_errors}"


def switch_screen(page: Page, screen_key: str) -> None:
    link = page.locator(f"#sidebar-nav a[data-path='{screen_key}']")
    menu_btn = page.locator("#mobile-menu-btn")
    if menu_btn.is_visible():
        box = link.bounding_box()
        if box is None or (box["x"] + box["width"]) <= 5:
            menu_btn.click()
            page.wait_for_timeout(240)
    link.click()
    page.wait_for_selector(f"#screen-{screen_key}:not(.hidden)", timeout=5000)


def send_chat(page: Page, prompt: str, *, input_selector: str = "#dash-agent-input") -> dict[str, Any]:
    with page.expect_response(lambda r: r.url.endswith("/api/nutritrack/chat") and r.request.method == "POST", timeout=15000) as resp_info:
        page.fill(input_selector, prompt)
        page.locator(input_selector).press("Enter")
    resp = resp_info.value
    data = resp.json()
    page.wait_for_timeout(200)
    return {"http_status": resp.status, "data": data}


def register_via_modal(page: Page, user_id: str, password: str) -> None:
    modal = page.locator("#auth-modal")
    if not modal.is_visible():
        page.click("#header-auth-btn")
        modal.wait_for(state="visible", timeout=5000)
    page.fill("#auth-user-id", user_id)
    page.fill("#auth-password", password)
    with page.expect_response(lambda r: "/api/nutritrack/auth/register" in r.url and r.request.method == "POST") as resp_info:
        page.click("#auth-register-btn")
    assert resp_info.value.status == 201
    modal.wait_for(state="hidden", timeout=5000)
    page.wait_for_timeout(250)
