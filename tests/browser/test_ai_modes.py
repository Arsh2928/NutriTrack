"""Browser verification of strict 3-mode AI selection, Gemini/Groq isolation, failure handling, and persistence."""

from __future__ import annotations

import pytest
from playwright.sync_api import Browser, Page

from tests.browser.conftest import NutriTrackServer, StubLLMServer, make_isolated_server, switch_screen


def test_strict_three_ai_modes_only_in_ui_and_api(app_page: Page, isolated_server: NutriTrackServer) -> None:
    opts = app_page.locator("#header-ai-mode-select option").all_text_contents()
    vals = app_page.locator("#header-ai-mode-select option").evaluate_all("els => els.map(e => e.value)")
    assert opts == ["Gemini", "Groq", "Offline"]
    assert vals == ["gemini", "groq", "offline"]

    switch_screen(app_page, "settings")
    chip_vals = app_page.locator("#settings-ai-mode-buttons .ai-mode-chip").evaluate_all(
        "els => els.map(e => e.getAttribute('data-ai-mode'))"
    )
    assert chip_vals == ["gemini", "groq", "offline"]

    # Reject legacy invalid mode strings via API
    for invalid_mode in ("automatic", "auto", "online", "github", "openai"):
        r = app_page.request.post(
            f"{isolated_server.base_url}/api/nutritrack/ai/mode",
            data={"mode": invalid_mode},
        )
        assert r.status == 400


@pytest.mark.parametrize("provider_mode", ["gemini", "groq"])
def test_provider_isolation_error_codes_timeouts_and_manual_retry(
    edge_browser: Browser, provider_mode: str
) -> None:
    gem_stub = StubLLMServer(name="gemini", default_reply="Gemini protein recovery guidance.")
    groq_stub = StubLLMServer(name="groq", default_reply="Groq balanced macro guidance.")
    tmp, srv = make_isolated_server(
        ai_mode=provider_mode,
        gemini_url=gem_stub.base_url,
        groq_url=groq_stub.base_url,
        gemini_key="stub-gemini-key",
        groq_key="stub-groq-key",
        extra_env={
            "AI_CIRCUIT_FAILURE_THRESHOLD": "25",
            "AI_MAX_RETRIES": "0",
            "AI_REQUEST_TIMEOUT_SECONDS": "1.2",
            "NUTRITRACK_TURN_TIMEOUT_SECONDS": "2.0",
        },
    )
    ctx = edge_browser.new_context(viewport={"width": 1366, "height": 768})
    try:
        page: Page = ctx.new_page()
        page.goto(srv.base_url, wait_until="domcontentloaded")
        page.wait_for_timeout(300)

        assert page.locator("#header-ai-mode-select").input_value() == provider_mode
        active_stub = gem_stub if provider_mode == "gemini" else groq_stub
        other_stub = groq_stub if provider_mode == "gemini" else gem_stub

        # 1. Valid complex-reasoning query hits ONLY selected provider
        query = "Explain why spacing protein meals across the day supports muscle recovery?"
        page.fill("#dash-agent-input", query)
        with page.expect_response(lambda r: r.url.endswith("/api/nutritrack/chat")) as ok_res:
            page.locator("#dash-agent-input").press("Enter")
        assert ok_res.value.status == 200
        ok_json = ok_res.value.json()
        assert ok_json["status"] == "success"
        assert len(active_stub.calls) >= 1
        assert len(other_stub.calls) == 0, f"Unexpected leak into {other_stub.name}!"

        # 2. HTTP 400, 401, 403, 404, 429, 500, 503 + Timeout keep mode pinned & preserve user input!
        err_query = "What healthy snack advice do you suggest for late evening?"
        for code in (400, 401, 403, 404, 429, 500, 503):
            active_stub.clear()
            other_stub.clear()
            active_stub.enqueue({"status": code, "headers": {"Retry-After": "1"} if code == 429 else {}})
            page.fill("#dash-agent-input", err_query)
            with page.expect_response(lambda r: r.url.endswith("/api/nutritrack/chat")) as fail_res:
                page.click("#dash-agent-submit")
            assert fail_res.value.status >= 400
            fail_json = fail_res.value.json()
            assert fail_json["status"] == "failed"
            assert fail_json["ai_status"]["selected_mode"] == provider_mode
            assert page.locator("#header-ai-mode-select").input_value() == provider_mode
            assert page.locator("#dash-agent-input").input_value() == err_query
            assert len(other_stub.calls) == 0

        # Simulate upstream timeout
        active_stub.clear()
        other_stub.clear()
        active_stub.enqueue({"sleep": 3.5})
        page.fill("#dash-agent-input", err_query)
        with page.expect_response(lambda r: r.url.endswith("/api/nutritrack/chat"), timeout=10000) as to_res:
            page.click("#dash-agent-submit")
        assert to_res.value.status >= 400
        assert page.locator("#header-ai-mode-select").input_value() == provider_mode
        assert page.locator("#dash-agent-input").input_value() == err_query
        assert len(other_stub.calls) == 0

        # 3. Click Retry Connection to recover after simulated outage, then manual retry succeeds cleanly
        active_stub.clear()
        page.click("#header-ai-status-btn")
        with page.expect_response(lambda r: r.url.endswith("/api/nutritrack/ai/retry")):
            page.click("#header-ai-retry-btn")
        page.click("#header-ai-status-btn")
        page.wait_for_timeout(150)
        with page.expect_response(lambda r: r.url.endswith("/api/nutritrack/chat")) as retry_res:
            page.click("#dash-agent-submit")
        assert retry_res.value.status == 200
        assert retry_res.value.json()["status"] == "success"
        assert len(active_stub.calls) >= 1
        assert len(other_stub.calls) == 0
    finally:
        ctx.close()
        srv.stop()
        tmp.cleanup()
        gem_stub.close()
        groq_stub.close()


def test_ai_mode_persists_across_reload_tabs_and_server_restart(edge_browser: Browser) -> None:
    tmp, srv = make_isolated_server(
        ai_mode="gemini",
        gemini_key="stub-g",
        groq_key="stub-q",
    )
    ctx = edge_browser.new_context(viewport={"width": 1366, "height": 768})
    try:
        p1 = ctx.new_page()
        p1.goto(srv.base_url, wait_until="domcontentloaded")
        p1.wait_for_timeout(300)
        assert p1.locator("#header-ai-mode-select").input_value() == "gemini"

        with p1.expect_response(lambda r: r.url.endswith("/api/nutritrack/ai/mode") and r.request.method == "POST"):
            p1.select_option("#header-ai-mode-select", "groq")
        p1.wait_for_timeout(200)

        # Persist across reload & navigation
        p1.reload(wait_until="domcontentloaded")
        p1.wait_for_timeout(250)
        assert p1.locator("#header-ai-mode-select").input_value() == "groq"
        switch_screen(p1, "settings")
        assert "Groq" in p1.locator("#settings-ai-mode-desc").inner_text()

        # Persist in second browser tab
        p2 = ctx.new_page()
        p2.goto(srv.base_url, wait_until="domcontentloaded")
        p2.wait_for_timeout(250)
        assert p2.locator("#header-ai-mode-select").input_value() == "groq"

        # Persist across server restart
        srv.restart()
        p1.reload(wait_until="domcontentloaded")
        p1.wait_for_timeout(300)
        assert p1.locator("#header-ai-mode-select").input_value() == "groq"
    finally:
        ctx.close()
        srv.stop()
        tmp.cleanup()
