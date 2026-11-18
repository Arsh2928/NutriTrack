"""Browser console, network HTTP error sweep, security headers, and XSS payload injection resistance tests."""

from __future__ import annotations

from playwright.sync_api import Page

from tests.browser.conftest import NutriTrackServer, register_via_modal, send_chat, switch_screen


def test_clean_flow_zero_console_errors_and_security_headers(
    app_page: Page, isolated_server: NutriTrackServer
) -> None:
    console_errors: list[str] = []
    bad_responses: list[tuple[int, str]] = []
    app_page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)
    app_page.on("response", lambda r: bad_responses.append((r.status, r.url)) if r.status >= 400 else None)

    for screen_key in ("dashboard", "food-diary", "recommendations", "weekly-progress", "agents", "settings"):
        switch_screen(app_page, screen_key)
        app_page.wait_for_timeout(150)

    switch_screen(app_page, "dashboard")
    send_chat(app_page, "Log 1 banana")
    app_page.click("#dash-add-water-btn")
    app_page.wait_for_timeout(200)

    # Check security headers on API & index responses
    api_resp = app_page.request.get(f"{isolated_server.base_url}/api/nutritrack/state")
    hdrs = {k.lower(): v for k, v in api_resp.headers.items()}
    assert hdrs.get("x-content-type-options") == "nosniff"
    assert hdrs.get("x-frame-options") == "SAMEORIGIN"
    assert hdrs.get("referrer-policy") == "no-referrer"
    assert "no-store" in (hdrs.get("cache-control") or "")

    assert console_errors == [], f"Unexpected console errors: {console_errors}"
    assert bad_responses == [], f"Unexpected >=400 HTTP responses during clean flow: {bad_responses}"


def test_xss_payload_rendering_is_strictly_escaped_in_dom(app_page: Page, isolated_server: NutriTrackServer) -> None:
    dialogs: list[str] = []
    app_page.on("dialog", lambda d: (dialogs.append(d.message), d.dismiss()))

    # Attempt XSS via Profile Name + Custom Food name
    xss_name = "<img src=x onerror=alert('XSS')>"
    switch_screen(app_page, "settings")
    app_page.fill("#profile-name", xss_name)
    app_page.click("#save-btn")
    app_page.wait_for_timeout(300)

    # Literal text shown safely via textContent, ZERO script/dialog execution
    assert xss_name in app_page.locator("#header-user-name").inner_text()
    assert dialogs == []
    assert app_page.locator("#header-user-name img").count() == 0

    switch_screen(app_page, "food-diary")
    app_page.click("#diary-toggle-custom-btn")
    app_page.fill("#cf-name", "<b>evil_food</b>")
    app_page.fill("#cf-cal", "100")
    app_page.fill("#cf-prot", "5")
    app_page.click("#cf-save-log-btn")
    app_page.wait_for_timeout(300)
    assert app_page.locator("#diary-categories-container b").count() == 0
    assert dialogs == []
