"""Offline Mode isolation & full-app usability verification with external network hard-blocked in Edge."""

from __future__ import annotations

from urllib.parse import urlparse

from playwright.sync_api import Browser, Page, Route

from tests.browser.conftest import make_isolated_server, send_chat, switch_screen


def test_offline_mode_full_workflow_and_zero_external_network_calls(edge_browser: Browser) -> None:
    tmp, srv = make_isolated_server(ai_mode="offline")
    ctx = edge_browser.new_context(viewport={"width": 1366, "height": 768})
    blocked_external_urls: list[str] = []

    def _block_non_local(route: Route) -> None:
        host = urlparse(route.request.url).hostname or ""
        if host in ("127.0.0.1", "localhost", "::1"):
            route.continue_()
        else:
            blocked_external_urls.append(route.request.url)
            route.abort()

    ctx.route("**/*", _block_non_local)
    try:
        page: Page = ctx.new_page()
        page.goto(srv.base_url, wait_until="domcontentloaded")
        page.wait_for_timeout(300)

        assert page.locator("#header-ai-mode-select").input_value() == "offline"
        assert "Offline Mode Active" in page.locator("#header-mode-pill").inner_text()

        # Log, query, remove via Chat
        r1 = send_chat(page, "Log 100 g paneer and 2 boiled eggs")
        assert r1["data"]["status"] == "success"
        r2 = send_chat(page, "Compare this week with last week")
        assert r2["data"]["status"] == "success"
        r3 = send_chat(page, "Show my meal history")
        assert r3["data"]["status"] == "success"
        rm = send_chat(page, "Remove paneer")
        assert rm["data"]["status"] == "success"

        # Add meal via Diary, view Weekly Progress, view Recommendations, update Settings, track Water
        switch_screen(page, "food-diary")
        page.fill("#diary-food-name", "banana")
        page.fill("#diary-qty-value", "2")
        page.select_option("#diary-qty-unit", "piece")
        with page.expect_response(lambda r: r.url.endswith("/api/nutritrack/meals") and r.request.method == "POST"):
            page.click("#diary-submit-btn")
        page.wait_for_timeout(300)
        assert float(page.locator("#diary-cal-val").inner_text()) == 366.0  # 156 + 210

        switch_screen(page, "weekly-progress")
        page.wait_for_timeout(250)
        assert "366" in page.locator("#weekly-avg-cal").inner_text()

        switch_screen(page, "recommendations")
        page.wait_for_timeout(350)
        assert page.locator("#rec-featured-container").is_visible()

        switch_screen(page, "settings")
        page.fill("#profile-name", "Offline Tester")
        page.fill("#target-calories", "2100")
        with page.expect_response(lambda r: r.url.endswith("/api/nutritrack/profile") and r.request.method == "POST"):
            page.click("#save-btn")
        page.wait_for_timeout(250)

        switch_screen(page, "dashboard")
        with page.expect_response(lambda r: r.url.endswith("/api/nutritrack/water") and r.request.method == "POST"):
            page.click("#dash-add-water-btn")
        page.wait_for_timeout(300)
        assert page.locator("#dash-water-consumed").inner_text().strip() == "0.25 L"
        assert "2,100 kcal" in page.locator("#dash-cal-target").inner_text()

        # Refresh & restart server -> stays Offline cleanly
        page.reload(wait_until="domcontentloaded")
        page.wait_for_timeout(300)
        assert page.locator("#header-ai-mode-select").input_value() == "offline"
        assert float(page.locator("#dash-cal-consumed").inner_text().replace(",", "")) == 366.0

        srv.restart()
        page.reload(wait_until="domcontentloaded")
        page.wait_for_timeout(300)
        assert page.locator("#header-ai-mode-select").input_value() == "offline"
        assert page.locator("#dash-water-consumed").inner_text().strip() == "0.25 L"

        assert blocked_external_urls == [], f"Offline mode made external requests: {blocked_external_urls}"
    finally:
        ctx.close()
        srv.stop()
        tmp.cleanup()
