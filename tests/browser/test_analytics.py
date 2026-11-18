"""Browser & API analytics verification: cache hits on reads, invalidation on partial commits, and 7-day trends."""

from __future__ import annotations

from datetime import date, timedelta

from playwright.sync_api import Page

from tests.browser.conftest import NutriTrackServer, send_chat, switch_screen


def test_analytics_cache_invalidation_and_weekly_accuracy(
    app_page: Page, isolated_server: NutriTrackServer
) -> None:
    base = isolated_server.base_url
    today = date.today().isoformat()
    yesterday = (date.today() - timedelta(days=1)).isoformat()

    # Log 100 g paneer today (265 kcal) & 2 bananas yesterday (210 kcal)
    send_chat(app_page, "Log 100 g paneer")
    send_chat(app_page, f"Log 2 bananas on {yesterday}")

    # 1. Read-only chat queries must NOT invalidate insights cache (repeat GET stays cache hit!)
    i1 = app_page.request.get(f"{base}/api/nutritrack/insights?days=7&end_date={today}").json()
    i2 = app_page.request.get(f"{base}/api/nutritrack/insights?days=7&end_date={today}").json()
    assert i2.get("cache", {}).get("cache_hit") is True

    send_chat(app_page, "Show today's calories")
    i3 = app_page.request.get(f"{base}/api/nutritrack/insights?days=7&end_date={today}").json()
    assert i3.get("cache", {}).get("cache_hit") is True, "Read-only chat query spuriously invalidated analytics cache!"

    # 2. Partial commit in /chat (needs_input with 1 saved item) MUST bump data_version and update analytics immediately!
    ver_before = i3["cache"]["data_version"]
    partial = send_chat(app_page, "Log 1 apple and 100 g dragonfruit")
    assert partial["data"]["status"] == "needs_input"
    assert len(partial["data"].get("added") or []) == 1  # apple saved!
    i4 = app_page.request.get(f"{base}/api/nutritrack/insights?days=7&end_date={today}").json()
    assert i4["cache"]["data_version"] > ver_before, "Partial commit did not bump analytics data_version!"
    # Today: 265 + 95 = 360 kcal; Yesterday: 210 kcal -> 2-day avg = 285 kcal (5 empty days NOT counted as zero)
    assert i4["trends"]["days_with_data"] == 2
    assert i4["trends"]["averages"]["calories"] == 285.0

    # Verify UI Weekly Progress reflects exact 285 avg kcal & 2/7 days recorded
    switch_screen(app_page, "weekly-progress")
    app_page.wait_for_timeout(300)
    assert app_page.locator("#weekly-avg-cal").inner_text().replace(",", "").strip() == "285"
    assert "2 of 7 days recorded" in app_page.locator("#weekly-days-hit-note").inner_text()
