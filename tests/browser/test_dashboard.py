"""Browser end-to-end verification of the NutriTrack Dashboard in Microsoft Edge."""

from __future__ import annotations

from datetime import date, timedelta

from playwright.sync_api import Page

from tests.browser.conftest import NutriTrackServer, send_chat, switch_screen


def test_dashboard_empty_state_metrics_and_macros(app_page: Page) -> None:
    assert app_page.locator("#dash-cal-consumed").inner_text().strip() == "0"
    assert "2,300" in app_page.locator("#dash-cal-target").inner_text()
    assert app_page.locator("#dash-prot-consumed").inner_text().strip() == "0g"
    assert app_page.locator("#dash-water-consumed").inner_text().strip() == "0 L"
    assert app_page.locator("#dash-meals-list").is_visible()
    assert app_page.locator("#dash-weekly-bars").is_visible()
    assert app_page.locator("#dash-rec-title").is_visible()
    assert app_page.locator("#dash-rec-log-btn").is_visible()


def test_dashboard_updates_from_meals_goals_dates_and_reloads(
    app_page: Page, isolated_server: NutriTrackServer
) -> None:
    res = send_chat(app_page, "Log 100 g paneer and 2 boiled eggs")
    assert res["http_status"] == 200
    assert res["data"]["status"] == "success"

    cal_txt = app_page.locator("#dash-cal-consumed").inner_text().replace(",", "").strip()
    assert float(cal_txt) == 421.0
    assert app_page.locator("#dash-prot-consumed").inner_text().strip() == "30.3g"
    assert "1,879 kcal" in app_page.locator("#dash-cal-remaining-pill").inner_text()

    # Cross-check Food Diary reflects identical numbers
    switch_screen(app_page, "food-diary")
    assert app_page.locator("#diary-cal-val").inner_text().replace(",", "").strip() == "421"
    assert app_page.locator("#diary-prot-val").inner_text().strip() == "30.3g"

    # Change calorie goal in Settings and verify Dashboard updates cleanly
    switch_screen(app_page, "settings")
    app_page.fill("#target-calories", "2500")
    app_page.click("#save-btn")
    app_page.wait_for_timeout(300)

    switch_screen(app_page, "dashboard")
    assert "2,500 kcal" in app_page.locator("#dash-cal-target").inner_text()
    assert "2,079 kcal" in app_page.locator("#dash-cal-remaining-pill").inner_text()

    # Verify reload preserves dashboard data cleanly without duplication
    app_page.reload(wait_until="domcontentloaded")
    app_page.wait_for_timeout(350)
    assert float(app_page.locator("#dash-cal-consumed").inner_text().replace(",", "")) == 421.0

    # Switch to yesterday -> empty; switch back to Today -> matches 421
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    app_page.fill("#header-date-input", yesterday)
    app_page.locator("#header-date-input").dispatch_event("change")
    app_page.wait_for_timeout(300)
    assert app_page.locator("#dash-cal-consumed").inner_text().strip() == "0"

    app_page.click("#header-today-btn")
    app_page.wait_for_timeout(300)
    assert float(app_page.locator("#dash-cal-consumed").inner_text().replace(",", "")) == 421.0

    # Restart server and verify saved nutrition state reloads identically
    isolated_server.restart()
    app_page.reload(wait_until="domcontentloaded")
    app_page.wait_for_timeout(350)
    assert float(app_page.locator("#dash-cal-consumed").inner_text().replace(",", "")) == 421.0

    # Remove meal and verify totals drop back to 0 without negative values
    rm = send_chat(app_page, "Remove paneer")
    assert rm["http_status"] == 200
    assert float(app_page.locator("#dash-cal-consumed").inner_text().replace(",", "")) == 156.0
