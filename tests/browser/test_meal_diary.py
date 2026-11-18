"""Browser end-to-end tests for Food Diary add, preview, edit, delete, and persistence."""

from __future__ import annotations

from datetime import date, timedelta

from playwright.sync_api import Page

from tests.browser.conftest import NutriTrackServer, switch_screen


def test_food_diary_preview_add_edit_delete_and_persistence(
    app_page: Page, isolated_server: NutriTrackServer
) -> None:
    switch_screen(app_page, "food-diary")

    # Live preview before submission
    app_page.select_option("#diary-meal-category", "breakfast")
    with app_page.expect_response(lambda r: "/api/nutritrack/foods/preview" in r.url):
        app_page.fill("#diary-food-name", "paneer")
        app_page.fill("#diary-qty-value", "100")
        app_page.select_option("#diary-qty-unit", "g")
    app_page.wait_for_timeout(120)
    assert "265" in app_page.locator("#diary-live-preview").inner_text()

    # Add meal via UI form
    with app_page.expect_response(lambda r: r.url.endswith("/api/nutritrack/meals") and r.request.method == "POST"):
        app_page.click("#diary-submit-btn")
    app_page.wait_for_timeout(300)
    assert app_page.locator("#diary-cal-val").inner_text().replace(",", "").strip() == "265"

    # Edit meal (double quantity 100g -> 150g)
    app_page.locator("button[aria-label='Edit paneer']").click()
    app_page.fill("#diary-qty-value", "150")
    with app_page.expect_response(lambda r: "/api/nutritrack/meals/" in r.url and r.request.method == "PUT"):
        app_page.click("#diary-submit-btn")
    app_page.wait_for_timeout(300)
    assert app_page.locator("#diary-cal-val").inner_text().replace(",", "").strip() == "397.5"

    # Cross-check dashboard totals match diary
    switch_screen(app_page, "dashboard")
    assert float(app_page.locator("#dash-cal-consumed").inner_text().replace(",", "")) == 397.5

    # Navigate to historical date and add historical meal
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    app_page.fill("#header-date-input", yesterday)
    app_page.locator("#header-date-input").dispatch_event("change")
    app_page.wait_for_timeout(300)

    switch_screen(app_page, "food-diary")
    assert app_page.locator("#diary-cal-val").inner_text().strip() == "0"
    app_page.select_option("#diary-meal-category", "lunch")
    app_page.fill("#diary-food-name", "banana")
    app_page.fill("#diary-qty-value", "2")
    app_page.select_option("#diary-qty-unit", "piece")
    with app_page.expect_response(lambda r: r.url.endswith("/api/nutritrack/meals") and r.request.method == "POST"):
        app_page.click("#diary-submit-btn")
    app_page.wait_for_timeout(300)
    assert app_page.locator("#diary-cal-val").inner_text().strip() == "210"

    # Delete historical meal & verify persistence after reload and server restart
    with app_page.expect_response(lambda r: "/api/nutritrack/meals/" in r.url and r.request.method == "DELETE"):
        app_page.locator("button[aria-label='Delete banana']").click()
    app_page.wait_for_timeout(300)
    assert app_page.locator("#diary-cal-val").inner_text().strip() == "0"

    app_page.reload(wait_until="domcontentloaded")
    app_page.wait_for_timeout(300)
    switch_screen(app_page, "food-diary")
    assert app_page.locator("#diary-cal-val").inner_text().strip() == "0"

    isolated_server.restart()
    app_page.reload(wait_until="domcontentloaded")
    app_page.wait_for_timeout(350)
    switch_screen(app_page, "food-diary")
    assert app_page.locator("#diary-cal-val").inner_text().strip() == "0"

    # Back to Today -> 397.5 kcal paneer still intact
    app_page.click("#header-today-btn")
    app_page.wait_for_timeout(300)
    assert app_page.locator("#diary-cal-val").inner_text().replace(",", "").strip() == "397.5"
