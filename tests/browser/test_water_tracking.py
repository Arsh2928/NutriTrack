"""Browser end-to-end tests for Water Tracking (exact fractions, presets, custom set, limits, persistence)."""

from __future__ import annotations

from playwright.sync_api import Page

from tests.browser.conftest import NutriTrackServer


def test_water_tracking_presets_fractions_validation_and_restart(
    app_page: Page, isolated_server: NutriTrackServer
) -> None:
    assert app_page.locator("#dash-water-consumed").inner_text().strip() == "0 L"

    # +250 ml must render exact "0.25 L" (not rounded to "0.3 L")
    with app_page.expect_response(lambda r: r.url.endswith("/api/nutritrack/water") and r.request.method == "POST"):
        app_page.click("#dash-add-water-btn")
    app_page.wait_for_timeout(250)
    assert app_page.locator("#dash-water-consumed").inner_text().strip() == "0.25 L"

    # +500 ml -> 0.75 L
    with app_page.expect_response(lambda r: r.url.endswith("/api/nutritrack/water") and r.request.method == "POST"):
        app_page.click("#dash-add-water-500-btn")
    app_page.wait_for_timeout(250)
    assert app_page.locator("#dash-water-consumed").inner_text().strip() == "0.75 L"

    # +1 L -> 1.75 L
    with app_page.expect_response(lambda r: r.url.endswith("/api/nutritrack/water") and r.request.method == "POST"):
        app_page.click("#dash-add-water-1l-btn")
    app_page.wait_for_timeout(250)
    assert app_page.locator("#dash-water-consumed").inner_text().strip() == "1.75 L"

    # Fractional set via custom water input (e.g. 1.35 L)
    app_page.fill("#water-custom-liters", "1.35")
    with app_page.expect_response(lambda r: r.url.endswith("/api/nutritrack/water") and r.request.method == "POST"):
        app_page.click("#water-set-btn")
    app_page.wait_for_timeout(250)
    assert app_page.locator("#dash-water-consumed").inner_text().strip() == "1.35 L"

    # Invalid negative water blocked in UI (bypass HTML5 validation via JS submit/negative check)
    app_page.evaluate("() => { document.querySelector('#water-custom-liters').value = '-1'; }")
    app_page.evaluate(
        "() => document.querySelector('#water-custom-form').dispatchEvent(new Event('submit', {cancelable: true}))"
    )
    app_page.wait_for_timeout(150)
    assert "Validation Error" in app_page.locator("#global-toast-title").inner_text()
    assert app_page.locator("#dash-water-consumed").inner_text().strip() == "1.35 L"

    # Reload & server restart persist exact 1.35 L
    app_page.reload(wait_until="domcontentloaded")
    app_page.wait_for_timeout(300)
    assert app_page.locator("#dash-water-consumed").inner_text().strip() == "1.35 L"

    isolated_server.restart()
    app_page.reload(wait_until="domcontentloaded")
    app_page.wait_for_timeout(300)
    assert app_page.locator("#dash-water-consumed").inner_text().strip() == "1.35 L"
