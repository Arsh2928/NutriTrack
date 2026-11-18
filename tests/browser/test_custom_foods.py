"""Browser end-to-end tests for Custom Foods registration, editing, preview, deletion, and user isolation."""

from __future__ import annotations

from playwright.sync_api import Browser, Page

from tests.browser.conftest import NutriTrackServer, register_via_modal, switch_screen


def test_custom_foods_add_edit_preview_delete_and_validation(app_page: Page) -> None:
    switch_screen(app_page, "food-diary")
    app_page.click("#diary-toggle-custom-btn")
    app_page.wait_for_selector("#custom-food-drawer:not(.hidden)")

    # Invalid negative values caught gracefully without API error crash
    app_page.fill("#cf-name", "bad bar")
    app_page.evaluate("() => { document.querySelector('#cf-cal').value = '-50'; document.querySelector('#cf-prot').value = '10'; }")
    app_page.evaluate(
        "() => document.querySelector('#custom-food-form').dispatchEvent(new Event('submit', {cancelable: true}))"
    )
    app_page.wait_for_timeout(150)
    assert "Validation Error" in app_page.locator("#global-toast-title").inner_text()

    # Valid custom food create + immediate log
    app_page.fill("#cf-name", "protein oats bar")
    app_page.fill("#cf-cal", "240")
    app_page.fill("#cf-prot", "18")
    app_page.fill("#cf-carbs", "22")
    app_page.fill("#cf-fat", "7")
    with app_page.expect_response(lambda r: r.url.endswith("/api/nutritrack/custom-food") and r.request.method == "POST") as cr:
        app_page.click("#cf-save-log-btn")
    assert cr.value.status == 200
    app_page.wait_for_timeout(300)
    assert app_page.locator("#diary-cal-val").inner_text().strip() == "240"

    # Re-open custom drawer -> verify saved custom food is listed, edit its macros, verify preview, then delete
    app_page.click("#diary-toggle-custom-btn")
    app_page.wait_for_selector("#custom-foods-list")
    assert "protein oats bar" in app_page.locator("#custom-foods-list").inner_text().lower()

    app_page.locator("button[aria-label='Edit custom food protein oats bar']").click()
    app_page.fill("#cf-cal", "300")
    with app_page.expect_response(lambda r: r.url.endswith("/api/nutritrack/custom-food") and r.request.method == "POST"):
        app_page.click("#cf-save-log-btn")
    app_page.wait_for_timeout(300)

    # Verify live preview reflects updated 300 kcal custom food
    with app_page.expect_response(lambda r: "/api/nutritrack/foods/preview" in r.url):
        app_page.fill("#diary-food-name", "protein oats bar")
        app_page.fill("#diary-qty-value", "1")
        app_page.select_option("#diary-qty-unit", "piece")
    app_page.wait_for_timeout(150)
    assert "300" in app_page.locator("#diary-live-preview").inner_text()

    # Delete custom food from manager list
    app_page.click("#diary-toggle-custom-btn")
    with app_page.expect_response(lambda r: "/api/nutritrack/custom-food/" in r.url and r.request.method == "DELETE") as dr:
        app_page.locator("button[aria-label='Delete custom food protein oats bar']").click()
    assert dr.value.status == 200
    app_page.wait_for_timeout(250)
    assert "protein oats bar" not in app_page.locator("#custom-foods-list").inner_text().lower()


def test_custom_foods_private_between_two_users(edge_browser: Browser, isolated_server: NutriTrackServer) -> None:
    ctx_a = edge_browser.new_context(viewport={"width": 1366, "height": 768})
    ctx_b = edge_browser.new_context(viewport={"width": 1366, "height": 768})
    try:
        pa = ctx_a.new_page()
        pb = ctx_b.new_page()
        pa.goto(isolated_server.base_url, wait_until="domcontentloaded")
        register_via_modal(pa, "alice_cf", "alice-pass-123")
        pb.goto(isolated_server.base_url, wait_until="domcontentloaded")
        register_via_modal(pb, "bob_cf", "bob-pass-1234")

        # Both create "secret recipe" with different calories
        switch_screen(pa, "food-diary")
        pa.click("#diary-toggle-custom-btn")
        pa.fill("#cf-name", "secret recipe")
        pa.fill("#cf-cal", "200")
        pa.fill("#cf-prot", "10")
        pa.click("#cf-save-log-btn")
        pa.wait_for_timeout(300)

        switch_screen(pb, "food-diary")
        pb.click("#diary-toggle-custom-btn")
        assert "secret recipe" not in pb.locator("#custom-foods-list").inner_text().lower()
        pb.fill("#cf-name", "secret recipe")
        pb.fill("#cf-cal", "650")
        pb.fill("#cf-prot", "42")
        pb.click("#cf-save-log-btn")
        pb.wait_for_timeout(300)

        # Reload and confirm neither leaks into the other's preview/totals
        with pa.expect_response(lambda r: "/api/nutritrack/foods/preview" in r.url):
            pa.fill("#diary-food-name", "secret recipe")
        pa.wait_for_timeout(120)
        assert "200" in pa.locator("#diary-live-preview").inner_text()
        with pb.expect_response(lambda r: "/api/nutritrack/foods/preview" in r.url):
            pb.fill("#diary-food-name", "secret recipe")
        pb.wait_for_timeout(120)
        assert "650" in pb.locator("#diary-live-preview").inner_text()
    finally:
        ctx_a.close()
        ctx_b.close()
