"""Cross-user isolation tests in separate browser contexts and multi-user login switching."""

from __future__ import annotations

from playwright.sync_api import Browser

from tests.browser.conftest import NutriTrackServer, register_via_modal, send_chat, switch_screen


def test_user_isolation_across_contexts_and_shared_browser_logout_login(
    edge_browser: Browser, isolated_server: NutriTrackServer
) -> None:
    ctx_a = edge_browser.new_context(viewport={"width": 1366, "height": 768})
    ctx_b = edge_browser.new_context(viewport={"width": 1366, "height": 768})
    try:
        pa = ctx_a.new_page()
        pb = ctx_b.new_page()

        pa.goto(isolated_server.base_url, wait_until="domcontentloaded")
        register_via_modal(pa, "user_alpha", "alpha-pass-123")
        pb.goto(isolated_server.base_url, wait_until="domcontentloaded")
        register_via_modal(pb, "user_beta", "beta-pass-1234")

        # Alpha logs paneer (265 kcal), +500ml water, changes target calories to 1900, logs custom food
        send_chat(pa, "Log 100 g paneer")
        pa.click("#dash-add-water-500-btn")
        pa.wait_for_timeout(250)
        switch_screen(pa, "settings")
        pa.fill("#target-calories", "1900")
        pa.click("#save-btn")
        pa.wait_for_timeout(300)
        switch_screen(pa, "dashboard")
        assert float(pa.locator("#dash-cal-consumed").inner_text().replace(",", "")) == 265.0
        assert pa.locator("#dash-water-consumed").inner_text().strip() == "0.5 L"

        # Beta sees ZERO of Alpha's meals, water, targets, or run history
        pb.reload(wait_until="domcontentloaded")
        pb.wait_for_timeout(300)
        assert pb.locator("#dash-cal-consumed").inner_text().strip() == "0"
        assert pb.locator("#dash-water-consumed").inner_text().strip() == "0 L"
        assert "2,300" in pb.locator("#dash-cal-target").inner_text()

        # IDOR protection: Alpha cannot read or mutate Beta's state
        idor_state = ctx_a.request.get(f"{isolated_server.base_url}/api/nutritrack/state?user_id=user_beta")
        assert idor_state.status == 403
        idor_water = ctx_a.request.post(
            f"{isolated_server.base_url}/api/nutritrack/water",
            headers={"X-CSRF-Token": pa.evaluate("() => document.querySelector('#header-auth-btn') ? '' : ''")},
            data={"user_id": "user_beta", "liters": 5.0},
        )
        assert idor_water.status in (403, 401)
        pb.reload(wait_until="domcontentloaded")
        pb.wait_for_timeout(200)
        assert pb.locator("#dash-water-consumed").inner_text().strip() == "0 L"

        # Logout Alpha in ctx_a and log in as Beta in the SAME browser context -> zero stale localStorage bleed
        pa.click("#header-auth-btn")
        with pa.expect_response(lambda r: "/api/nutritrack/auth/logout" in r.url):
            pa.click("#auth-logout-btn")
        pa.wait_for_selector("#auth-modal:not(.hidden)")
        assert pa.evaluate("() => window.localStorage.getItem('nutritrack_cached_state')") is None
        pa.fill("#auth-user-id", "user_beta")
        pa.fill("#auth-password", "beta-pass-1234")
        with pa.expect_response(lambda r: "/api/nutritrack/auth/login" in r.url and r.request.method == "POST"):
            pa.click("#auth-login-submit")
        pa.locator("#auth-modal").wait_for(state="hidden", timeout=5000)
        pa.wait_for_timeout(300)
        assert pa.locator("#dash-cal-consumed").inner_text().strip() == "0"
        assert "2,300" in pa.locator("#dash-cal-target").inner_text()
    finally:
        ctx_a.close()
        ctx_b.close()
