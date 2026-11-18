"""Browser & API security verification for registration, login, brute-force rate limit, CSRF, and logout."""

from __future__ import annotations

from playwright.sync_api import Browser, Page

from tests.browser.conftest import NutriTrackServer, register_via_modal, send_chat


def test_auth_registration_duplicate_login_ratelimit_and_logout_cleanup(
    edge_browser: Browser, isolated_server: NutriTrackServer
) -> None:
    ctx = edge_browser.new_context(viewport={"width": 1366, "height": 768})
    try:
        page: Page = ctx.new_page()
        http_401s: list[str] = []
        page.on("response", lambda r: http_401s.append(r.url) if r.status == 401 else None)

        page.goto(isolated_server.base_url, wait_until="domcontentloaded")
        register_via_modal(page, "alice_auth", "alice-secret-123")
        assert "Account (alice_auth)" in page.locator("#header-auth-btn").inner_text()

        # Log a meal and confirm cached state exists in localStorage
        send_chat(page, "Log 1 banana")
        page.wait_for_timeout(200)
        assert page.evaluate("() => window.localStorage.getItem('nutritrack_cached_state')") is not None

        # Duplicate registration -> 409 (NOT 500!)
        ctx2 = edge_browser.new_context(viewport={"width": 1366, "height": 768})
        try:
            p2 = ctx2.new_page()
            # Track 401s ONLY on initial boot before sign-in (must be 0!)
            boot_401s: list[str] = []
            p2.on("response", lambda r: boot_401s.append(r.url) if r.status == 401 else None)
            p2.goto(isolated_server.base_url, wait_until="domcontentloaded")
            p2.wait_for_selector("#auth-modal:not(.hidden)")
            p2.wait_for_timeout(300)
            assert boot_401s == [], f"Unexpected 401 calls during unauthenticated boot: {boot_401s}"

            # Protected endpoints return 401 unauthenticated once users exist
            for protected_ep in (
                "/api/nutritrack/state",
                "/api/nutritrack/mode",
                "/api/nutritrack/ai/status",
                "/api/nutritrack/ai/mode",
                "/api/nutritrack/ai/providers",
                "/api/nutritrack/perf",
            ):
                resp = p2.request.get(f"{isolated_server.base_url}{protected_ep}")
                assert resp.status == 401, f"Expected 401 on {protected_ep}, got {resp.status}"

            p2.fill("#auth-user-id", "alice_auth")
            p2.fill("#auth-password", "some-other-password")
            with p2.expect_response(lambda r: "/api/nutritrack/auth/register" in r.url) as dup_resp:
                p2.click("#auth-register-btn")
            assert dup_resp.value.status == 409
            assert dup_resp.value.json()["error_code"] == "user_exists"

            # Rapid wrong passwords trigger 429 rate limit with Retry-After header
            statuses = []
            for _ in range(6):
                with p2.expect_response(lambda r: "/api/nutritrack/auth/login" in r.url) as bad_login:
                    p2.click("#auth-login-submit")
                statuses.append(bad_login.value.status)
            assert statuses[:5] == [401, 401, 401, 401, 401]
            assert statuses[5] == 429

            # Another fresh account is not blocked and valid CSRF works
            p2.fill("#auth-user-id", "bob_auth")
            p2.fill("#auth-password", "bob-secret-456")
            with p2.expect_response(lambda r: "/api/nutritrack/auth/register" in r.url) as bob_reg:
                p2.click("#auth-register-btn")
            assert bob_reg.value.status == 201

            # Missing CSRF on cookie-authenticated POST returns 403
            csrf_fail = ctx2.request.post(
                f"{isolated_server.base_url}/api/nutritrack/water",
                data={"delta_liters": 0.25},
            )
            assert csrf_fail.status == 403
        finally:
            ctx2.close()

        # Logout clears cached localStorage & resets UI to Sign-In prompt cleanly
        page.click("#header-auth-btn")
        page.wait_for_selector("#auth-logout-btn:visible")
        with page.expect_response(lambda r: "/api/nutritrack/auth/logout" in r.url):
            page.click("#auth-logout-btn")
        page.wait_for_timeout(300)
        assert page.evaluate("() => window.localStorage.getItem('nutritrack_cached_state')") is None
        assert page.evaluate("() => window.localStorage.getItem('nutritrack_cached_insights')") is None
        assert page.locator("#auth-modal").is_visible()
    finally:
        ctx.close()
