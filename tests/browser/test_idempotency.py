"""End-to-end idempotency, concurrency, retry, conflict, and corrupt-store safety tests in Edge & Flask."""

from __future__ import annotations

import json

from playwright.sync_api import Page

from tests.browser.conftest import NutriTrackServer, switch_screen


def test_browser_double_submit_and_retry_idempotency_across_endpoints(
    app_page: Page, isolated_server: NutriTrackServer
) -> None:
    base = isolated_server.base_url

    # Scenario A: Double rapid click on Diary submit & Water button produces exactly ONE entry
    switch_screen(app_page, "food-diary")
    app_page.fill("#diary-food-name", "apple")
    app_page.fill("#diary-qty-value", "1")
    app_page.select_option("#diary-qty-unit", "piece")
    app_page.click("#diary-submit-btn")
    app_page.evaluate("() => document.querySelector('#diary-quick-add-form').requestSubmit()")
    app_page.wait_for_timeout(400)
    assert app_page.locator("#diary-cal-val").inner_text().strip() == "95"

    # Scenario B: Chat & Water retry with same request_id returns cached replay without duplicate logging
    r1 = app_page.request.post(
        f"{base}/api/nutritrack/chat",
        data={"message": "Log 1 banana", "request_id": "browser-idem-chat-1"},
    )
    assert r1.status == 200
    r2 = app_page.request.post(
        f"{base}/api/nutritrack/chat",
        data={"message": "Log 1 banana", "request_id": "browser-idem-chat-1"},
    )
    assert r2.status == 200
    assert r2.json().get("duplicate") is True

    # Scenario C: Different payload with same request_id returns 409 idempotency_conflict
    conflict = app_page.request.post(
        f"{base}/api/nutritrack/chat",
        data={"message": "Log 2 bananas", "request_id": "browser-idem-chat-1"},
    )
    assert conflict.status == 409
    assert conflict.json()["error_code"] == "idempotency_conflict"

    # Scenario D: Water request_id replay vs conflict
    w1 = app_page.request.post(
        f"{base}/api/nutritrack/water",
        data={"delta_liters": 0.25, "request_id": "browser-idem-water-1"},
    )
    assert w1.status == 200
    w2 = app_page.request.post(
        f"{base}/api/nutritrack/water",
        data={"delta_liters": 0.25, "request_id": "browser-idem-water-1"},
    )
    assert w2.status == 200
    assert w2.json().get("duplicate") is True
    assert w2.json()["water_today_liters"] == 0.25

    w_conf = app_page.request.post(
        f"{base}/api/nutritrack/water",
        data={"delta_liters": 0.5, "request_id": "browser-idem-water-1"},
    )
    assert w_conf.status == 409

    # Scenario E: Restart preserves idempotency history
    isolated_server.restart()
    r3 = app_page.request.post(
        f"{base}/api/nutritrack/chat",
        data={"message": "Log 1 banana", "request_id": "browser-idem-chat-1"},
    )
    assert r3.status == 200
    assert r3.json().get("duplicate") is True

    # Confirm exact single apple (95) + single banana (105) = 200 kcal
    st_resp = app_page.request.get(f"{base}/api/nutritrack/state").json()
    assert st_resp["state"]["consumed"]["calories"] == 200.0

    # Scenario F: Corrupt operations.json fails cleanly with 500 operation_store_corrupt & does NOT silently wipe file
    ops_path = isolated_server.root_dir / "data" / "operations" / "demo_user.json"
    assert ops_path.exists()
    ops_path.write_text("{CORRUPT_JSON!!!", encoding="utf-8")
    corrupt_res = app_page.request.post(
        f"{base}/api/nutritrack/chat",
        data={"message": "Log 1 apple", "request_id": "after-corrupt-1"},
    )
    assert corrupt_res.status == 500
    assert corrupt_res.json()["error_code"] == "operation_store_corrupt"
    assert ops_path.read_text(encoding="utf-8") == "{CORRUPT_JSON!!!"
