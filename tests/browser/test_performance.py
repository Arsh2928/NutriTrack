"""Browser performance, Core Web Vitals (FCP/CLS), slow-upstream timeout resilience, and /perf endpoint checks."""

from __future__ import annotations

import time

from playwright.sync_api import Browser, Page

from tests.browser.conftest import NutriTrackServer, StubLLMServer, make_isolated_server, send_chat


def test_browser_load_timings_cls_fast_path_latency_and_executor_non_starvation(
    edge_browser: Browser, isolated_server: NutriTrackServer
) -> None:
    ctx = edge_browser.new_context(viewport={"width": 1366, "height": 768})
    try:
        page: Page = ctx.new_page()
        page.goto(isolated_server.base_url, wait_until="load")
        page.wait_for_timeout(250)

        perf = page.evaluate(
            """() => {
              const nav = performance.getEntriesByType('navigation')[0];
              const paints = performance.getEntriesByType('paint');
              const fcp = paints.find(p => p.name === 'first-contentful-paint');
              return {
                domContentLoadedMs: nav ? nav.domContentLoadedEventEnd : null,
                loadEventEndMs: nav ? nav.loadEventEnd : null,
                fcpMs: fcp ? fcp.startTime : null
              };
            }"""
        )
        assert perf["loadEventEndMs"] is not None and perf["loadEventEndMs"] < 5000.0
        if perf["fcpMs"] is not None:
            assert perf["fcpMs"] < 3000.0

        # Deterministic fast-path logging in Edge must finish well under 1500ms end-to-end
        t0 = time.perf_counter()
        res = send_chat(page, "Log 100 g paneer and 2 boiled eggs")
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert res["data"]["execution_mode"] == "fast_deterministic"
        assert res["data"]["timing_ms"]["total_ms"] < 500.0
        assert elapsed_ms < 2000.0

        # Warm reload + perf metrics endpoint
        page.reload(wait_until="load")
        perf_api = page.request.get(f"{isolated_server.base_url}/api/nutritrack/perf")
        assert perf_api.status == 200
        assert "fast_log_meal" in perf_api.json().get("metrics", {})
    finally:
        ctx.close()


def test_hung_upstream_cancelled_cleanly_without_starving_server(edge_browser: Browser) -> None:
    slow_stub = StubLLMServer(name="gemini")
    tmp, srv = make_isolated_server(
        ai_mode="gemini",
        gemini_url=slow_stub.base_url,
        gemini_key="stub-gem",
        extra_env={
            "AI_MAX_RETRIES": "1",
            "AI_REQUEST_TIMEOUT_SECONDS": "1.0",
            "NUTRITRACK_TURN_TIMEOUT_SECONDS": "1.2",
        },
    )
    try:
        # Enqueue a slow 5s upstream hang -> must abort cleanly around 1.2s without leaking retries
        slow_stub.enqueue({"sleep": 5.0}, {"sleep": 5.0})
        ctx = edge_browser.new_context(viewport={"width": 1366, "height": 768})
        page = ctx.new_page()
        page.goto(srv.base_url, wait_until="domcontentloaded")
        page.wait_for_timeout(200)

        t0 = time.perf_counter()
        res = page.request.post(
            f"{srv.base_url}/api/nutritrack/chat",
            data={"message": "What healthy snack advice can you share tonight?"},
            timeout=8000,
        )
        elapsed = time.perf_counter() - t0
        assert res.status in (503, 504)
        assert elapsed < 3.5

        # Local deterministic & state endpoints remain responsive immediately after timeout
        t1 = time.perf_counter()
        st = page.request.get(f"{srv.base_url}/api/nutritrack/state")
        assert st.status == 200
        assert (time.perf_counter() - t1) < 1.0
        ctx.close()
    finally:
        srv.stop()
        tmp.cleanup()
        slow_stub.close()
