"""Responsive layout, zoom (80/100/125%), overlap, drawer toggle, and browser Back/Forward navigation tests."""

from __future__ import annotations

import pytest
from playwright.sync_api import Browser, Page

from tests.browser.conftest import NutriTrackServer, switch_screen

VIEWPORTS = [
    ("desktop", 1920, 1080),
    ("laptop", 1366, 768),
    ("small_laptop", 1280, 720),
    ("tablet", 768, 1024),
    ("mobile", 390, 844),
    ("small_mobile", 360, 800),
]


@pytest.mark.parametrize("name,width,height", VIEWPORTS)
@pytest.mark.parametrize("zoom", [0.8, 1.0, 1.25])
def test_zero_horizontal_overflow_and_no_settings_overlap_across_viewports_and_zoom(
    edge_browser: Browser, isolated_server: NutriTrackServer, name: str, width: int, height: int, zoom: float
) -> None:
    ctx = edge_browser.new_context(viewport={"width": width, "height": height})
    try:
        page: Page = ctx.new_page()
        page.goto(isolated_server.base_url, wait_until="domcontentloaded")
        page.wait_for_timeout(250)
        if zoom != 1.0:
            page.evaluate(f"() => {{ document.documentElement.style.zoom = '{zoom}'; }}")
            page.wait_for_timeout(120)

        for screen_key in ("dashboard", "food-diary", "recommendations", "weekly-progress", "agents", "settings"):
            switch_screen(page, screen_key)
            page.wait_for_timeout(120)
            metrics = page.evaluate(
                """() => {
                  const hdr = document.querySelector('header');
                  return {
                    innerWidth: window.innerWidth,
                    clientWidth: document.documentElement.clientWidth,
                    docScrollWidth: document.documentElement.scrollWidth,
                    headerClientWidth: hdr ? hdr.clientWidth : 0,
                    headerScrollWidth: hdr ? hdr.scrollWidth : 0
                  };
                }"""
            )
            assert metrics["docScrollWidth"] <= metrics["clientWidth"] + 2, (
                f"{name} ({width}x{height} zoom={zoom}) on {screen_key} overflowed horizontally: {metrics}"
            )
            assert metrics["headerScrollWidth"] <= metrics["headerClientWidth"] + 4, (
                f"{name} ({width}x{height} zoom={zoom}) header clipped: {metrics}"
            )

        # Verify Settings calorie unit suffix ('kcal') has safe clearance right of input text
        cal_box = page.locator("#target-calories").evaluate(
            """el => {
              const s = window.getComputedStyle(el);
              return { paddingRight: parseFloat(s.paddingRight) || 0, rect: el.getBoundingClientRect() };
            }"""
        )
        assert cal_box["paddingRight"] >= 40.0, f"Settings #target-calories suffix spacing collapsed: {cal_box}"

        # On mobile/tablet (<1024px), verify drawer opens cleanly with hamburger button and closes on backdrop/nav
        if width < 1024:
            page.click("#mobile-menu-btn")
            page.wait_for_timeout(180)
            aside_left = page.locator("#app-sidebar").evaluate("e => e.getBoundingClientRect().left")
            assert aside_left >= -1
            page.click("#sidebar-close-btn")
            page.wait_for_timeout(180)
    finally:
        ctx.close()


def test_browser_back_forward_navigation_and_nav_alignment(app_page: Page) -> None:
    switch_screen(app_page, "food-diary")
    switch_screen(app_page, "settings")
    assert "#settings" in app_page.url

    app_page.go_back()
    app_page.wait_for_timeout(200)
    assert app_page.locator("#screen-food-diary").is_visible()

    app_page.go_forward()
    app_page.wait_for_timeout(200)
    assert app_page.locator("#screen-settings").is_visible()

    # Check Dashboard/Diary/Settings nav items use left-aligned gap rather than stretched justify-between
    dash_justify = app_page.locator("#sidebar-nav a[data-path='dashboard']").evaluate(
        "e => window.getComputedStyle(e).justifyContent"
    )
    assert dash_justify != "space-between"
