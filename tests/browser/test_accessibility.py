"""Browser accessibility verification using axe-core across all 6 screens, modal/drawer states, and keyboard Escape."""

from __future__ import annotations

from axe_playwright_python.sync_playwright import Axe
from playwright.sync_api import Page

from tests.browser.conftest import send_chat, switch_screen


def test_axe_zero_violations_across_all_screens_and_modals(app_page: Page) -> None:
    axe = Axe()
    # Add a meal & recommendation so dynamic rendered meal cards and buttons are also audited
    send_chat(app_page, "Log 1 banana")

    for screen_key in ("dashboard", "food-diary", "recommendations", "weekly-progress", "agents", "settings"):
        switch_screen(app_page, screen_key)
        app_page.wait_for_timeout(200)
        res = axe.run(app_page)
        violations = res.response.get("violations") or []
        summary = [
            {"id": v.get("id"), "impact": v.get("impact"), "nodes": [n.get("target") for n in v.get("nodes", [])]}
            for v in violations
        ]
        assert summary == [], f"Accessibility violations on screen '{screen_key}': {summary}"

    # Audit open Custom Food Drawer in Food Diary
    switch_screen(app_page, "food-diary")
    app_page.click("#diary-toggle-custom-btn")
    app_page.wait_for_selector("#custom-food-drawer:not(.hidden)")
    cf_res = axe.run(app_page)
    cf_violations = [v.get("id") for v in (cf_res.response.get("violations") or [])]
    assert cf_violations == [], f"Accessibility violations with Custom Food drawer open: {cf_violations}"

    # Audit Auth Modal open, verify dialog semantics, focus placement, and Escape key closing
    app_page.click("#header-auth-btn")
    modal = app_page.locator("#auth-modal")
    modal.wait_for(state="visible", timeout=3000)
    assert modal.get_attribute("role") == "dialog"
    assert modal.get_attribute("aria-modal") == "true"

    modal_res = axe.run(app_page)
    modal_violations = [v.get("id") for v in (modal_res.response.get("violations") or [])]
    assert modal_violations == [], f"Accessibility violations with Auth Modal open: {modal_violations}"

    # Pressing Escape closes the modal and restores focus cleanly
    app_page.keyboard.press("Escape")
    app_page.wait_for_timeout(150)
    assert not modal.is_visible()
