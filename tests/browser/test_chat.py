"""End-to-end natural-language Chat Interface browser tests in Microsoft Edge."""

from __future__ import annotations

from datetime import date, timedelta

from playwright.sync_api import Page

from tests.browser.conftest import send_chat


def test_chat_simple_multi_food_and_fast_path(app_page: Page) -> None:
    prompts_expected_fast = [
        ("Log 100 g paneer", 265.0),
        ("Add two boiled eggs", 156.0),
        ("Log 200 g chicken breast", 330.0),
        ("Add one banana", 105.0),
        ("Log 250 ml milk", 155.2),
        ("Log three eggs, two bananas, and 250 ml milk", 599.2),
    ]
    running_kcal = 0.0
    for prompt, expected_delta in prompts_expected_fast:
        r = send_chat(app_page, prompt)
        assert r["http_status"] == 200
        assert r["data"]["status"] == "success"
        assert r["data"]["execution_mode"] == "fast_deterministic"
        assert r["data"]["is_llm"] is False
        running_kcal = round(running_kcal + expected_delta, 1)
        assert float(app_page.locator("#dash-cal-consumed").inner_text().replace(",", "")) == running_kcal

    # Unit mismatch ("1/2 cup rice", "1 1/2 cups rice"), unknown food ("7up"), partial ("paneer rice"), no qty ("chicken breast and rice")
    for prompt, err_code in [
        ("Add 1/2 cup rice", "unit_mismatch"),
        ("Log 1 1/2 cups rice", "unit_mismatch"),
        ("Log one can of 7up", "unknown_food"),
        ("Add 100 g paneer rice for breakfast yesterday", "low_confidence_match"),
        ("Log chicken breast and rice in the same meal", "missing_quantity"),
    ]:
        r = send_chat(app_page, prompt)
        assert r["http_status"] == 200
        assert r["data"]["status"] == "needs_input"
        assert r["data"]["error_code"] == err_code
        # User's input MUST remain in the box for easy editing on needs_input!
        assert app_page.locator("#dash-agent-input").input_value() == prompt

    # Confirm plain "chicken" never falsely fuzzy-matches "chickpeas"
    r_chk = send_chat(app_page, "Log chicken and rice in the same meal")
    assert r_chk["data"]["status"] == "needs_input"
    assert "chickpea" not in r_chk["data"]["response"].lower()


def test_chat_date_handling_queries_removals_and_invalids(app_page: Page) -> None:
    today = date.today().isoformat()
    yesterday = (date.today() - timedelta(days=1)).isoformat()

    # Log yesterday and explicit ISO / slash dates at start/end of sentence
    d1 = send_chat(app_page, "Log 1 apple yesterday")
    assert d1["data"]["status"] == "success"
    assert d1["data"]["date"] == yesterday

    d2_date = (date.today() - timedelta(days=2)).isoformat()
    d2 = send_chat(app_page, f"{d2_date} I ate 1 banana")
    assert d2["data"]["status"] == "success"
    assert d2["data"]["date"] == d2_date

    slash_date = (date.today() - timedelta(days=3)).strftime("%d/%m/%Y")
    d3 = send_chat(app_page, f"Log 1 banana on {slash_date}")
    assert d3["data"]["status"] == "success"
    assert d3["data"]["date"] == (date.today() - timedelta(days=3)).isoformat()

    # Invalid & future dates must NOT log meals to today
    bad1 = send_chat(app_page, "Log 1 apple on 2026-02-30")
    assert bad1["data"]["status"] == "needs_input"
    assert bad1["data"]["error_code"] == "invalid_date"

    bad_future = send_chat(app_page, "Log 1 apple on 2099-01-01")
    assert bad_future["data"]["status"] == "failed"
    assert "future" in bad_future["data"]["error_message"].lower()

    # Log something today and test queries + removal
    send_chat(app_page, "Log 100 g paneer")
    for q_prompt, expected_sub in [
        ("Show today's calories", today),
        ("Show today's protein", "18.3 g protein"),
        ("Show remaining calories", "kcal remaining"),
        ("Show my weekly nutrition", "Weekly summary:"),
        ("Compare this week with last week", "Week-over-week comparison"),
        ("Show my meal history", "paneer"),
    ]:
        qr = send_chat(app_page, q_prompt)
        assert qr["http_status"] == 200
        assert qr["data"]["status"] == "success"
        assert expected_sub.lower() in qr["data"]["response"].lower()

    # Negative / zero / missing food / malformed date / too long
    neg = send_chat(app_page, "Log -2 eggs")
    assert neg["data"]["status"] != "success"

    zero = send_chat(app_page, "Log 0 eggs")
    assert zero["data"]["status"] == "needs_input"
    assert zero["data"]["error_code"] == "invalid_quantity"

    nofood = send_chat(app_page, "Log 2")
    assert nofood["data"]["status"] == "needs_input"

    long_res = send_chat(app_page, "a" * 1200)
    assert long_res["http_status"] == 400
    assert long_res["data"]["error_code"] == "input_too_long"

    # Empty / whitespace triggers UI validation without network request
    chat_requests_before = []
    app_page.fill("#dash-agent-input", "   ")
    app_page.click("#dash-agent-submit")
    app_page.wait_for_timeout(150)
    assert "Input Needed" in app_page.locator("#global-toast-title").inner_text()

    # Removals including plural/modifier name ("Remove boiled eggs"), last meal, and non-existent + repeated removal
    send_chat(app_page, "Add two boiled eggs")
    rm_plural = send_chat(app_page, "Remove boiled eggs")
    assert rm_plural["data"]["status"] == "success"
    rm1 = send_chat(app_page, "Remove my last meal")
    assert rm1["data"]["status"] == "success"
    rm_miss = send_chat(app_page, "Remove pizza")
    assert rm_miss["http_status"] == 400
    assert rm_miss["data"]["error_code"] == "not_found"
    rm_miss2 = send_chat(app_page, "Remove pizza")
    assert rm_miss2["http_status"] == 400
