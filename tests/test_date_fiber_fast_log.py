"""Tests for date extraction fixes, dietary fiber calculations, and fast deterministic meal logging."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from nutritrack.ca1 import food_database
from nutritrack.core.nutrition import resolve_item
from nutritrack.core.quantity import extract_date_from_text, split_meal_text_with_meta
from nutritrack.state import RunLog, StateManager
from nutritrack.tools import analysis_tools, intake_tools, new_context
from nutritrack.tools.fast_log import fast_log_meal, fast_remove_meal
from nutritrack.tools.performance import get_perf_monitor


class DateExtractionAndFiberTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.sm = StateManager(base_dir=self._tmp.name, ca1_log_path=None)
        self.rl = RunLog(path=f"{self._tmp.name}/runs.jsonl")
        get_perf_monitor().reset()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_date_extraction_strips_on_date_and_rejects_impossible_calendar_dates(self) -> None:
        meal_type, date_hint, date_err, phrases = split_meal_text_with_meta(
            "Log 100 g paneer rice for breakfast on 2026-11-16"
        )
        self.assertEqual(meal_type, "breakfast")
        self.assertEqual(date_hint, "2026-11-16")
        self.assertIsNone(date_err)
        self.assertEqual(phrases, ["100 g paneer rice"])

        m2, d2, e2, p2 = split_meal_text_with_meta("On 2026-11-15, I ate 2 boiled eggs for dinner")
        self.assertEqual(m2, "dinner")
        self.assertEqual(d2, "2026-11-15")
        self.assertIsNone(e2)
        self.assertEqual(p2, ["2 boiled eggs"])

        m3, d3, e3, p3 = split_meal_text_with_meta("Log 1 banana on 16/11/2026 for snack")
        self.assertEqual(m3, "snack")
        self.assertEqual(d3, "2026-11-16")
        self.assertIsNone(e3)
        self.assertEqual(p3, ["1 banana"])

        _, d_bad, e_bad, p_bad = split_meal_text_with_meta("Log 2 eggs for breakfast on 2026-02-30")
        self.assertIsNone(d_bad)
        self.assertIsNotNone(e_bad)
        self.assertNotIn("2026-02-30", " ".join(p_bad))

        # Digit foods & fractions must NEVER be mangled as dates
        d_frac, e_frac, c_frac = extract_date_from_text("I had 1/2 cup oats and 1 can 7up")
        self.assertIsNone(d_frac)
        self.assertIsNone(e_frac)
        self.assertIn("1/2 cup oats", c_frac)
        self.assertIn("7up", c_frac)

    def test_dietary_fiber_calculations_preserve_consumed_contract(self) -> None:
        oats = resolve_item("oats", 50, "g")
        self.assertTrue(oats["ok"])
        self.assertEqual(oats["fiber"], 5.0)  # round(10.1 * 0.5, 1) -> 5.0 (IEEE-754 half-to-even)

        banana = resolve_item("banana", 2, "piece")
        self.assertTrue(banana["ok"])
        self.assertEqual(banana["fiber"], 6.2)

        with patch.dict(os.environ, {"NUTRITRACK_CUSTOM_FOODS": f"{self._tmp.name}/custom.json"}):
            food_database.FOOD_DB.pop("chia pudding", None)
            food_database.add_custom_food(
                "chia pudding", 180, 6.0, 14.0, 10.0, serving_size="1 bowl", unit="piece", base_amount=1, fiber=9.5
            )
            custom = resolve_item("chia pudding", 1, "piece")
            self.assertEqual(custom["fiber"], 9.5)
            food_database.FOOD_DB.pop("chia pudding", None)

        ctx = new_context(self.sm, self.rl, task="log fiber meal")
        add_res = intake_tools.add_meal_to_state(
            ctx,
            items=[{"food": "oats", "quantity": 50, "unit": "g"}, {"food": "banana", "quantity": 2, "unit": "piece"}],
            date="2026-11-16",
        )
        self.assertEqual(add_res["status"], "success")
        self.assertEqual(add_res["daily_totals"]["fiber"], 11.2)

        snap = self.sm.get("demo_user").snapshot(day="2026-11-16")
        # Consumed keeps exact 4-key compatibility while exposing top-level fiber
        self.assertEqual(set(snap["consumed"].keys()), {"calories", "protein", "carbs", "fat"})
        self.assertEqual(snap["fiber"], 11.2)

        tot = analysis_tools.calculate_daily_totals(ctx, date="2026-11-16")
        self.assertEqual(tot["fiber"], 11.2)
        trends = analysis_tools.get_weekly_trends(ctx, end_date="2026-11-16", days=7)
        self.assertEqual(trends["averages"]["fiber"], 11.2)

    def test_fast_log_meal_and_remove_meal_with_timing_and_unknown_food_safety(self) -> None:
        res = fast_log_meal(
            self.sm,
            self.rl,
            message="2 boiled eggs and 100g oats for breakfast on 2026-11-16",
            user_id="demo_user",
        )
        self.assertEqual(res["status"], "success")
        self.assertFalse(res["is_llm"])
        self.assertEqual(res["date"], "2026-11-16")
        self.assertEqual(res["state"]["consumed"]["calories"], 526.0)
        self.assertEqual(res["state"]["fiber"], 10.1)
        self.assertIn("total_ms", res["timing_ms"])
        self.assertLess(res["timing_ms"]["total_ms"], 500.0)

        # Unknown food asks user instead of fabricating calories
        unk = fast_log_meal(self.sm, self.rl, message="1 bowl mystic dragon fruit", user_id="demo_user")
        self.assertEqual(unk["status"], "needs_input")
        self.assertEqual(unk["error_code"], "unknown_food")

        # Fast removal works deterministically
        meal_id = res["meal_ids"][0]
        rm = fast_remove_meal(self.sm, self.rl, meal_id=meal_id, user_id="demo_user", date="2026-11-16")
        self.assertEqual(rm["status"], "success")
        self.assertEqual(len(rm["state"]["meals"]), 1)
        self.assertIsNotNone(get_perf_monitor().stats_for("fast_log_meal"))


if __name__ == "__main__":
    unittest.main()
