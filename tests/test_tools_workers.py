"""Phase 3 tests: the tools of the four worker agents, plus the five scenarios from the CA2 brief."""
import json
import unittest
from unittest import mock

from nutritrack.ca1.food_database import FOOD_DB
from nutritrack.core.nutrition import resolve_item
from nutritrack.state import NutritionState
from tests.support import ToolCase, make_entry


def item(food, quantity=None, unit=None):
    return {"food": food, "quantity": quantity, "unit": unit}


# ===========================================================================
class TestBriefScenarios(ToolCase):
    """The five scenarios required by the CA2 brief, at the level of the Python tools."""

    def test_1_meal_logging(self):
        """TEST 1: 'Add 2 boiled eggs and 1 banana.'"""
        parsed = self.call("parse_meal_entry", "Add 2 boiled eggs and 1 banana")
        self.assertEqual([(i["food"], i["quantity"]) for i in parsed["items"]], [("boiled eggs", 2.0), ("banana", 1.0)])   # extraction
        look = self.call("lookup_food_nutrition", "boiled eggs", 2)
        self.assertEqual((look["food"], look["calories"], look["protein"]), ("egg", 156.0, 12.0))                          # nutrition lookup
        added = self.call("add_meal_to_state", [item("boiled eggs", 2), item("banana", 1)], meal_type="breakfast")
        self.assertEqual((added["status"], added["code"]), ("success", "ok"))                                              # logging
        meals = self.state.get_day_meals()
        self.assertEqual([(m["food"], m["quantity_str"], m["calories"]) for m in meals],
                         [("egg", "2 pieces", 156.0), ("banana", "1 piece", 105.0)])
        snap = self.state.snapshot()                                                                                       # shared state updated
        self.assertEqual(snap["consumed"], {"calories": 261.0, "protein": 13.3, "carbs": 28.2, "fat": 11.0})
        self.assertEqual(snap["remaining"]["calories"], 2039.0)
        self.assertEqual({e["tool"] for e in self.logged(status="success")},
                         {"parse_meal_entry", "lookup_food_nutrition", "add_meal_to_state"})

    def test_2_calorie_budget(self):
        """TEST 2: a meal that exceeds the remaining calorie target."""
        self.call("set_calorie_target", 2000)
        self.seed(0, 1750, protein=105)
        left = self.call("calculate_remaining_targets")
        self.assertEqual(left["remaining"]["calories"], 250.0)                                                             # remaining calories
        check = self.call("check_meal_against_budget", [item("bread", 2, "slice"), item("banana", 1)])
        self.assertEqual((check["fits"], check["code"]), (False, "exceeds_budget"))                                        # identified as exceeding
        self.assertEqual((check["meal"]["calories"], check["surplus_deficit"]), (263.0, -13.0))
        self.assertEqual(check["remaining_before"]["calories"], 250.0)
        self.assertIn("Does NOT fit", check["message"])
        self.assertEqual(self.state.day_totals()["calories"], 1750.0, "checking a meal must not log it")

    def test_3_meal_recommendation(self):
        """TEST 3: 'Recommend a meal with 500 calories remaining.'"""
        cands = self.call("get_recommendation_candidates", remaining_calories=500)
        self.assertGreater(cands["count"], 0)                                                                              # candidates retrieved
        est = self.call("estimate_meal_nutrition")
        self.assertEqual(len(est["estimates"]), cands["count"])                                                            # nutrition estimated
        for e in est["estimates"]:
            self.assertLessEqual(e["calories"], 500.0)
        ranked = self.call("rank_meal_candidates")
        scores = [r["score"] for r in ranked["ranked"]]
        self.assertEqual(scores, sorted(scores, reverse=True))                                                             # ranked
        best = ranked["ranked"][0]
        self.assertIn("500 kcal", best["reason"])                                                                          # justified
        self.assertIn(f"{best['protein']:g} g protein", best["reason"])
        valid = self.call("validate_recommendation_fit", ranked["best_id"])
        self.assertTrue(valid["valid"])
        self.assertEqual([r["id"] for r in self.state.snapshot()["recommendations"]], [ranked["best_id"]])
        self.assertEqual({e["tool"] for e in self.logged()},
                         {"get_recommendation_candidates", "estimate_meal_nutrition", "rank_meal_candidates",
                          "validate_recommendation_fit"})

    def test_4_weekly_analytics(self):
        """TEST 4: weekly report from SAVED records, with missing dates handled."""
        # target 2300 / 150. today partial, -2 and -4 have no records at all.
        for ago, kcal, protein in ((0, 1500, 90), (1, 2100, 140), (3, 2400, 120), (5, 900, 40), (6, 2200, 155)):
            self.seed(ago, kcal, protein)
        res = self.call("get_weekly_trends")
        self.assertEqual((res["status"], res["days_with_data"]), ("success", 5))
        self.assertEqual(res["days_without_data"], [self.clock.days_ago(4), self.clock.days_ago(2)])                       # missing dates named
        self.assertEqual(res["averages"]["calories"], round((1500 + 2100 + 2400 + 900 + 2200) / 5, 1))                    # over the 5 real days
        self.assertIn("5 of 7", res["data_period_note"])
        self.assertIn("NOT counted as zero", res["data_period_note"])
        gaps = {r["date"]: r["calories"] for r in res["daily"]}
        self.assertIsNone(gaps[self.clock.days_ago(2)])                                                                    # never zero-filled
        self.assertEqual(res["calorie_target_summary"]["days_within_target"], 4)
        self.assertEqual(res["calorie_target_summary"]["days_over_target"], 1)
        self.assertEqual(res["protein_target_summary"]["days_met"], 1)
        self.assertEqual([u["date"] for u in res["unusual_days"]], [self.clock.days_ago(5)])                               # 900 kcal = low
        self.assertEqual(res["unusual_days"][0]["flag"], "low")
        # the report is built from what is actually saved: identical after a restart
        self.restart()
        again = self.call("get_weekly_trends", ctx=self.new_ctx())
        self.assertEqual(again["averages"], res["averages"])

    def test_5_invalid_food(self):
        """TEST 5: a food that is not in the database."""
        res = self.call("add_meal_to_state", [item("idli", 2)])
        self.assertEqual((res["status"], res["code"]), ("failed", "unknown_food"))
        self.assertTrue(res["needs_user_input"])
        self.assertIn("calories", res["question"])                                                                          # user is asked
        self.assertNotIn("calories", {k for k in res if k != "question"} & {"calories", "protein", "carbs", "fat"})        # nothing invented
        self.assertEqual(self.state.get_day_meals(), [])                                                                    # nothing saved
        look = self.call("lookup_food_nutrition", "idli", 2)
        self.assertEqual(look["code"], "unknown_food")
        failed = self.logged(status="failed")                                                                               # failed lookup recorded
        self.assertEqual([(e["tool"], e["error"]["code"]) for e in failed],
                         [("add_meal_to_state", "unknown_food"), ("lookup_food_nutrition", "unknown_food")])
        # ... and once the user supplies the numbers, the same food works
        saved = self.call("register_custom_food", "idli", 39, 2, 8, 0.4, "1 idli (~40 g)")
        self.forget_custom_food("idli")
        self.assertEqual(saved["code"], "saved")
        self.assertEqual(self.call("lookup_food_nutrition", "idli", 2)["calories"], 78.0)


# ===========================================================================
class TestIntakeTools(ToolCase):

    def test_parse_the_example_sentence_from_the_brief(self):
        res = self.call("parse_meal_entry", "I ate 3 boiled eggs and 2 bananas.")
        self.assertEqual([(i["food"], i["quantity"], i["unit"]) for i in res["items"]],
                         [("boiled eggs", 3.0, None), ("bananas", 2.0, None)])
        self.assertFalse(res["has_problems"])

    def test_parse_reads_meal_type_date_and_separates_non_food_requests(self):
        res = self.call("parse_meal_entry", "Add my breakfast: two roti + 100 g curd, yesterday")
        self.assertEqual((res["meal_type"], res["date_hint"]), ("breakfast", "yesterday"))
        self.assertEqual([(i["food"], i["quantity"], i["unit"]) for i in res["items"]], [("roti", 2.0, None), ("curd", 100.0, "g")])
        mixed = self.call("parse_meal_entry", "I had 2 eggs and calculate how much protein I have consumed")
        self.assertEqual([i["food"] for i in mixed["items"]], ["eggs"])
        self.assertEqual(len(mixed["ignored_phrases"]), 1)

    def test_parse_reports_problems_instead_of_hiding_them(self):
        res = self.call("parse_meal_entry", "0 eggs and -2 bananas")
        self.assertTrue(res["has_problems"])
        self.assertEqual({p["code"] for p in res["problems"]}, {"invalid_quantity"})

    def test_parse_failures_ask_the_user(self):
        for text, code in (("Add my breakfast", "no_food_found"), ("", "empty_input"), ("   ", "empty_input"),
                           ("x" * 1001, "input_too_long")):
            with self.subTest(text=text[:20]):
                res = self.call("parse_meal_entry", text)
                self.assertEqual((res["status"], res["code"], res["needs_user_input"]), ("failed", code, True))
        self.assertEqual(self.call("parse_meal_entry", None)["code"], "empty_input")

    def test_lookup_matches_the_shared_engine_exactly(self):
        for food, qty, unit in (("egg", 3, None), ("oats", 40, "g"), ("milk", 250, "ml"), ("bread", 2, "slice"), ("soya chunks", 100, "g")):
            with self.subTest(food=food):
                res, direct = self.call("lookup_food_nutrition", food, qty, unit), resolve_item(food, qty, unit)
                self.assertEqual({k: res[k] for k in ("calories", "protein", "carbs", "fat")},
                                 {k: direct[k] for k in ("calories", "protein", "carbs", "fat")})
                self.assertTrue(res["estimated"])

    def test_lookup_failures_carry_a_question_and_never_numbers(self):
        cases = [(("pizza", 1, None), "unknown_food"), (("banana", 200, "g"), "unit_mismatch"),
                 (("curd", 2, None), "ambiguous_quantity"), (("curd", None, None), "missing_quantity"),
                 (("pintola high protein muesli", 1, None), "low_confidence_match"), (("egg", 0, None), "invalid_quantity"),
                 (("egg", 999, None), "implausible_quantity"), (("milk", 1, "glass"), "unsupported_unit"),
                 (("", 1, None), "no_food_name")]
        for args, code in cases:
            with self.subTest(args=args):
                res = self.call("lookup_food_nutrition", *args)
                self.assertEqual((res["status"], res["code"], res["needs_user_input"]), ("failed", code, True))
                for key in ("calories", "protein", "carbs", "fat"):
                    self.assertNotIn(key, res)
        self.assertTrue(self.call("lookup_food_nutrition", "milk", 1, None, "glass")["needs_user_input"])

    def test_add_ignores_any_nutrition_numbers_supplied_by_the_caller(self):
        """The language model must not be able to change calories on the way into the state."""
        self.call("add_meal_to_state", [{"food": "egg", "quantity": 3, "calories": 1, "protein": 999, "quantity_str": "x"}])
        meal = self.state.get_day_meals()[0]
        self.assertEqual((meal["calories"], meal["protein"]), (234.0, 18.0))

    def test_partial_success_saves_what_it_can_and_reports_the_rest(self):
        res = self.call("add_meal_to_state", [item("egg", 2), item("idli", 2)])
        self.assertEqual((res["status"], res["code"], res["needs_user_input"]), ("success", "partial", True))
        self.assertEqual([u["food"] for u in res["unresolved"]], ["idli"])
        self.assertEqual([m["food"] for m in self.state.get_day_meals()], ["egg"])
        self.assertIn("idli", res["message"])

    def test_nothing_saved_when_nothing_resolves(self):
        res = self.call("add_meal_to_state", [item("idli", 2), item("samosa", 1)])
        self.assertEqual((res["status"], res["code"]), ("failed", "no_items_resolved"))
        self.assertEqual(len(res["unresolved"]), 2)
        self.assertEqual(self.state.get_day_meals(), [])

    def test_unit_mismatch_saves_nothing(self):
        res = self.call("add_meal_to_state", [item("banana", 200, "g")])
        self.assertEqual(res["code"], "unit_mismatch")
        self.assertEqual(self.state.get_day_meals(), [])

    def test_the_same_items_in_the_same_run_are_logged_once(self):
        first = self.call("add_meal_to_state", [item("egg", 3)])
        again = self.call("add_meal_to_state", [item("egg", 3)])
        self.assertEqual((first["code"], again["code"]), ("ok", "duplicate_ignored"))
        self.assertEqual(again["added"], [])
        self.assertEqual(len(self.state.get_day_meals()), 1)
        self.assertEqual(again["meal_ids"], first["meal_ids"])

    def test_eating_the_same_food_in_a_different_run_is_logged_again(self):
        self.call("add_meal_to_state", [item("egg", 3)])
        self.call("add_meal_to_state", [item("egg", 3)], ctx=self.new_ctx())
        self.assertEqual(len(self.state.get_day_meals()), 2)

    def test_an_explicit_request_id_deduplicates_across_runs(self):
        self.call("add_meal_to_state", [item("egg", 3)], request_id="ui-click-77")
        again = self.call("add_meal_to_state", [item("egg", 3)], request_id="ui-click-77", ctx=self.new_ctx())
        self.assertEqual(again["code"], "duplicate_ignored")
        self.assertEqual(len(self.state.get_day_meals()), 1)

    def test_date_and_meal_type_are_validated(self):
        ok_ = self.call("add_meal_to_state", [item("egg", 1)], date="yesterday", meal_type="dinner")
        self.assertEqual(ok_["date"], self.clock.days_ago(1))
        self.assertEqual(self.state.get_day_meals(), [])
        for kwargs in ({"date": "2099-01-01"}, {"date": "garbage"}, {"meal_type": "elevenses"}):
            with self.subTest(kwargs=kwargs):
                res = self.call("add_meal_to_state", [item("egg", 1)], **kwargs)
                self.assertEqual((res["status"], res["code"]), ("failed", "invalid_input"))

    def test_malformed_item_lists_are_rejected_not_crashed_on(self):
        for bad in ("egg", [], [{"quantity": 2}], [{"food": ""}], [{"food": "egg", "quantity": "two"}],
                    [{"food": "egg", "quantity": True}], [{"food": "egg", "unit": 5}], ["egg"], [item("egg")] * 21):
            with self.subTest(bad=str(bad)[:30]):
                res = self.call("add_meal_to_state", bad)
                self.assertEqual((res["status"], res["code"]), ("failed", "invalid_input"))
        self.assertEqual(self.state.get_day_meals(), [])

    def test_remove_by_name_id_and_batch(self):
        added = self.call("add_meal_to_state", [item("egg", 2), item("banana", 1)])
        gone = self.call("remove_meal_from_state", food_name="banana")
        self.assertEqual((gone["status"], gone["daily_totals"]["calories"]), ("success", 156.0))
        by_id = self.call("remove_meal_from_state", meal_id=added["meal_ids"][0])
        self.assertEqual(by_id["daily_totals"]["meal_count"], 0)
        self.call("add_meal_to_state", [item("egg", 1), item("roti", 1)], ctx=self.new_ctx())
        batch = self.state.get_day_meals()[0]["batch_id"]
        self.assertEqual(len(self.call("remove_meal_from_state", batch_id=batch)["removed"]), 2)

    def test_remove_failures_ask_the_user(self):
        self.call("add_meal_to_state", [item("egg", 2)])
        nope = self.call("remove_meal_from_state", food_name="pizza")
        self.assertEqual((nope["status"], nope["code"], nope["needs_user_input"]), ("failed", "not_found", True))
        both = self.call("remove_meal_from_state", food_name="egg", meal_id="m-1")
        self.assertEqual((both["status"], both["code"]), ("failed", "invalid_input"))
        self.assertEqual(self.call("remove_meal_from_state")["code"], "invalid_input")
        self.assertEqual(len(self.state.get_day_meals()), 1)

    def test_remove_reports_other_matches_it_kept(self):
        self.call("add_meal_to_state", [item("egg", 1)])
        self.call("add_meal_to_state", [item("egg", 2)], ctx=self.new_ctx())
        res = self.call("remove_meal_from_state", food_name="egg")
        self.assertEqual(res["remaining_matches"], 1)
        self.assertIn("1 other matching entry kept", res["message"])

    def test_register_custom_food_validation(self):
        bad = [dict(name="", calories=1, protein=1, carbs=1, fat=1), dict(name="123", calories=1, protein=1, carbs=1, fat=1),
               dict(name="x", calories=-1, protein=1, carbs=1, fat=1), dict(name="x", calories=1, protein=float("nan"), carbs=1, fat=1),
               dict(name="x", calories=1, protein=1, carbs=1, fat=1, unit="bowl"),
               dict(name="x", calories=1, protein=1, carbs=1, fat=1, base_amount=0),
               dict(name="x", calories=1, protein=1, carbs=1, fat=1, base_amount=6000),
               dict(name="x", calories=1500, protein=1, carbs=1, fat=1, unit="g", base_amount=100),     # >10 kcal per gram
               dict(name="x", calories=100, protein=150, carbs=1, fat=1, unit="g", base_amount=100)]    # more protein than food
        for kwargs in bad:
            with self.subTest(kwargs=kwargs):
                res = self.call("register_custom_food", **kwargs)
                self.assertEqual(res["status"], "failed")
                self.assertTrue(res["needs_user_input"])
        self.assertNotIn("x", FOOD_DB)

    def test_register_custom_food_will_not_silently_replace_a_builtin_or_existing_food(self):
        self.assertEqual(self.call("register_custom_food", "Egg", 1, 1, 1, 1)["code"], "food_exists")
        self.assertEqual(FOOD_DB["egg"]["calories_per_unit"], 78)
        self.call("register_custom_food", "mom's rajma", 320, 14, 45, 8)
        self.forget_custom_food("mom's rajma")
        self.assertEqual(self.call("register_custom_food", "mom's rajma", 300, 14, 45, 8)["code"], "food_exists")
        replaced = self.call("register_custom_food", "mom's rajma", 300, 14, 45, 8, replace=True)
        self.assertEqual(replaced["code"], "saved")
        self.assertEqual(self.call("lookup_food_nutrition", "mom's rajma", 1)["calories"], 300.0)

    def test_a_custom_food_is_saved_to_disk_exactly_as_the_user_typed_it(self):
        import os
        self.call("register_custom_food", "ragi dosa", 120, 3.5, 22, 2, "1 dosa", "piece", 1)
        self.forget_custom_food("ragi dosa")
        with open(os.environ["NUTRITRACK_CUSTOM_FOODS"], encoding="utf-8") as fh:
            entry = json.load(fh)["ragi dosa"]
        self.assertEqual((entry["calories_per_unit"], entry["protein"], entry["carbs"], entry["fat"]), (120.0, 3.5, 22.0, 2.0))
        self.assertEqual(self.call("add_meal_to_state", [item("ragi dosa", 2)])["daily_totals"]["calories"], 240.0)


# ===========================================================================
class TestAnalysisTools(ToolCase):

    def test_daily_totals_with_and_without_data(self):
        empty = self.call("calculate_daily_totals")
        self.assertEqual((empty["code"], empty["has_data"], empty["consumed"]["calories"]), ("no_data", False, 0.0))
        self.assertIn("nothing was recorded", empty["message"])
        self.call("add_meal_to_state", [item("egg", 3), item("banana", 2)])
        res = self.call("calculate_daily_totals")
        self.assertEqual((res["code"], res["meal_count"], res["consumed"]["calories"]), ("ok", 2, 444.0))
        self.assertEqual(res["targets"], {"calories": 2300.0, "protein": 150.0})
        self.assertEqual(self.call("calculate_daily_totals", "yesterday")["code"], "no_data")

    def test_macro_split_formula_4_4_9(self):
        self.call("add_meal_to_state", [item("chicken breast", 200, "g")])            # P 62, C 0, F 7.2
        res = self.call("calculate_macro_split")
        kcal = 62.0 * 4 + 7.2 * 9
        self.assertEqual(res["macro_kcal"], round(kcal, 1))
        self.assertEqual(res["percent"], {"protein": round(248 / kcal * 100, 1), "carbs": 0.0, "fat": round(64.8 / kcal * 100, 1)})
        self.assertEqual(res["logged_kcal"], 330.0)
        self.assertEqual(self.call("calculate_macro_split", "yesterday")["code"], "no_data")

    def test_weekly_with_no_records_says_so_instead_of_inventing_numbers(self):
        res = self.call("get_weekly_trends")
        self.assertEqual((res["status"], res["code"], res["days_with_data"], res["averages"]), ("success", "no_data", 0, None))
        self.assertIn("0 of 7", res["data_period_note"])
        self.assertTrue(all(r["calories"] is None for r in res["daily"]))

    def test_weekly_with_fewer_than_seven_days_states_the_available_period(self):
        self.seed(2, 2000, 100); self.seed(1, 2200, 120)
        res = self.call("get_weekly_trends")
        self.assertEqual(res["days_with_data"], 2)
        self.assertIn(f"2 of 7 days that have saved records ({self.clock.days_ago(2)} to {self.clock.days_ago(1)})", res["data_period_note"])
        self.assertEqual(len(res["days_without_data"]), 5)
        self.assertEqual(res["averages"]["calories"], 2100.0)

    def test_weekly_custom_range_and_include_today(self):
        for ago in range(0, 10):
            self.seed(ago, 2000 + ago)
        custom = self.call("get_weekly_trends", start_date=self.clock.days_ago(9), end_date=self.clock.days_ago(5))
        self.assertEqual((custom["period"]["days_in_range"], custom["days_with_data"]), (5, 5))
        self.assertEqual(custom["averages"]["calories"], 2007.0)
        no_today = self.call("get_weekly_trends", include_today=False)
        self.assertNotIn(self.clock.today, [r["date"] for r in no_today["daily"]])
        with_today = self.call("get_weekly_trends")
        self.assertIn("Today", with_today["data_period_note"])
        self.assertTrue(with_today["daily"][-1]["is_today"])
        self.assertEqual(len(self.call("get_weekly_trends", days=30)["daily"]), 30)

    def test_weekly_rejects_bad_ranges(self):
        for kwargs in ({"days": 0}, {"days": 93}, {"days": True}, {"days": "7"}, {"end_date": "2099-01-01"},
                       {"start_date": self.clock.today, "end_date": self.clock.days_ago(3)},
                       {"start_date": self.clock.days_ago(200), "end_date": self.clock.today}, {"end_date": "garbage"}):
            with self.subTest(kwargs=kwargs):
                res = self.call("get_weekly_trends", **kwargs)
                self.assertEqual((res["status"], res["code"]), ("failed", "invalid_input"))

    def test_unusual_high_days_and_todays_partial_total_is_never_flagged(self):
        self.seed(1, 3500, 100); self.seed(2, 2000, 100); self.seed(0, 100, 5)         # today is tiny but still in progress
        res = self.call("get_weekly_trends")
        self.assertEqual([(u["date"], u["flag"]) for u in res["unusual_days"]], [(self.clock.days_ago(1), "high")])

    def test_unusual_days_fall_back_to_the_users_own_median_without_targets(self):
        alice = self.new_ctx(user_id="alice")
        for ago, kcal in ((1, 1800), (2, 2000), (3, 2200), (4, 5000)):
            self.manager.get("alice").add_meals([make_entry(calories=float(kcal))], day=self.clock.days_ago(ago), request_id=f"a{ago}")
        res = self.call("get_weekly_trends", ctx=alice)
        self.assertEqual([(u["date"], u["flag"]) for u in res["unusual_days"]], [(self.clock.days_ago(4), "high")])
        self.assertIsNone(res["calorie_target_summary"]["days_evaluated"] or None)       # no targets -> nothing judged
        self.assertEqual(res["averages"]["calories"], 2750.0)

    def test_weekly_uses_the_targets_that_applied_on_each_day(self):
        self.seed(1, 2000)                     # saved while the target was 2300
        self.call("set_calorie_target", 1800)
        self.seed(0, 1900)
        res = {r["date"]: r for r in self.call("get_weekly_trends")["daily"]}
        self.assertEqual((res[self.clock.days_ago(1)]["within_calorie_target"], res[self.clock.today]["within_calorie_target"]), (True, False))

    def test_compare_intake_with_targets(self):
        self.seed(0, 2400, 100)
        res = self.call("compare_intake_with_targets")
        self.assertEqual((res["calories"]["status"], res["calories"]["difference"], res["calories"]["percent_of_target"]), ("over_target", 100.0, 104.3))
        self.assertEqual((res["protein"]["status"], res["protein"]["percent_of_target"]), ("below_target", 66.7))
        self.seed(0, 100, 60)
        self.assertEqual(self.call("compare_intake_with_targets")["protein"]["status"], "met")

    def test_compare_handles_missing_targets_and_empty_days(self):
        alice = self.new_ctx(user_id="alice")
        none_set = self.call("compare_intake_with_targets", ctx=alice)
        self.assertEqual((none_set["status"], none_set["code"], none_set["needs_user_input"]), ("failed", "no_target", True))
        self.manager.get("alice").set_target("calories", 2000)
        partial = self.call("compare_intake_with_targets", ctx=alice)
        self.assertEqual((partial["status"], partial["missing_targets"], partial["protein"]["status"]), ("success", ["protein"], "no_target"))
        self.assertTrue(partial["needs_user_input"])
        empty = self.call("compare_intake_with_targets")
        self.assertFalse(empty["has_data"])
        self.assertIn("Nothing is logged", empty["message"])


# ===========================================================================
class TestBudgetTools(ToolCase):

    def test_setting_targets_updates_state_and_remaining(self):
        res = self.call("set_calorie_target", 2000)
        self.assertEqual((res["target"], res["previous"], res["previous_source"]), (2000.0, 2300.0, "demo_default"))
        self.assertIn("replacing a demonstration default", res["message"])
        self.assertEqual(self.state.targets_source()["calories"], "user")
        self.assertEqual(self.call("set_protein_target", 120)["remaining"], {"calories": 2000.0, "protein": 120.0})
        self.assertNotIn("demonstration", self.call("set_calorie_target", 2100)["message"])

    def test_invalid_targets_are_rejected_and_change_nothing(self):
        for tool, bad in (("set_calorie_target", (50, 99999, -5, "lots", None, float("inf"), True)),
                          ("set_protein_target", (1, 9999, -1, "lots", None, float("nan"), False))):
            for value in bad:
                with self.subTest(tool=tool, value=value):
                    res = self.call(tool, value)
                    self.assertEqual((res["status"], res["code"], res["needs_user_input"]), ("failed", "invalid_input", True))
        self.assertEqual(self.state.targets_for(), {"calories": 2300.0, "protein": 150.0})

    def test_remaining_targets_matches_the_example_in_the_brief(self):
        self.seed(0, 1750, 105)                                    # targets 2300 kcal / 150 g
        res = self.call("calculate_remaining_targets")
        self.assertEqual(res["remaining"], {"calories": 550.0, "protein": 45.0})
        self.assertEqual((res["consumed"]["calories"], res["consumed"]["protein"]), (1750.0, 105.0))
        self.assertIn("550 kcal left", res["message"])
        self.assertEqual(res["calories_over_by"], 0.0)

    def test_remaining_is_negative_when_over_target(self):
        self.seed(0, 2500, 160)
        res = self.call("calculate_remaining_targets")
        self.assertEqual(res["remaining"], {"calories": -200.0, "protein": -10.0})
        self.assertEqual(res["calories_over_by"], 200.0)
        self.assertIn("200 kcal OVER", res["message"])

    def test_remaining_without_targets_asks_and_partial_targets_are_reported(self):
        alice = self.new_ctx(user_id="alice")
        none_set = self.call("calculate_remaining_targets", ctx=alice)
        self.assertEqual((none_set["status"], none_set["code"], none_set["needs_user_input"]), ("failed", "no_target", True))
        self.manager.get("alice").set_target("protein", 100)
        partial = self.call("calculate_remaining_targets", ctx=alice)
        self.assertEqual((partial["remaining"], partial["missing_targets"]), ({"calories": None, "protein": 100.0}, ["calories"]))

    def test_check_meal_fits_exactly_exceeds_and_never_logs(self):
        self.call("set_calorie_target", 1000)
        fits = self.call("check_meal_against_budget", [item("egg", 3)])
        self.assertEqual((fits["fits"], fits["surplus_deficit"], fits["code"]), (True, 766.0, "fits_budget"))
        exact = self.call("check_meal_against_budget", [item("banana", 1), item("egg", 1)])
        self.call("set_calorie_target", 500)
        self.assertEqual(exact["meal"]["calories"], 183.0)
        boundary = self.call("check_meal_against_budget", [item("bread", 2, "slice"), item("curd", 100, "g"), item("egg", 1)])  # 158+98+78=334 vs 500
        self.assertTrue(boundary["fits"])
        self.seed(0, 395)                                                            # 105 left; banana is exactly 105
        edge = self.call("check_meal_against_budget", [item("banana", 1)])
        self.assertEqual((edge["fits"], edge["surplus_deficit"]), (True, 0.0))
        self.seed(0, 0.1)
        self.assertFalse(self.call("check_meal_against_budget", [item("banana", 1)])["fits"])
        self.assertEqual(self.state.day_totals()["meal_count"], 2, "no meal may be logged by a check")

    def test_check_meal_replays_the_recorded_ca1_decisions(self):
        """CA1's notebook: 2 roti + 100 g curd logged, 94 kcal left -> banana is 11 kcal too much.
        (CA1 used a 400 kcal budget; CA2 enforces a 500 kcal minimum target, so the same 94 kcal
        remainder is reached with a 500 kcal target and 100 kcal of other food.)"""
        self.call("set_calorie_target", 500)
        self.call("add_meal_to_state", [item("roti", 2), item("curd", 100, "g")])      # 208 + 98 = 306
        self.seed(0, 100)
        self.assertEqual(self.call("calculate_remaining_targets")["remaining"]["calories"], 94.0)
        self.assertEqual(self.call("check_meal_against_budget", [item("banana", 1)])["surplus_deficit"], -11.0)
        self.assertEqual(self.call("check_meal_against_budget", [item("soya chunks", 100, "g")])["surplus_deficit"], -251.0)

    def test_check_meal_protein_after(self):
        self.seed(0, 1000, 100)
        res = self.call("check_meal_against_budget", [item("chicken breast", 100, "g")])
        self.assertEqual((res["remaining_before"]["protein"], res["protein_remaining_after"]), (50.0, 19.0))

    def test_check_meal_unknown_foods_and_missing_target_ask_instead_of_guessing(self):
        res = self.call("check_meal_against_budget", [item("egg", 2), item("idli", 2)])
        self.assertEqual((res["status"], res["code"], res["needs_user_input"]), ("failed", "unknown_food", True))
        self.assertIn("idli", res["message"])
        multi = self.call("check_meal_against_budget", [item("idli", 1), item("samosa", 1)])
        self.assertEqual(multi["code"], "unresolved_items")
        alice = self.new_ctx(user_id="alice")
        none_set = self.call("check_meal_against_budget", [item("egg", 3)], ctx=alice)
        self.assertEqual((none_set["code"], none_set["needs_user_input"]), ("no_target", True))
        self.assertEqual(none_set["meal"]["calories"], 234.0)                      # still tells the user what the meal is
        self.assertEqual(self.call("check_meal_against_budget", "egg")["code"], "invalid_input")

    def test_a_meal_checked_when_already_over_budget_says_so(self):
        self.seed(0, 2500)
        res = self.call("check_meal_against_budget", [item("banana", 1)])
        self.assertFalse(res["fits"])
        self.assertIn("already exceeded", res["message"])


# ===========================================================================
class TestRecommendationTools(ToolCase):

    def _pipeline(self, **kwargs):
        cands = self.call("get_recommendation_candidates", **kwargs)
        self.call("estimate_meal_nutrition")
        return cands, self.call("rank_meal_candidates")

    def test_the_brief_example_550_kcal_and_45_g_protein(self):
        self.seed(0, 1750, 105)
        cands, ranked = self._pipeline()
        self.assertEqual((cands["filters"]["remaining_calories"], cands["filters"]["remaining_protein"], cands["filters"]["source"]),
                         (550.0, 45.0, "shared_state"))
        best = ranked["ranked"][0]
        self.assertLessEqual(best["calories"], 550.0)
        self.assertGreaterEqual(best["protein"], 40.0)
        self.assertIn("45 g protein still needed", best["reason"])
        self.assertEqual(ranked["remaining_used"], {"calories": 550.0, "protein": 45.0})

    def test_no_candidate_ever_exceeds_the_remaining_calories(self):
        for remaining in (120, 300, 550, 900, 2000):
            with self.subTest(remaining=remaining):
                self.setUp()
                cands = self.call("get_recommendation_candidates", remaining_calories=remaining)
                est = self.call("estimate_meal_nutrition")
                if cands["count"]:
                    self.assertLessEqual(max(e["calories"] for e in est["estimates"]), remaining)

    def test_estimates_are_recomputed_from_the_database(self):
        self.call("get_recommendation_candidates", remaining_calories=600, max_candidates=8)
        for e in self.call("estimate_meal_nutrition")["estimates"]:
            expected = 0.0
            for it in e["items"]:
                r = resolve_item(it["food"], it["quantity"], it["unit"])
                expected += r["calories"]
            self.assertEqual(e["calories"], round(expected, 1))
            self.assertEqual(e["name"], " + ".join(f"{i['quantity_str']} {i['food']}" for i in e["items"]))

    def test_ranking_is_best_first_deterministic_and_explained(self):
        _, first = self._pipeline(remaining_calories=550, remaining_protein=45)
        self.setUp()
        _, second = self._pipeline(remaining_calories=550, remaining_protein=45)
        self.assertEqual([r["id"] for r in first["ranked"]], [r["id"] for r in second["ranked"]])
        self.assertEqual([r["name"] for r in first["ranked"]], [r["name"] for r in second["ranked"]])
        for r in first["ranked"]:
            self.assertEqual(set(r["breakdown"]), {"protein", "calorie_fill", "variety", "preference", "protein_share_g"})
            self.assertIn("not medical advice", r["reason"])
            self.assertTrue(r["fits_calories"])
        self.assertEqual(first["alternative_ids"], [r["id"] for r in first["ranked"][1:]])
        self.assertEqual(len(first["ranked"]), 5)
        self.assertAlmostEqual(first["ranked"][0]["score"], 0.5 * first["ranked"][0]["breakdown"]["protein"] + 0.25 * first["ranked"][0]["breakdown"]["calorie_fill"]
                               + 0.15 * first["ranked"][0]["breakdown"]["variety"] + 0.10 * first["ranked"][0]["breakdown"]["preference"], places=3)

    def test_alternatives_are_not_all_variations_of_one_food(self):
        cands = self.call("get_recommendation_candidates", remaining_calories=600, max_candidates=40)
        mains = [c["name"].split(" + ")[0].split(" ", 2)[-1] for c in cands["candidates"]]
        self.assertLessEqual(max(mains.count(m) for m in set(mains)), 4)
        ranked = self._pipeline(remaining_calories=600)[1]["ranked"]
        self.assertGreaterEqual(len({r["items"][0]["food"] for r in ranked}), 3)

    def test_diet_filters(self):
        def foods(diet):
            self.setUp()
            cands = self.call("get_recommendation_candidates", remaining_calories=700, diet=diet, max_candidates=60)
            self.call("estimate_meal_nutrition")
            return {i["food"] for c in self.state.pool["candidates"].values() for i in c["items"]}
        self.assertTrue({"egg", "chicken breast"} <= foods("any") | {"egg", "chicken breast"})
        veg = foods("vegetarian")
        self.assertFalse(veg & {"egg", "chicken breast", "chicken leg"})
        self.assertTrue(veg)
        egg = foods("eggetarian")
        self.assertFalse(egg & {"chicken breast", "chicken leg"})
        res = self.call("get_recommendation_candidates", remaining_calories=700, diet="keto")
        self.assertEqual(res["code"], "invalid_input")

    def test_avoid_list_is_respected_including_aliases_and_saved_preferences(self):
        self.call("get_recommendation_candidates", remaining_calories=700, avoid=["chawal", "Paneer"], max_candidates=60)
        used = {i["food"] for c in self.state.pool["candidates"].values() for i in c["items"]}
        self.assertFalse(used & {"rice", "paneer"})
        self.state.set_preferences(avoid=["roti", "bread"])
        self.call("get_recommendation_candidates", remaining_calories=700, max_candidates=60)     # saved preference applies
        used = {i["food"] for c in self.state.pool["candidates"].values() for i in c["items"]}
        self.assertFalse(used & {"roti", "bread"})

    def test_preferred_foods_get_a_ranking_bonus(self):
        self.call("get_recommendation_candidates", remaining_calories=500, remaining_protein=40, prefer=["banana"], max_candidates=60)
        self.call("estimate_meal_nutrition")
        ranked = self.call("rank_meal_candidates", top_n=10)["ranked"]
        liked = [r for r in ranked if any(i["food"] == "banana" for i in r["items"])]
        self.assertTrue(all(r["breakdown"]["preference"] == 1.0 for r in liked))
        self.assertTrue(all(r["breakdown"]["preference"] == 0.0 for r in ranked if r not in liked))

    def test_custom_foods_can_be_recommended(self):
        self.call("register_custom_food", "mom's rajma", 320, 14, 45, 8, "1 bowl")
        self.forget_custom_food("mom's rajma")
        cands = self.call("get_recommendation_candidates", remaining_calories=500, max_candidates=60)
        self.assertTrue(any("mom's rajma" in c["name"] for c in cands["candidates"]))

    def test_no_budget_no_target_and_tiny_budgets_are_handled(self):
        self.seed(0, 2400)
        over = self.call("get_recommendation_candidates")
        self.assertEqual((over["status"], over["code"], over["count"]), ("success", "no_budget_left", 0))
        self.assertIn("100 kcal over", over["message"])
        self.assertEqual(self.call("rank_meal_candidates")["code"], "no_candidates")
        tiny = self.call("get_recommendation_candidates", remaining_calories=40)
        self.assertEqual((tiny["code"], tiny["count"]), ("no_candidates", 0))
        alice = self.new_ctx(user_id="alice")
        none_set = self.call("get_recommendation_candidates", ctx=alice)
        self.assertEqual((none_set["status"], none_set["code"], none_set["needs_user_input"]), ("failed", "no_target", True))

    def test_the_pipeline_must_be_run_in_order(self):
        self.assertEqual(self.call("estimate_meal_nutrition")["code"], "no_candidates")
        self.assertEqual(self.call("rank_meal_candidates")["code"], "no_candidates")
        self.assertEqual(self.call("validate_recommendation_fit", "c01")["code"], "unknown_candidate")
        self.call("get_recommendation_candidates", remaining_calories=500)
        self.assertEqual(self.call("rank_meal_candidates")["code"], "not_estimated")
        self.assertEqual(self.call("estimate_meal_nutrition", ["c01", "zzz"])["code"], "unknown_candidate")
        self.assertEqual(self.call("estimate_meal_nutrition", ["c01"], [item("egg", 1)])["code"], "invalid_input")

    def test_bad_arguments_are_rejected(self):
        for kwargs in ({"max_candidates": 0}, {"max_candidates": 61}, {"max_candidates": True}, {"remaining_calories": "lots"},
                       {"remaining_calories": float("nan")}, {"avoid": "paneer"}, {"prefer": 5}):
            with self.subTest(kwargs=kwargs):
                self.assertEqual(self.call("get_recommendation_candidates", **kwargs)["code"], "invalid_input")
        self.call("get_recommendation_candidates", remaining_calories=500)
        self.call("estimate_meal_nutrition")
        for top_n in (0, 11, True):
            self.assertEqual(self.call("rank_meal_candidates", top_n=top_n)["code"], "invalid_input")

    def test_ad_hoc_estimate(self):
        res = self.call("estimate_meal_nutrition", items=[item("egg", 2), item("bread", 2, "slice")])
        self.assertEqual((res["estimates"][0]["calories"], res["estimates"][0]["protein"]), (314.0, 17.4))
        self.assertIn("not medical", res["disclaimer"])
        unknown = self.call("estimate_meal_nutrition", items=[item("idli", 2)])
        self.assertEqual((unknown["status"], unknown["code"]), ("failed", "unknown_food"))

    def test_validate_saves_only_valid_meals_and_marks_them_stale_after_eating(self):
        self.call("get_recommendation_candidates", remaining_calories=800)
        self.call("estimate_meal_nutrition")
        ranked = self.call("rank_meal_candidates", top_n=3)
        ok1 = self.call("validate_recommendation_fit", ranked["ranked"][0]["id"])
        self.call("validate_recommendation_fit", ranked["ranked"][1]["id"])
        recs = self.state.snapshot()["recommendations"]
        self.assertEqual([r["id"] for r in recs], [ok1["recommendation"]["id"], ranked["ranked"][1]["id"]])
        self.assertTrue(all(r["estimated"] and "not medical" in r["disclaimer"] for r in recs))
        self.assertFalse(self.state.snapshot()["recommendations_stale"] and False)
        self.call("add_meal_to_state", [item("egg", 1)], ctx=self.new_ctx())
        self.assertTrue(self.state.snapshot()["recommendations_stale"])

    def test_validate_catches_a_meal_that_no_longer_fits(self):
        self.seed(0, 1800)                                                          # 500 kcal left
        self.call("get_recommendation_candidates")
        self.call("estimate_meal_nutrition")
        best = self.call("rank_meal_candidates")["best_id"]
        self.seed(0, 300)                                                           # ... then 300 more is eaten: 200 left
        res = self.call("validate_recommendation_fit", best)
        self.assertEqual((res["status"], res["code"], res["valid"]), ("failed", "invalid_recommendation", False))
        self.assertIn("exceeds", res["issues"][0])
        self.assertEqual(self.state.snapshot()["recommendations"], [])

    def test_validate_catches_a_tampered_estimate(self):
        self.call("get_recommendation_candidates", remaining_calories=500)
        self.call("estimate_meal_nutrition")
        self.state.pool["candidates"]["c01"]["estimate"]["calories"] = 1.0         # something altered the numbers
        res = self.call("validate_recommendation_fit", "c01")
        self.assertFalse(res["valid"])
        self.assertIn("no longer matches", res["issues"][0])

    def test_validate_uses_the_hypothetical_remaining_it_was_built_for(self):
        self.call("get_recommendation_candidates", remaining_calories=500)
        self.call("estimate_meal_nutrition")
        best = self.call("rank_meal_candidates")["best_id"]
        self.assertTrue(self.call("validate_recommendation_fit", best)["valid"])
        self.assertFalse(self.call("validate_recommendation_fit", best, remaining_calories=100)["valid"])


if __name__ == "__main__":
    unittest.main()
