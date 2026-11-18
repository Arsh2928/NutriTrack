"""
Phase 2 tests: the CA1 code was moved into the nutritrack package and must
behave exactly as it did in CA1.

Written as unittest.TestCase classes: pytest collects these natively
(`pytest`), and they also run with the standard library
(`python -m unittest discover -s tests -t .`).
"""
import hashlib
import json
import os
import subprocess
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

from tests.support import IsolatedCase, known_defect

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _tools():
    from nutritrack.ca1 import tools
    return tools


# ===========================================================================
class TestLayoutAndPaths(IsolatedCase):

    def test_ca1_modules_import_from_the_package(self):
        import importlib
        for name in ("memory", "config", "llm_config", "tool_schemas",
                     "food_database", "tools", "daily_store", "agent"):
            importlib.import_module(f"nutritrack.ca1.{name}")

    def test_optional_llm_module_needs_only_the_openai_package(self):
        """llm_agent.py is the only CA1 module with a third-party import. If it
        fails here, the ONLY acceptable reason is that 'openai' isn't installed."""
        import importlib
        try:
            importlib.import_module("nutritrack.ca1.llm_agent")
        except ModuleNotFoundError as exc:
            self.assertEqual(exc.name, "openai")

    def test_new_subpackages_exist(self):
        import importlib
        for name in ("agents", "tools", "state", "web"):
            importlib.import_module(f"nutritrack.{name}")

    def test_paths_follow_environment_and_are_created_on_demand(self):
        from nutritrack import paths
        target = Path(self.tmp) / "does" / "not" / "exist" / "yet"
        self.patch_env("NUTRITRACK_DATA_DIR", str(target))
        self.assertEqual(paths.data_dir(), target)
        self.assertTrue(target.is_dir(), "data_dir() must create the folder (CA1 saved silently to nowhere)")
        logs = Path(self.tmp) / "logs_here"
        self.patch_env("NUTRITRACK_LOG_DIR", str(logs))
        self.assertEqual(paths.logs_dir(), logs)
        self.assertTrue(logs.is_dir())

    def test_default_locations_are_project_data_and_logs(self):
        from nutritrack import paths
        self.assertEqual(paths.PROJECT_ROOT, PROJECT_ROOT)
        env = {k: v for k, v in os.environ.items() if k not in ("NUTRITRACK_DATA_DIR", "NUTRITRACK_LOG_DIR")}
        out = subprocess.run(
            [sys.executable, "-c", "from nutritrack import paths; print(paths.data_dir()); print(paths.logs_dir())"],
            cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        data, logs = [line.strip() for line in out.stdout.strip().splitlines()]
        self.assertEqual(Path(data), PROJECT_ROOT / "data")
        self.assertEqual(Path(logs), PROJECT_ROOT / "logs")

    def test_custom_foods_path_follows_environment(self):
        from nutritrack.ca1 import food_database
        self.assertEqual(food_database._custom_path(), os.environ["NUTRITRACK_CUSTOM_FOODS"])


# ===========================================================================
class TestCA1RecordedBehaviour(IsolatedCase):
    """Everything CA1 itself recorded (notebook outputs, screenshots, TESTING.md)."""

    # (kcal, protein, carbs, fat) exactly as recorded in CA1
    RECORDED = {
        "3 boiled eggs": (234.0, 18.0, 1.8, 15.9),
        "4 boiled eggs": (312.0, 24.0, 2.4, 21.2),
        "250 ml milk": (155.2, 8.3, 12.5, 8.3),
        "40 g oats": (148.0, 5.4, 26.6, 2.6),
        "50 gm oats": (185.0, 6.8, 33.2, 3.2),
        "100gm paneer": (265.0, 18.3, 1.2, 20.8),
        "200 g chicken breast": (330.0, 62.0, 0.0, 7.2),
        "2 slices of bread": (158.0, 5.4, 30.0, 2.0),
        "2 roti": (208.0, 6.2, 36.0, 5.4),
        "100 g curd": (98.0, 11.0, 3.4, 4.3),
        "a banana": (105.0, 1.3, 27.0, 0.4),
        "apple": (95.0, 0.5, 25.0, 0.3),
        "100 g soya chunks": (345.0, 52.0, 33.0, 0.5),
    }

    def test_recorded_nutrition_values_reproduce_exactly(self):
        tools = _tools()
        for query, expected in self.RECORDED.items():
            with self.subTest(query=query):
                res = tools.lookup_calories(query)
                self.assertEqual(res["status"], "found")
                self.assertEqual((res["calories"], res["protein"], res["carbs"], res["fat"]), expected)

    def test_ca1_testing_md_script(self):
        from nutritrack.ca1.agent import MealCaloricAgent
        agent = MealCaloricAgent(verbose=False)
        r = agent.chat_capture("My calorie budget is 2200 kcal");  self.assertIn("2200", r["response"])
        r = agent.chat_capture("Add 3 boiled eggs");               self.assertIn("234", r["response"])
        r = agent.chat_capture("Can I eat a banana?");             self.assertIn("kcal", r["response"].lower())
        r = agent.chat_capture("Show today's summary");            self.assertIn("Egg", r["response"])
        r = agent.chat_capture("delete meal 1");                   self.assertIn("removed", r["response"].lower())
        r = agent.chat_capture("Reset my meals");                  self.assertIn("clear", r["response"].lower())

    def test_scenario_3_budget_decisions(self):
        tools = _tools()
        tools.set_budget(400)
        tools.add_meal("2 roti")
        tools.add_meal("100 g curd")
        banana = tools.check_snack_fit("banana")
        self.assertEqual((banana["status"], banana["surplus_deficit"]), ("does_not_fit", -11.0))
        soya = tools.check_snack_fit("100 g soya chunks")
        self.assertEqual((soya["status"], soya["surplus_deficit"]), ("does_not_fit", -251.0))

    def test_snack_decision_flips_with_remaining_budget(self):
        """TESTING.md section 3: same food, opposite verdicts, computed each time."""
        tools = _tools()
        tools.set_budget(2200)
        self.assertEqual(tools.check_snack_fit("a banana")["status"], "fits")
        tools.add_meal("30 boiled eggs")   # deliberately overshoot
        self.assertEqual(tools.check_snack_fit("a banana")["status"], "does_not_fit")

    def test_snack_that_exactly_equals_the_remaining_budget_fits(self):
        """Boundary: 105 kcal banana, 105 kcal budget -> fits with 0 to spare."""
        tools = _tools()
        tools.set_budget(105)
        res = tools.check_snack_fit("a banana")
        self.assertEqual((res["status"], res["surplus_deficit"]), ("fits", 0.0))
        tools.set_budget(104.9)                                   # one tenth of a kcal short
        self.assertEqual(tools.check_snack_fit("a banana")["status"], "does_not_fit")

    def test_unknown_foods_are_never_given_invented_nutrition(self):
        tools = _tools()
        for query in ("pizza", "samosa", "a plate of biryani", "maple syrup", "peanut butter", "qwxyz123abc"):
            with self.subTest(query=query):
                res = tools.lookup_calories(query)
                self.assertEqual(res["status"], "not_found")
                self.assertNotIn("calories", res)
                self.assertTrue(res.get("can_add_manually"))

    def test_add_meal_without_budget_and_check_without_budget(self):
        tools = _tools()
        self.assertEqual(tools.check_snack_fit("banana")["status"], "no_budget")
        res = tools.add_meal("2 roti")
        self.assertEqual((res["status"], res["remaining_kcal"]), ("added", None))

    def test_delete_by_index_and_by_name_recompute_totals(self):
        tools = _tools()
        tools.set_budget(2000)
        tools.add_meal("3 boiled eggs"); tools.add_meal("200 g chicken breast"); tools.add_meal("2 roti")
        res = tools.delete_meal(food_name="chicken")
        self.assertEqual(res["status"], "deleted")
        self.assertEqual(res["daily_totals"]["calories"], 234.0 + 208.0)
        res = tools.delete_meal(index=1)
        self.assertEqual(res["daily_totals"]["calories"], 208.0)
        self.assertEqual(tools.delete_meal(index=9)["status"], "not_found")
        self.assertEqual(tools.delete_meal()["status"], "error")

    def test_macro_split_formula(self):
        tools = _tools()
        tools.add_meal("200 g chicken breast")   # P 62, C 0, F 7.2
        ms = tools.get_daily_summary()["macro_split"]
        kcal = 62.0 * 4 + 0.0 * 4 + 7.2 * 9
        self.assertEqual(ms["total_from_macros_kcal"], round(kcal, 1))
        self.assertEqual(ms["protein_pct"], round(62.0 * 4 / kcal * 100, 1))


# ===========================================================================
class TestCA1Persistence(IsolatedCase):
    """TESTING.md sections 6 and 8, automated."""

    def test_same_day_restart_restores_meals_and_budget(self):
        from nutritrack.ca1 import daily_store
        from nutritrack.ca1.memory import SessionMemory
        tools = _tools()
        tools.set_budget(2200); tools.add_meal("3 boiled eggs")
        daily_store.sync_after_change(tools.get_memory())
        fresh = SessionMemory()                                   # simulated restart
        self.assertEqual(daily_store.load_or_rollover(fresh)["action"], "restored")
        self.assertEqual((fresh.daily_budget, fresh.total_calories, len(fresh.meals)), (2200.0, 234.0, 1))

    def test_new_day_archives_yesterday_and_carries_budget(self):
        from nutritrack.ca1 import daily_store
        from nutritrack.ca1.memory import SessionMemory
        tools = _tools()
        tools.set_budget(2200); tools.add_meal("3 boiled eggs"); tools.add_meal("2 roti")
        daily_store.sync_after_change(tools.get_memory())
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        store = json.loads(Path(daily_store._STORE_FILE).read_text(encoding="utf-8"))
        store["current_date"] = yesterday
        Path(daily_store._STORE_FILE).write_text(json.dumps(store), encoding="utf-8")

        fresh = SessionMemory()
        result = daily_store.load_or_rollover(fresh)
        self.assertEqual(result, {"action": "rolled_over", "previous_date": yesterday})
        self.assertEqual((len(fresh.meals), fresh.daily_budget), (0, 2200.0))
        day = daily_store.get_history()[yesterday]
        self.assertEqual((day["total_calories"], len(day["meals"])), (442.0, 2))

    def test_custom_food_survives_a_restart(self):
        from nutritrack.ca1 import food_database
        tools = _tools()
        res = tools.add_custom_food_tool("mom's rajma", 320, 14, 45, 8, "1 bowl (250g)")
        self.assertEqual(res["status"], "added_to_database")
        self.addCleanup(food_database.FOOD_DB.pop, "mom's rajma", None)
        self.assertEqual(tools.lookup_calories("1 mom's rajma")["calories"], 320.0)
        del food_database.FOOD_DB["mom's rajma"]                  # simulated restart ...
        self.assertIsNone(food_database.FOOD_DB.get("mom's rajma"))
        food_database._load_custom_foods()                        # ... reload from disk
        self.assertEqual(tools.lookup_calories("1 mom's rajma")["calories"], 320.0)
        self.assertEqual(food_database.find_food("1 mom's rajma"), "mom's rajma")  # beats built-in 'rajma'


# ===========================================================================
class TestWebApiContract(IsolatedCase):
    """The four endpoints and JSON keys that frontend/index.html depends on."""

    SUMMARY_KEYS = {"daily_budget", "total_calories", "remaining_calories",
                    "total_protein", "total_carbs", "total_fat", "meal_count", "meals"}

    def setUp(self):
        super().setUp()
        from nutritrack.web import app as web_app
        self.client = web_app.app.test_client()

    def test_index_page_is_served_from_the_new_static_folder(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Meal Calorie Helper", res.data)
        res.close()

    def test_mode_endpoint_reports_rule_based_without_a_token(self):
        data = self.client.get("/api/mode").get_json()
        self.assertEqual(data, {"agent_mode": "rule_based", "model": None})

    def test_summary_endpoint_has_every_key_the_frontend_reads(self):
        data = self.client.get("/api/summary").get_json()
        self.assertEqual(set(data), self.SUMMARY_KEYS)

    def test_chat_flow_updates_summary(self):
        self.client.post("/api/chat", json={"message": "My calorie budget is 2200 kcal"})
        res = self.client.post("/api/chat", json={"message": "Add 3 boiled eggs"})
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(set(body), {"response", "trace", "summary", "agent_mode"})
        self.assertIn("234", body["response"])
        self.assertIn("[ACT]", body["trace"])
        self.assertEqual(set(body["summary"]), self.SUMMARY_KEYS)
        self.assertEqual((body["summary"]["total_calories"], body["summary"]["remaining_calories"]), (234.0, 1966.0))

    def test_frontend_delete_button_path(self):
        """The ledger's X button sends 'delete meal N' through /api/chat."""
        self.client.post("/api/chat", json={"message": "Add 3 boiled eggs"})
        body = self.client.post("/api/chat", json={"message": "delete meal 1"}).get_json()
        self.assertEqual(body["summary"]["meal_count"], 0)

    def test_empty_message_is_rejected_with_400(self):
        res = self.client.post("/api/chat", json={"message": "   "})
        self.assertEqual(res.status_code, 400)
        self.assertIn("error", res.get_json())

    def test_history_endpoint_starts_empty(self):
        self.assertEqual(self.client.get("/api/history").get_json(), {})

    def test_chat_persists_to_the_redirected_data_file(self):
        from nutritrack.ca1 import daily_store
        self.client.post("/api/chat", json={"message": "Add 3 boiled eggs"})
        stored = json.loads(Path(daily_store._STORE_FILE).read_text(encoding="utf-8"))
        self.assertEqual(stored["current_date"], date.today().isoformat())
        self.assertEqual(len(stored["meals"]), 1)


# ===========================================================================
class TestEntryPoints(IsolatedCase):

    def _run(self, args, stdin=None):
        env = dict(os.environ, PYTHONIOENCODING="utf-8", NUTRITRACK_DATA_DIR=self.tmp)
        env.pop("NUTRITRACK_CUSTOM_FOODS", None)
        return subprocess.run([sys.executable, *args], cwd=PROJECT_ROOT, env=env,
                              input=stdin, capture_output=True, text=True, timeout=60)

    def test_python_app_py_still_works_as_the_entry_point(self):
        out = self._run(["-c", "import app; assert callable(app.main); print('entry-ok')"])
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("entry-ok", out.stdout)

    def test_ca1_terminal_chat_runs_as_a_module(self):
        out = self._run(["-m", "nutritrack.ca1.agent"],
                        stdin="My calorie budget is 2200 kcal\nAdd 3 boiled eggs\nquit\n")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("234", out.stdout)


# ===========================================================================
class TestDataMigrationScript(IsolatedCase):

    SCRIPT = PROJECT_ROOT / "scripts" / "migrate_ca1_data.py"
    GOOD_FOOD = {"unit": "piece", "base_amount": 1.0, "serving_size": "1 bowl (250g)",
                 "calories_per_unit": 320.0, "protein": 14.0, "carbs": 45.0, "fat": 8.0}

    def setUp(self):
        super().setUp()
        self.old = Path(self.tmp) / "old_ca1_code"
        self.dest = Path(self.tmp) / "new_data"
        self.old.mkdir()
        (self.old / "custom_foods.json").write_text(json.dumps({
            "mom's rajma": self.GOOD_FOOD,
            "weird food": {"calories": 100},                     # wrong shape: must be reported
        }), encoding="utf-8")
        (self.old / "daily_log.json").write_text(json.dumps({
            "current_date": "2026-09-01", "budget": 2200.0, "meals": [],
            "history": {"2026-08-31": {"budget": 2200.0, "meals": [], "total_calories": 0.0,
                                       "total_protein": 0.0, "total_carbs": 0.0, "total_fat": 0.0}},
        }), encoding="utf-8")

    def _run(self, *extra, src=None):
        env = dict(os.environ, NUTRITRACK_DATA_DIR=str(self.dest))
        env.pop("NUTRITRACK_CUSTOM_FOODS", None)
        return subprocess.run([sys.executable, str(self.SCRIPT), "--from", str(src or self.old), *extra],
                              cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, timeout=60)

    @staticmethod
    def _digest(folder):
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(folder).glob("*.json"))}

    def test_copies_valid_files_reports_incompatible_entries_and_leaves_source_untouched(self):
        before = self._digest(self.old)
        out = self._run()
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn("1 of 2 custom food(s) compatible", out.stdout)
        self.assertIn("'weird food': missing unit", out.stdout)
        self.assertIn("1 past day(s) in history", out.stdout)
        self.assertEqual(self._digest(self.dest), before)          # copied byte-for-byte
        self.assertEqual(self._digest(self.old), before)           # source never modified

    def test_dry_run_copies_nothing(self):
        out = self._run("--dry-run")
        self.assertEqual(out.returncode, 0)
        self.assertIn("would copy", out.stdout)
        self.assertEqual(self._digest(self.dest), {})

    def test_existing_destination_is_not_overwritten_without_force(self):
        self.dest.mkdir()
        keep = '{"current_date": null, "budget": 1, "meals": [], "history": {}}'
        (self.dest / "daily_log.json").write_text(keep, encoding="utf-8")
        out = self._run()
        self.assertEqual(out.returncode, 0)
        self.assertEqual((self.dest / "daily_log.json").read_text(encoding="utf-8"), keep)
        self.assertIn("left unchanged", out.stdout)
        self._run("--force")
        self.assertNotEqual((self.dest / "daily_log.json").read_text(encoding="utf-8"), keep)

    def test_unreadable_file_is_reported_and_not_copied(self):
        (self.old / "custom_foods.json").write_text('{"broken": ', encoding="utf-8")
        out = self._run()
        self.assertEqual(out.returncode, 1)
        self.assertIn("UNREADABLE", out.stdout)
        self.assertFalse((self.dest / "custom_foods.json").exists())
        self.assertTrue((self.dest / "daily_log.json").exists())   # the valid file is still migrated

    def test_missing_source_folder_is_an_error(self):
        out = self._run(src=Path(self.tmp) / "nope")
        self.assertEqual(out.returncode, 1)
        self.assertIn("does not exist", out.stdout)

    def test_migrated_custom_food_actually_loads_in_the_new_project(self):
        self._run()
        env = dict(os.environ, NUTRITRACK_DATA_DIR=str(self.dest))
        env.pop("NUTRITRACK_CUSTOM_FOODS", None)
        code = ("from nutritrack.ca1.food_database import find_food, FOOD_DB;"
                "k = find_food(\"1 mom's rajma\"); print(k, FOOD_DB[k]['calories_per_unit'])")
        out = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, env=env,
                             capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "mom's rajma 320.0")


# ===========================================================================
class TestFixedCA1Defects(IsolatedCase):
    """Defects found in the Phase 1 audit, fixed in Phase 3. In Phase 2 these were
    @known_defect markers (reported as skipped); each flipped to "appears to be
    FIXED" when the fix landed and is now an ordinary regression test."""

    def _lookup(self, query):
        return _tools().lookup_calories(query)

    def _assert_refused(self, query, reason):
        res = self._lookup(query)
        self.assertEqual(res["status"], "not_found", (query, res))
        self.assertEqual(res["reason"], reason, (query, res))
        self.assertNotIn("calories", res, "a refused lookup must never carry nutrition values")
        return res

    # --- D1: unit mismatch -------------------------------------------------
    def test_d1_grams_of_a_piece_food_is_refused_not_multiplied(self):
        for query in ("200 g banana", "100 g egg"):
            with self.subTest(query=query):
                res = self._assert_refused(query, "unit_mismatch")
                self.assertIn("How many", res["error"])

    def test_d1_conversions_that_need_an_invented_factor_are_refused(self):
        for query in ("1 cup of rice", "2 tbsp butter", "2 slices of banana", "30 g whey protein", "100 ml curd"):
            with self.subTest(query=query):
                self._assert_refused(query, "unit_mismatch")

    def test_d1_valid_conversions_still_work(self):
        self.assertEqual(self._lookup("1 cup of milk")["calories"], 149.0)          # 240 ml
        self.assertEqual(self._lookup("1 kg rice")["calories"], 1300.0)             # 1000 g
        self.assertEqual(self._lookup("2 scoops whey protein")["calories"], 240.0)

    # --- D2: ambiguous bare numbers ----------------------------------------
    def test_d2_bare_number_on_a_weight_food_asks_for_the_unit(self):
        self._assert_refused("2 curd", "ambiguous_quantity")
        self._assert_refused("250 curd", "ambiguous_quantity")
        res = self._assert_refused("curd", "missing_quantity")
        self.assertIn("grams", res["error"])

    # --- D3: invalid quantities --------------------------------------------
    def test_d3_invalid_quantities_are_rejected(self):
        cases = {"0 eggs": "invalid_quantity", "-3 eggs": "invalid_quantity", "0 g oats": "invalid_quantity",
                 "999999 eggs": "implausible_quantity", "6 kg rice": "implausible_quantity",
                 "1e3 eggs": "ambiguous_quantity", "vitamin b12 pills": "ambiguous_quantity"}
        for query, reason in cases.items():
            with self.subTest(query=query):
                self._assert_refused(query, reason)

    def test_d3_number_words_and_fractions_are_read_correctly(self):
        """CA1 logged "two roti" as ONE roti."""
        self.assertEqual(self._lookup("two roti")["calories"], 208.0)
        self.assertEqual(self._lookup("a dozen eggs")["calories"], 936.0)
        half = self._lookup("half a banana")
        self.assertEqual((half["calories"], half["quantity_str"]), (52.5, "0.5 pieces"))
        self.assertEqual(self._lookup("1/2 cup of milk")["calories"], 74.5)

    def test_d3_trailing_quantities_are_understood(self):
        self.assertEqual(self._lookup("paneer 100g")["calories"], 265.0)
        self.assertEqual(self._lookup("eggs x3")["calories"], 234.0)
        self.assertEqual(self._lookup("some fresh spinach 50 g")["calories"], 11.5)

    # --- D4: brand / preparation matched to a generic food -----------------
    def test_d4_branded_product_is_not_silently_matched(self):
        res = self._assert_refused("pintola high protein muesli", "low_confidence_match")
        self.assertTrue(res["can_add_manually"])
        self.assertIn("pintola", res["error"])

    def test_d4_preparation_words_that_change_nutrition_are_not_ignored(self):
        self._assert_refused("100 g spicy paneer", "low_confidence_match")
        self._assert_refused("fried egg", "unknown_food")

    def test_d4_harmless_descriptors_are_still_accepted(self):
        for query in ("3 boiled eggs", "a large banana", "100 g cooked rice", "some fresh spinach 50 g"):
            with self.subTest(query=query):
                self.assertEqual(self._lookup(query)["status"], "found")

    # --- D5: persistence ---------------------------------------------------
    def test_d5_budget_survives_restart_when_no_meals_logged(self):
        from nutritrack.ca1 import daily_store
        from nutritrack.ca1.memory import SessionMemory
        first = SessionMemory(); first.set_budget(2200)
        daily_store.load_or_rollover(first)
        daily_store.sync_after_change(first)
        second = SessionMemory()
        self.assertEqual(daily_store.load_or_rollover(second)["action"], "restored")
        self.assertEqual(second.daily_budget, 2200.0)

    def test_d5_default_store_is_not_shared_between_loads(self):
        from nutritrack.ca1 import daily_store
        one = daily_store._load_store()
        one["meals"].append({"food": "LEAK"}); one["history"]["2026-01-01"] = {}
        two = daily_store._load_store()
        self.assertEqual((two["meals"], two["history"]), ([], {}))
        self.assertEqual((daily_store._EMPTY_STORE["meals"], daily_store._EMPTY_STORE["history"]), ([], {}))

    def test_d5_corrupt_log_is_quarantined_not_overwritten(self):
        from nutritrack.ca1 import daily_store
        Path(daily_store._STORE_FILE).write_text('{"current_date": "2026-11-12", "meals": [ {"food"', encoding="utf-8")
        with self.assertLogs("nutritrack.ca1.daily_store", level="ERROR"):
            store = daily_store._load_store()
        self.assertEqual(store["meals"], [])
        backups = list(Path(self.tmp).glob("daily_log.json.corrupt-*"))
        self.assertEqual(len(backups), 1, "the unreadable file must be kept, not deleted")
        self.assertIn('"food"', backups[0].read_text(encoding="utf-8"))

    def test_d5_failed_save_is_reported_not_silent(self):
        from nutritrack.ca1 import daily_store
        blocker = Path(self.tmp) / "i_am_a_file"
        blocker.write_text("x")
        self.patch_attr(daily_store, "_STORE_FILE", str(blocker / "daily_log.json"))   # cannot be a directory
        with self.assertLogs("nutritrack.ca1.daily_store", level="ERROR"):
            self.assertFalse(daily_store.sync_after_change(_tools().get_memory()))

    def test_d5_save_is_atomic(self):
        from unittest import mock
        from nutritrack.ca1 import daily_store
        tools = _tools()
        tools.set_budget(2000)
        self.assertTrue(daily_store.sync_after_change(tools.get_memory()))
        good = Path(daily_store._STORE_FILE).read_text(encoding="utf-8")
        with mock.patch.object(daily_store.json, "dump", side_effect=OSError("disk full")):
            with self.assertLogs("nutritrack.ca1.daily_store", level="ERROR"):
                self.assertFalse(daily_store.sync_after_change(tools.get_memory()))
        self.assertEqual(Path(daily_store._STORE_FILE).read_text(encoding="utf-8"), good)
        self.assertEqual(list(Path(self.tmp).glob(".daily_log_*.tmp")), [], "no temp files left behind")

    # --- D6: totals ---------------------------------------------------------
    def test_d6_totals_do_not_drift_after_many_add_remove_cycles(self):
        from nutritrack.ca1.memory import MealEntry, SessionMemory
        mem = SessionMemory()
        for _ in range(1000):
            mem.add_meal(MealEntry("x", "1 piece", 0.1, 0.1, 0.1, 0.1))
        for _ in range(1000):
            mem.remove_meal(1)
        self.assertEqual((mem._total_calories, mem._total_protein, mem._total_carbs, mem._total_fat), (0.0, 0.0, 0.0, 0.0))
        for _ in range(10):
            mem.add_meal(MealEntry("x", "1 piece", 0.1, 0.1, 0.1, 0.1))
        self.assertEqual(mem.total_calories, 1.0)

    # --- D7: online lookup is opt-in ---------------------------------------
    def test_d7_online_lookup_is_off_unless_explicitly_enabled(self):
        from nutritrack.ca1 import config
        self.patch_attr(config, "API_PROVIDER", "openfoodfacts")
        os.environ.pop("NUTRITRACK_ONLINE_LOOKUP", None)
        self.assertFalse(config.has_api_key())
        self.patch_env("NUTRITRACK_ONLINE_LOOKUP", "1")
        self.assertTrue(config.has_api_key())


# ===========================================================================
class TestKnownDefectDecorator(unittest.TestCase):
    """The decorator must never hide a broken test or a fixed defect."""

    @staticmethod
    def _run(body):
        class Probe(IsolatedCase):
            @known_defect("DX", "demo")
            def test_probe(self):
                body(self)
        result = unittest.TestResult()
        Probe("test_probe").run(result)
        return result

    def test_assertion_failure_is_reported_as_skipped_known_defect(self):
        result = self._run(lambda t: t.assertTrue(False))
        self.assertEqual((len(result.skipped), len(result.failures), len(result.errors)), (1, 0, 0))
        self.assertIn("KNOWN DEFECT DX", result.skipped[0][1])

    def test_passing_body_fails_so_a_fixed_defect_is_noticed(self):
        result = self._run(lambda t: None)
        self.assertEqual((len(result.failures), len(result.skipped)), (1, 0))
        self.assertIn("appears to be FIXED", result.failures[0][1])

    def test_unexpected_exception_is_a_real_error_not_a_skip(self):
        def broken(_):
            raise KeyError("typo in the test itself")
        result = self._run(broken)
        self.assertEqual((len(result.errors), len(result.skipped)), (1, 0))


if __name__ == "__main__":
    unittest.main()
