"""Phase 3 tests: the shared NutritionState and the append-only run log."""
import json
import math
import random
import threading
import unittest
from pathlib import Path
from unittest import mock

from nutritrack.state import (DEMO_DEFAULT_TARGETS, RunLog, StateError, StateManager, StateUpdateError,
                              new_run_id, redact, summarize)
from tests.support import StateCase, make_entry


# ===========================================================================
class TestStateBasics(StateCase):

    def test_demo_user_gets_clearly_labelled_demo_defaults(self):
        snap = self.state.snapshot()
        self.assertEqual(snap["targets"], DEMO_DEFAULT_TARGETS)
        self.assertEqual(snap["targets_source"], {"calories": "demo_default", "protein": "demo_default"})
        self.assertEqual((snap["workflow_status"], snap["meals"], snap["weekly_history"], snap["recommendations"]),
                         ("idle", [], [], []))

    def test_snapshot_has_every_field_of_the_brief(self):
        snap = self.state.snapshot()
        for key in ("user_id", "date", "targets", "consumed", "remaining", "meals", "weekly_history",
                    "recommendations", "agent_results", "workflow_status"):
            self.assertIn(key, snap)
        self.assertEqual(snap["consumed"], {"calories": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0})
        self.assertEqual(snap["remaining"], {"calories": 2300.0, "protein": 150.0})
        self.assertEqual(snap["date"], self.clock.today)

    def test_other_users_start_without_targets_and_are_never_given_defaults(self):
        alice = self.manager.get("alice")
        self.assertEqual(alice.snapshot()["targets"], {"calories": None, "protein": None})
        self.assertEqual(alice.snapshot()["remaining"], {"calories": None, "protein": None})

    def test_users_are_completely_separate(self):
        alice = self.manager.get("alice")
        self.state.add_meals([make_entry()])
        alice.add_meals([make_entry("banana", "1 piece", 105, 1.3, 27, 0.4)])
        self.assertEqual([m["food"] for m in self.state.get_day_meals()], ["egg"])
        self.assertEqual([m["food"] for m in alice.get_day_meals()], ["banana"])
        self.assertEqual(self.manager.list_users(), ["alice", "demo_user"])
        self.assertNotEqual(self.manager.path_for("alice"), self.manager.path_for("demo_user"))

    def test_unsafe_user_ids_are_rejected(self):
        for bad in ("../evil", "a/b", "", "x" * 65, "has space", "dots.dots", None, 5):
            with self.subTest(user_id=bad):
                with self.assertRaises(StateError):
                    self.manager.get(bad)
        self.assertEqual(list(self.base.rglob("evil*")), [])

    def test_returned_data_is_a_copy_so_callers_cannot_corrupt_state(self):
        self.state.add_meals([make_entry()])
        snap = self.state.snapshot()
        snap["meals"][0]["calories"] = 99999
        snap["targets"]["calories"] = 1
        self.assertEqual(self.state.day_totals()["calories"], 234.0)
        self.assertEqual(self.state.targets_for()["calories"], 2300.0)


# ===========================================================================
class TestMealsAndConsistency(StateCase):

    def test_totals_and_remaining_are_derived_from_the_meals(self):
        self.state.add_meals([make_entry(), make_entry("banana", "1 piece", 105, 1.3, 27, 0.4)])
        snap = self.state.snapshot()
        self.assertEqual(snap["consumed"], {"calories": 339.0, "protein": 19.3, "carbs": 28.8, "fat": 16.3})
        self.assertEqual(snap["remaining"], {"calories": 1961.0, "protein": 130.7})

    def test_consumed_always_equals_the_sum_of_the_meals_after_random_operations(self):
        """Property test: 300 random adds/removes; after EVERY step the derived totals must equal an
        independent recomputation, and nothing may drift."""
        rng = random.Random(20261113)
        model = []                                                   # independent list of live entries
        for step in range(300):
            if model and rng.random() < 0.4:
                victim = rng.choice(model)
                self.state.remove_meal(meal_id=victim["meal_id"])
                model.remove(victim)
            else:
                batch = [make_entry(rng.choice(["egg", "rice", "oats"]), "x",
                                    round(rng.uniform(0, 900), 1), round(rng.uniform(0, 60), 1),
                                    round(rng.uniform(0, 120), 1), round(rng.uniform(0, 60), 1))
                         for _ in range(rng.randint(1, 3))]
                model.extend(self.state.add_meals(batch, request_id=f"r{step}")["added"])
            snap = self.state.snapshot()
            for key in ("calories", "protein", "carbs", "fat"):
                self.assertEqual(snap["consumed"][key], round(math.fsum(m[key] for m in model), 1), (step, key))
            self.assertEqual(snap["remaining"]["calories"], round(2300.0 - snap["consumed"]["calories"], 1))
            self.assertEqual(len(snap["meals"]), len(model))
        self.assertGreater(len(model), 0)

    def test_totals_do_not_drift_after_many_cycles(self):
        for i in range(200):
            ids = self.state.add_meals([make_entry("x", "1", 0.1, 0.1, 0.1, 0.1)], request_id=f"d{i}")["meal_ids"]
            self.state.remove_meal(meal_id=ids[0])
        self.assertEqual(self.state.day_totals()["calories"], 0.0)
        self.assertEqual(self.state.recorded_dates(), [])

    def test_meal_entries_get_ids_batch_timestamps_and_type(self):
        res = self.state.add_meals([make_entry(), make_entry("banana", "1 piece", 105, 1.3, 27, 0.4)],
                                   meal_type="breakfast", source="meal_intake")
        a, b = res["added"]
        self.assertNotEqual(a["meal_id"], b["meal_id"])
        self.assertEqual(a["batch_id"], b["batch_id"])
        self.assertEqual((a["meal_type"], a["source"], a["date"]), ("breakfast", "meal_intake", self.clock.today))
        self.assertEqual(a["logged_at"], "2026-11-13T12:00:00")
        self.assertTrue(a["estimated"])

    # --- duplicates --------------------------------------------------------
    def test_replaying_a_request_id_adds_nothing(self):
        first = self.state.add_meals([make_entry()], request_id="click-1")
        again = self.state.add_meals([make_entry()], request_id="click-1")
        self.assertFalse(first["duplicate"])
        self.assertTrue(again["duplicate"])
        self.assertEqual(again["meal_ids"], first["meal_ids"])
        self.assertEqual(len(self.state.get_day_meals()), 1)

    def test_eating_the_same_food_twice_is_allowed_with_different_request_ids(self):
        self.state.add_meals([make_entry()], request_id="lunch")
        self.state.add_meals([make_entry()], request_id="dinner")
        self.assertEqual(len(self.state.get_day_meals()), 2)

    def test_a_deleted_meal_can_be_logged_again_with_the_same_request_id(self):
        first = self.state.add_meals([make_entry()], request_id="r1")
        self.state.remove_meal(meal_id=first["meal_ids"][0])
        again = self.state.add_meals([make_entry()], request_id="r1")
        self.assertFalse(again["duplicate"])
        self.assertEqual(len(self.state.get_day_meals()), 1)

    def test_request_id_memory_is_bounded(self):
        from nutritrack.state import nutrition_state as ns
        with mock.patch.object(ns, "MAX_REQUEST_IDS", 5):
            for i in range(12):
                self.state.add_meals([make_entry()], request_id=f"r{i}")
        self.assertLessEqual(len(self.state._doc["request_ids"]), 5)

    # --- validation --------------------------------------------------------
    def test_invalid_entries_are_rejected_and_change_nothing(self):
        bad_entries = [
            make_entry(calories=-1), make_entry(calories=float("nan")), make_entry(calories=float("inf")),
            make_entry(calories=True), make_entry(calories="234"), make_entry(calories=10001),
            make_entry(food=""), make_entry(food="x" * 101), make_entry(quantity_str=""),
            {"food": "egg"}, "not a dict",
        ]
        for entry in bad_entries:
            with self.subTest(entry=entry):
                with self.assertRaises(StateError):
                    self.state.add_meals([entry])
        with self.assertRaises(StateError):
            self.state.add_meals([])
        with self.assertRaises(StateError):
            self.state.add_meals([make_entry()] * 31)
        with self.assertRaises(StateError):
            self.state.add_meals([make_entry()], meal_type="elevenses")
        with self.assertRaises(StateError):                                     # one bad entry rejects the whole request
            self.state.add_meals([make_entry(), make_entry(calories=-5)])
        self.assertEqual(self.state.get_day_meals(), [])

    # --- dates -------------------------------------------------------------
    def test_meals_can_be_logged_for_yesterday_but_not_the_future(self):
        res = self.state.add_meals([make_entry()], day="yesterday")
        self.assertEqual(res["date"], self.clock.days_ago(1))
        self.assertEqual(self.state.get_day_meals(), [])                        # not today
        self.assertEqual(len(self.state.get_day_meals(self.clock.days_ago(1))), 1)
        for bad in (self.clock.days_ago(-1), "2026-13-45", "next tuesday", self.clock.days_ago(401)):
            with self.subTest(day=bad):
                with self.assertRaises(StateError):
                    self.state.add_meals([make_entry()], day=bad)

    def test_a_day_rolls_over_by_itself_when_the_date_changes(self):
        self.state.add_meals([make_entry()])
        self.clock.advance(days=1)
        self.assertEqual(self.state.day_totals()["calories"], 0.0)              # new today is empty
        self.assertEqual(self.state.snapshot()["consumed"]["calories"], 0.0)
        self.assertEqual(self.state.recorded_dates(), [self.clock.days_ago(1)])  # yesterday is kept
        self.assertEqual([r["date"] for r in self.state.snapshot()["weekly_history"]], [self.clock.days_ago(1)])

    # --- removal -----------------------------------------------------------
    def test_remove_by_meal_id_batch_and_food_name(self):
        a = self.state.add_meals([make_entry("egg"), make_entry("banana", "1 piece", 105, 1.3, 27, 0.4)])
        self.state.add_meals([make_entry("egg", "1 piece", 78, 6, 0.6, 5.3)])
        one = self.state.remove_meal(meal_id=a["meal_ids"][1])
        self.assertEqual([m["food"] for m in one["removed"]], ["banana"])
        named = self.state.remove_meal(food_name="EGG")                         # case-insensitive, newest first
        self.assertEqual((named["removed"][0]["calories"], named["remaining_matches"]), (78.0, 1))
        whole = self.state.remove_meal(batch_id=a["batch_id"])
        self.assertEqual(len(whole["removed"]), 1)
        self.assertEqual(self.state.get_day_meals(), [])

    def test_removing_unknown_things_is_a_clean_no_op(self):
        self.state.add_meals([make_entry()])
        for kwargs in ({"meal_id": "m-nope"}, {"batch_id": "b-nope"}, {"food_name": "pizza"}):
            self.assertEqual(self.state.remove_meal(**kwargs)["removed"], [])
        self.assertEqual(len(self.state.get_day_meals()), 1)

    def test_remove_needs_exactly_one_selector(self):
        for kwargs in ({}, {"meal_id": "a", "food_name": "b"}):
            with self.assertRaises(StateError):
                self.state.remove_meal(**kwargs)

    def test_removing_the_last_meal_of_a_day_leaves_no_record_for_that_day(self):
        res = self.state.add_meals([make_entry()], day=self.clock.days_ago(2))
        self.state.remove_meal(meal_id=res["meal_ids"][0])
        self.assertEqual(self.state.recorded_dates(), [])
        self.assertEqual(self.state.history(days=7), [])
        day = self.clock.days_ago(2)
        self.assertNotIn(day, self.state._doc["days"])                                  # not even an empty shell in memory ...
        self.assertNotIn(day, json.loads(self.state._path.read_text(encoding="utf-8"))["days"])   # ... or on disk


# ===========================================================================
class TestHistoryAndTargets(StateCase):

    def test_history_contains_only_days_with_saved_meals(self):
        for ago in (0, 2, 5):
            self.state.add_meals([make_entry(calories=1000.0 + ago)], day=self.clock.days_ago(ago))
        rows = self.state.history(days=7)
        self.assertEqual([r["date"] for r in rows], [self.clock.days_ago(5), self.clock.days_ago(2), self.clock.days_ago(0)])
        self.assertEqual([r["calories"] for r in rows], [1005.0, 1002.0, 1000.0])
        self.assertEqual(len(self.state.history(days=3)), 2)                    # window respected
        self.assertEqual(len(self.state.history(days=30)), 3)

    def test_days_keep_the_targets_that_were_in_effect_when_they_were_saved(self):
        self.state.add_meals([make_entry()], day=self.clock.days_ago(1))
        self.state.set_target("calories", 1800)
        self.state.add_meals([make_entry()])
        by_day = {r["date"]: r["calorie_target"] for r in self.state.history(days=3)}
        self.assertEqual(by_day[self.clock.days_ago(1)], 2300.0)
        self.assertEqual(by_day[self.clock.today], 1800.0)

    def test_set_target_validates_and_reports_old_and_new(self):
        res = self.state.set_target("protein", 120)
        self.assertEqual((res["old"], res["new"], res["source"]), (150.0, 120.0, "user"))
        self.assertEqual(self.state.targets_source()["protein"], "user")
        for kind, value in (("protein", 5), ("protein", 501), ("calories", 499), ("calories", 10001),
                            ("calories", float("nan")), ("calories", True), ("calories", "2300"), ("fibre", 30)):
            with self.subTest(kind=kind, value=value):
                with self.assertRaises(StateError):
                    self.state.set_target(kind, value)
        self.assertEqual(self.state.targets_for()["calories"], 2300.0)

    def test_preferences_are_normalised_and_validated(self):
        prefs = self.state.set_preferences(diet="vegetarian", avoid=[" Paneer ", "paneer", "Soya Chunks"], prefer=["banana"])
        self.assertEqual(prefs, {"diet": "vegetarian", "avoid": ["paneer", "soya chunks"], "prefer": ["banana"]})
        self.assertEqual(self.state.set_preferences(avoid=[])["avoid"], [])
        self.assertEqual(self.state.preferences()["diet"], "vegetarian")        # untouched fields stay
        for kwargs in ({"diet": "keto"}, {"avoid": "paneer"}, {"prefer": ["x"] * 3 + [str(i) for i in range(30)]}):
            with self.assertRaises(StateError):
                self.state.set_preferences(**kwargs)

    def test_recommendations_are_marked_stale_when_the_remaining_budget_changes(self):
        self.state.set_recommendations([{"name": "demo meal"}])
        self.assertFalse(self.state.snapshot()["recommendations_stale"])
        self.state.add_meals([make_entry()])
        self.assertTrue(self.state.snapshot()["recommendations_stale"])
        with self.assertRaises(StateError):
            self.state.set_recommendations([{}] * 11)

    def test_workflow_status_is_validated_recorded_and_reset_by_a_restart(self):
        self.assertEqual(self.state.set_workflow_status("delegating", "meal_intake"), {"previous": "idle", "current": "delegating"})
        self.assertEqual(self.state.snapshot()["workflow_status"], "delegating")
        with self.assertRaises(StateError):
            self.state.set_workflow_status("exploding")
        self.assertEqual([h["to"] for h in self.state.get_workflow()["history"]], ["delegating"])
        self.assertEqual(self.restart().snapshot()["workflow_status"], "idle")

    def test_agent_results_are_per_agent_and_cleared_for_a_new_run(self):
        self.state.update_agent_result("meal_intake", bump_tool_calls=True, status="success", run_id="r1")
        self.state.update_agent_result("meal_intake", bump_tool_calls=True)
        self.assertEqual(self.state.get_agent_results()["meal_intake"]["tool_calls"], 2)
        self.state.reset_run_data()
        self.assertEqual(self.state.get_agent_results(), {})


# ===========================================================================
class TestPersistence(StateCase):

    def test_everything_important_survives_a_restart(self):
        self.state.set_target("calories", 2000)
        self.state.set_preferences(diet="eggetarian", prefer=["banana"])
        self.state.add_meals([make_entry()], request_id="r1", meal_type="lunch")
        self.state.add_meals([make_entry("rice", "100.0 g", 130, 2.7, 28.2, 0.3)], day=self.clock.days_ago(3))
        before = self.state.snapshot()
        after = self.restart().snapshot()
        for key in ("targets", "targets_source", "consumed", "remaining", "meals", "weekly_history", "preferences"):
            self.assertEqual(after[key], before[key], key)
        self.assertTrue(self.state.add_meals([make_entry()], request_id="r1")["duplicate"])   # dedupe survives too

    def test_a_failed_save_rolls_the_in_memory_state_back(self):
        self.state.add_meals([make_entry()])
        on_disk = self.state._path.read_text(encoding="utf-8")
        before = self.state.snapshot()
        with mock.patch.object(self.state, "_persist", side_effect=OSError("disk full")):
            with self.assertRaises(StateUpdateError) as caught:
                self.state.add_meals([make_entry("banana", "1 piece", 105, 1.3, 27, 0.4)], request_id="x")
            self.assertIn("disk full", str(caught.exception))
            with self.assertRaises(StateUpdateError):
                self.state.set_target("calories", 1900)
            with self.assertRaises(StateUpdateError):
                self.state.remove_meal(food_name="egg")
        self.assertEqual(self.state.snapshot(), before)
        self.assertEqual(self.state._path.read_text(encoding="utf-8"), on_disk)
        self.assertFalse(self.state.add_meals([make_entry("banana", "1 piece", 105, 1.3, 27, 0.4)], request_id="x")["duplicate"])

    def test_the_write_is_atomic_and_leaves_no_temp_files(self):
        from nutritrack.state import io as state_io
        self.state.add_meals([make_entry()])
        good = self.state._path.read_text(encoding="utf-8")
        with mock.patch.object(state_io.json, "dump", side_effect=OSError("crash mid-write")):
            with self.assertRaises(StateUpdateError):
                self.state.add_meals([make_entry("banana", "1 piece", 105, 1.3, 27, 0.4)])
        self.assertEqual(self.state._path.read_text(encoding="utf-8"), good)
        self.assertEqual(list(self.state._path.parent.glob(".*.tmp")), [])
        json.loads(good)

    def test_a_corrupt_file_is_quarantined_not_destroyed(self):
        path = self.state._path
        path.write_text('{"schema_version": 1, "days": {"2026-11-12": {"meals": [ {"food"', encoding="utf-8")
        state = self.restart()
        self.assertTrue(any("unreadable" in w for w in state.load_warnings))
        self.assertEqual(state.get_day_meals(), [])
        backups = list(path.parent.glob("demo_user.json.corrupt-*"))
        self.assertEqual(len(backups), 1)
        self.assertIn('"food"', backups[0].read_text(encoding="utf-8"))
        self.assertEqual(state.snapshot()["meta"]["load_warnings"], state.load_warnings)

    def test_a_file_from_a_newer_version_is_refused_and_left_alone(self):
        path = self.state._path
        text = json.dumps({"schema_version": 99, "user_id": "demo_user"})
        path.write_text(text, encoding="utf-8")
        with self.assertRaises(StateUpdateError):
            self.restart()
        self.assertEqual(path.read_text(encoding="utf-8"), text)

    def test_an_older_or_partial_file_is_completed_with_defaults(self):
        path = self.state._path
        path.write_text(json.dumps({"user_id": "demo_user", "days": {}}), encoding="utf-8")
        state = self.restart()
        self.assertEqual(state.snapshot()["preferences"]["diet"], "any")
        # the DEMO user's missing targets fall back to the labelled demo defaults ...
        self.assertEqual((state.targets_for()["calories"], state.targets_source()["calories"]), (2300.0, "demo_default"))
        self.assertIs(state.add_meals([make_entry()])["duplicate"], False)
        # ... but a real user's missing targets are NEVER invented
        other = self.manager.path_for("alice")
        other.parent.mkdir(parents=True, exist_ok=True)
        other.write_text(json.dumps({"user_id": "alice", "days": {}}), encoding="utf-8")
        alice = self.manager.get("alice")
        self.assertEqual(alice.targets_for(), {"calories": None, "protein": None})
        self.assertEqual(alice.targets_source(), {"calories": None, "protein": None})

    def test_concurrent_writers_never_lose_or_corrupt_data(self):
        errors = []

        def worker(n):
            try:
                for i in range(10):
                    self.state.add_meals([make_entry("egg", "1 piece", 78, 6, 0.6, 5.3)], request_id=f"t{n}-{i}")
            except Exception as exc:                                   # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(errors, [])
        self.assertEqual(len(self.state.get_day_meals()), 80)
        self.assertEqual(self.state.day_totals()["calories"], 80 * 78.0)
        self.assertEqual(len(self.restart().get_day_meals()), 80)      # what is on disk matches memory


# ===========================================================================
class TestCA1Import(StateCase):

    CA1_LOG = {
        "current_date": "2026-11-13", "budget": 2200.0,
        "meals": [{"food": "egg", "quantity": "3 pieces", "calories": 234.0, "protein": 18.0, "carbs": 1.8, "fat": 15.9}],
        "history": {
            "2026-11-12": {"budget": 2000.0, "meals": [
                {"food": "rice", "quantity": "200.0 g", "calories": 260.0, "protein": 5.4, "carbs": 56.4, "fat": 0.6},
                {"food": "BROKEN", "quantity": "1", "calories": "lots"}],
                "total_calories": 260.0},
            "2026-11-11": {"budget": 2000.0, "meals": [], "total_calories": 0.0},      # CA1 archived an empty day as 0 kcal
            "not-a-date": {"budget": 1, "meals": [{"food": "x", "quantity": "1", "calories": 1, "protein": 0, "carbs": 0, "fat": 0}]},
        },
    }

    def _write_ca1(self, content=None):
        path = Path(self.tmp) / "daily_log.json"
        path.write_text(json.dumps(self.CA1_LOG if content is None else content), encoding="utf-8")
        return path

    def test_history_and_budget_are_imported_into_a_new_demo_state(self):
        ca1 = self._write_ca1()
        before = ca1.read_bytes()
        state = StateManager(base_dir=self.base / "fresh", clock=self.clock, ca1_log_path=ca1).get("demo_user")
        report = state.snapshot()["meta"]["ca1_import"]
        self.assertEqual((report["imported_days"], report["imported_meals"]), (2, 2))
        self.assertEqual((report["skipped_meals"], report["skipped_empty_days"]), (1, 2))   # BROKEN meal; empty day + bad date
        self.assertEqual(state.recorded_dates(), ["2026-11-12", "2026-11-13"])
        self.assertEqual(state.targets_for("2026-11-12")["calories"], 2000.0)             # that day's own CA1 budget
        self.assertEqual(state.targets_for()["calories"], 2200.0)
        self.assertEqual(state.snapshot()["targets_source"]["calories"], "ca1_import")
        self.assertEqual(state.snapshot()["targets_source"]["protein"], "demo_default")  # CA1 had no protein target
        self.assertEqual(state.day_totals("2026-11-12")["calories"], 260.0)
        self.assertEqual(state.get_day_meals()[0]["source"], "ca1_import")
        self.assertEqual(ca1.read_bytes(), before, "the CA1 file must never be modified")

    def test_empty_ca1_days_are_not_turned_into_zero_calorie_days(self):
        state = StateManager(base_dir=self.base / "fresh", clock=self.clock, ca1_log_path=self._write_ca1()).get("demo_user")
        self.assertNotIn("2026-11-11", state.recorded_dates())
        self.assertEqual(state.history("2026-11-13", 7)[0]["date"], "2026-11-12")

    def test_import_happens_once_only(self):
        ca1 = self._write_ca1()
        mgr = StateManager(base_dir=self.base / "fresh", clock=self.clock, ca1_log_path=ca1)
        mgr.get("demo_user").remove_meal(food_name="egg")
        again = StateManager(base_dir=self.base / "fresh", clock=self.clock, ca1_log_path=ca1).get("demo_user")
        self.assertEqual(again.get_day_meals(), [], "a restart must not re-import deleted CA1 meals")

    def test_other_users_never_get_a_ca1_import(self):
        mgr = StateManager(base_dir=self.base / "fresh", clock=self.clock, ca1_log_path=self._write_ca1())
        self.assertEqual(mgr.get("alice").recorded_dates(), [])

    def test_unreadable_ca1_file_is_reported_and_does_not_block_startup(self):
        ca1 = Path(self.tmp) / "daily_log.json"
        ca1.write_text("{ not json", encoding="utf-8")
        state = StateManager(base_dir=self.base / "fresh", clock=self.clock, ca1_log_path=ca1).get("demo_user")
        self.assertTrue(any("CA1 daily log" in w for w in state.load_warnings))
        self.assertEqual(state.recorded_dates(), [])
        self.assertEqual(ca1.read_text(encoding="utf-8"), "{ not json")

    def test_a_ca1_log_with_the_wrong_shape_imports_nothing_and_changes_nothing(self):
        ca1 = self._write_ca1(["not", "an", "object"])
        state = StateManager(base_dir=self.base / "fresh", clock=self.clock, ca1_log_path=ca1).get("demo_user")
        self.assertTrue(state.load_warnings)
        self.assertEqual(state.targets_for(), DEMO_DEFAULT_TARGETS)


# ===========================================================================
class TestRunLog(StateCase):

    def _append(self, **overrides):
        args = dict(run_id="run-1", agent="Meal Intake Agent", task="log breakfast", tool="parse_meal_entry",
                    status="success", input_summary={"text": "3 eggs"}, output_summary={"items": 1}, user_id="demo_user")
        args.update(overrides)
        return self.run_log.append(**args)

    def test_entries_have_every_field_the_brief_asks_for(self):
        self.assertTrue(self._append(duration_ms=12.345))
        entry = self.run_log.read()[0]
        for key in ("timestamp", "run_id", "agent", "task", "tool", "input_summary", "output_summary",
                    "status", "error"):
            self.assertIn(key, entry)
        self.assertEqual((entry["run_id"], entry["agent"], entry["tool"], entry["status"], entry["error"]),
                         ("run-1", "Meal Intake Agent", "parse_meal_entry", "success", None))
        self.assertEqual((entry["timestamp"], entry["duration_ms"], entry["user_id"]),
                         ("2026-11-13T12:00:00.000", 12.3, "demo_user"))

    def test_failed_entries_carry_error_details(self):
        self._append(status="failed", error={"code": "unknown_food", "message": "idli is not in the database"})
        entry = self.run_log.read()[0]
        self.assertEqual((entry["status"], entry["error"]["code"]), ("failed", "unknown_food"))

    def test_status_must_be_success_or_failed(self):
        with self.assertRaises(ValueError):
            self._append(status="ok")
        self.assertEqual(self.run_log.read(), [])

    def test_the_log_is_append_only(self):
        for i in range(5):
            self._append(run_id=f"run-{i}")
        path = self.run_log.path
        snapshot = path.read_bytes()
        for i in range(5, 12):
            self._append(run_id=f"run-{i}")
        self.assertTrue(path.read_bytes().startswith(snapshot), "existing lines must never change")
        RunLog(path, clock=self.clock).append(run_id="run-new", agent="a", task="t", tool="x", status="success")  # a second writer
        self.assertEqual([e["run_id"] for e in self.run_log.read()][:5], [f"run-{i}" for i in range(5)])
        self.assertEqual(len(self.run_log.read()), 13)
        self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 13, "exactly one line per entry")

    def test_secrets_never_reach_the_file(self):
        token = "ghp_" + "A1b2C3d4E5" * 4
        self._append(input_summary={"api_key": "super-secret-key", "Authorization": "Bearer abcdefghijklmnop",
                                    "note": f"my token is {token}", "password": "hunter2", "food": "3 eggs"},
                     output_summary=f"echo sk-{'x' * 30} and {token}",
                     error={"code": "boom", "message": f"failed with {token}"})
        raw = self.run_log.path.read_text(encoding="utf-8")
        for secret in ("super-secret-key", "abcdefghijklmnop", token, "hunter2", "sk-" + "x" * 30):
            self.assertNotIn(secret, raw)
        self.assertIn("3 eggs", raw)
        self.assertIn("***", raw)

    def test_redact_and_summarize_helpers(self):
        self.assertEqual(redact({"a": {"API_KEY": "x", "ok": 1}, "b": [{"token": "y"}]}),
                         {"a": {"API_KEY": "***", "ok": 1}, "b": [{"token": "***"}]})
        long = summarize("x" * 1000)
        self.assertLess(len(long), 450)
        self.assertIn("+600 chars", long)
        self.assertEqual(summarize(None), "")

    def test_non_ascii_text_round_trips(self):
        self._append(input_summary="2 रोटी और दही")
        self.assertIn("रोटी", self.run_log.read()[0]["input_summary"])

    def test_reader_skips_and_counts_damaged_lines(self):
        self._append(run_id="a")
        with open(self.run_log.path, "a", encoding="utf-8") as fh:
            fh.write("this is not json\n\n[1, 2, 3]\n")
        self._append(run_id="b")
        self.assertEqual([e["run_id"] for e in self.run_log.read()], ["a", "b"])
        self.assertEqual(self.run_log.skipped_lines, 2)

    def test_read_filters(self):
        for run, user in (("r1", "demo_user"), ("r2", "alice"), ("r1", "demo_user"), ("r3", "demo_user")):
            self._append(run_id=run, user_id=user)
        self.assertEqual(len(self.run_log.read(run_id="r1")), 2)
        self.assertEqual(len(self.run_log.read(user_id="alice")), 1)
        self.assertEqual([e["run_id"] for e in self.run_log.read(limit=2)], ["r1", "r3"])
        self.assertEqual(RunLog(Path(self.tmp) / "nothing.jsonl").read(), [])

    def test_concurrent_appends_produce_whole_valid_lines(self):
        def worker(n):
            for i in range(25):
                self._append(run_id=f"t{n}", output_summary=f"item {i}")
        threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(len(self.run_log.read()), 200)
        self.assertEqual(self.run_log.skipped_lines, 0)

    def test_a_broken_log_reports_failure_but_never_raises(self):
        blocker = Path(self.tmp) / "i_am_a_file"
        blocker.write_text("x")
        broken = RunLog(blocker / "agent_runs.jsonl")
        with self.assertLogs("nutritrack.state.run_log", level="ERROR"):
            self.assertFalse(broken.append(run_id="r", agent="a", task="t", tool="x", status="success"))
        self.assertEqual(broken.failures, 1)

    def test_run_ids_are_unique_and_well_formed(self):
        ids = {new_run_id() for _ in range(500)}
        self.assertEqual(len(ids), 500)
        self.assertRegex(new_run_id(self.clock()), r"^run-20261113-120000-[0-9a-f]{10}$")


if __name__ == "__main__":
    unittest.main()
