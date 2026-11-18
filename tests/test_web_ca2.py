"""Tests for Phase 6 & 7: NutriTrack CA2 Flask multi-agent APIs, insights, recommendations, and UI."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from nutritrack.web import app as web_app


class WebCA2EndpointsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._data_dir = Path(self._tmp.name) / "data"
        self._logs_dir = Path(self._tmp.name) / "logs"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        self._env_patch = patch.dict(
            os.environ,
            {
                "NUTRITRACK_DATA_DIR": str(self._data_dir),
                "NUTRITRACK_LOG_DIR": str(self._logs_dir),
                "NUTRITRACK_CUSTOM_FOODS": str(self._data_dir / "custom_foods.json"),
            },
            clear=False,
        )
        self._env_patch.start()
        web_app.reset_ca2_runtime()
        self.client = web_app.app.test_client()

    def tearDown(self) -> None:
        web_app.reset_ca2_runtime()
        self._env_patch.stop()
        self._tmp.cleanup()

    def test_frontend_renders_cleanly_without_unsafe_innerhtml(self) -> None:
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn("Meal Calorie Helper", html)
        self.assertIn("Agent Activity Monitor", html)
        self.assertIn("Weekly Nutrition Insights", html)
        self.assertIn("Smart Meal Recommendations", html)
        # Verify all 6 Google Stitch screens are present
        for screen in ("dashboard", "food-diary", "recommendations", "weekly-progress", "agents", "settings"):
            self.assertIn(f'data-screen="{screen}"', html)
        self.assertNotIn("innerHTML", html)

    def test_mode_and_users_endpoints(self) -> None:
        mode = self.client.get("/api/nutritrack/mode").get_json()
        self.assertIn("mode", mode)
        self.assertNotIn("api_key", mode)
        users = self.client.get("/api/nutritrack/users").get_json()
        self.assertIn("demo_user", users["users"])

    def test_all_five_scenarios_over_http_and_run_log(self) -> None:
        # 1. Log meals
        r1 = self.client.post("/api/nutritrack/chat", json={"message": "I ate 3 boiled eggs and 2 bananas."})
        self.assertEqual(r1.status_code, 200)
        b1 = r1.get_json()
        self.assertEqual(b1["status"], "success")
        self.assertEqual(b1["workflow_status"], "completed")
        self.assertEqual(b1["activated_specialists"], ["meal_intake"])
        self.assertEqual(b1["execution_mode"], "fast_deterministic")
        self.assertEqual(b1["state"]["consumed"]["calories"], 444.0)

        # 2. Check today's totals
        r2 = self.client.post("/api/nutritrack/chat", json={"message": "How many calories and protein did I eat today?"})
        b2 = r2.get_json()
        self.assertEqual(b2["activated_specialists"], ["nutrition_analysis"])
        self.assertIn("444", b2["response"])

        # 3. Update goals via /api/nutritrack/goals
        r3 = self.client.post("/api/nutritrack/goals", json={"calories": 2100, "protein": 140})
        b3 = r3.get_json()
        self.assertEqual(b3["status"], "success")
        self.assertEqual(b3["activated_specialists"], ["goal_budget"])
        self.assertEqual(b3["state"]["targets"], {"calories": 2100.0, "protein": 140.0})

        # 4. Recommendations via /api/nutritrack/recommendations
        r4 = self.client.post("/api/nutritrack/recommendations", json={})
        b4 = r4.get_json()
        self.assertEqual(b4["status"], "success")
        self.assertEqual(b4["activated_specialists"], ["goal_budget", "meal_recommendation"])
        self.assertTrue(b4["state"]["recommendations"])

        # 5. Weekly insights via /api/nutritrack/insights
        r5 = self.client.get("/api/nutritrack/insights?days=7")
        b5 = r5.get_json()
        self.assertEqual(r5.status_code, 200)
        self.assertEqual(b5["status"], "success")
        self.assertEqual(b5["trends"]["days_with_data"], 1)
        self.assertEqual(b5["trends"]["averages"]["calories"], 444.0)

        # Append-only run log captured runs cleanly
        runs = self.client.get("/api/nutritrack/runs?limit=50").get_json()
        self.assertGreater(runs["count"], 5)

    def test_delete_meal_preferences_custom_food_and_user_isolation(self) -> None:
        # Log on user alice
        add_res = self.client.post(
            "/api/nutritrack/chat",
            json={"user_id": "alice", "message": "I ate 2 bananas."},
        ).get_json()
        meal_id = add_res["state"]["meals"][0]["meal_id"]

        # Bob stays completely isolated
        bob_state = self.client.get("/api/nutritrack/state?user_id=bob").get_json()
        self.assertEqual(bob_state["state"]["consumed"]["calories"], 0)
        self.assertEqual(bob_state["state"]["meals"], [])

        # Preferences update works
        pref_resp = self.client.post(
            "/api/nutritrack/preferences",
            json={"user_id": "alice", "diet": "vegetarian", "avoid": ["peanut"], "prefer": ["oats"]},
        )
        self.assertEqual(pref_resp.status_code, 200)
        self.assertEqual(pref_resp.get_json()["preferences"]["diet"], "vegetarian")

        # Remove Alice's meal via DELETE endpoint
        del_resp = self.client.delete(f"/api/nutritrack/meals/{meal_id}?user_id=alice")
        self.assertEqual(del_resp.status_code, 200)
        self.assertEqual(del_resp.get_json()["state"]["meals"], [])

        # Custom food registration & immediate log
        cf_resp = self.client.post(
            "/api/nutritrack/custom-food",
            json={
                "user_id": "alice",
                "name": "quinoa bowl",
                "calories": 380,
                "protein": 14,
                "carbs": 52,
                "fat": 12,
                "unit": "piece",
                "serving_size": "1 bowl",
                "log_immediately": True,
            },
        )
        self.assertEqual(cf_resp.status_code, 200)
        cf_body = cf_resp.get_json()
        self.assertEqual(cf_body["status"], "success")
        self.assertEqual(cf_body["logged_turn"]["state"]["consumed"]["calories"], 380.0)

    def test_invalid_payloads_return_400(self) -> None:
        self.assertEqual(self.client.post("/api/nutritrack/chat", json={"message": "   "}).status_code, 400)
        self.assertEqual(self.client.post("/api/nutritrack/goals", json={"calories": -100}).status_code, 400)
        self.assertEqual(self.client.post("/api/nutritrack/goals", json={"calories": "abc"}).status_code, 400)
        self.assertEqual(self.client.post("/api/nutritrack/custom-food", json={"name": "test"}).status_code, 400)
        self.assertEqual(self.client.get("/api/nutritrack/insights?days=bad").status_code, 400)

    def test_stitch_food_diary_edit_preview_water_and_profile_endpoints(self) -> None:
        # 1. Food catalog and live preview
        foods_res = self.client.get("/api/nutritrack/foods?q=egg").get_json()
        self.assertEqual(foods_res["status"], "success")
        self.assertGreater(foods_res["count"], 0)

        prev_ok = self.client.get("/api/nutritrack/foods/preview?food=boiled+egg&portion=2+pieces").get_json()
        self.assertTrue(prev_ok["ok"])
        self.assertEqual(prev_ok["calories"], 156.0)

        prev_unk = self.client.get("/api/nutritrack/foods/preview?food=dragonfruit&portion=1+piece").get_json()
        self.assertFalse(prev_unk["ok"])
        self.assertEqual(prev_unk["status"], "unresolved")

        # 2. Quick Add Meal with "snacks" category normalized to "snack"
        add_res = self.client.post(
            "/api/nutritrack/meals",
            json={"food": "banana", "portion": "2 pieces", "meal_type": "snacks"},
        ).get_json()
        self.assertEqual(add_res["status"], "success")
        meals = add_res["state"]["meals"]
        self.assertEqual(len(meals), 1)
        self.assertEqual(meals[0]["meal_type"], "snack")
        self.assertEqual(add_res["state"]["consumed"]["calories"], 210.0)
        meal_id = meals[0]["meal_id"]

        # 3. Invalid edit must NOT delete existing meal
        bad_edit = self.client.put(
            f"/api/nutritrack/meals/{meal_id}",
            json={"food": "unknown_mystery_food", "portion": "1 piece"},
        )
        self.assertEqual(bad_edit.status_code, 400)
        st_after_bad = self.client.get("/api/nutritrack/state").get_json()
        self.assertEqual(len(st_after_bad["state"]["meals"]), 1)

        # 4. Valid edit updates meal and recomputes totals in Python
        good_edit = self.client.put(
            f"/api/nutritrack/meals/{meal_id}",
            json={"food": "boiled egg", "portion": "3 pieces", "meal_type": "breakfast"},
        )
        self.assertEqual(good_edit.status_code, 200)
        edit_body = good_edit.get_json()
        self.assertEqual(edit_body["state"]["consumed"]["calories"], 234.0)
        self.assertEqual(edit_body["state"]["meals"][0]["meal_type"], "breakfast")

        # 5. Water hydration logging & Settings profile save
        w_res = self.client.post("/api/nutritrack/water", json={"delta_liters": 0.5}).get_json()
        self.assertEqual(w_res["status"], "success")
        self.assertEqual(w_res["water_today_liters"], 0.5)

        p_res = self.client.post(
            "/api/nutritrack/profile",
            json={
                "name": "Arshdeep Singh",
                "calories": 2250,
                "protein": 155,
                "carbs_target": 240,
                "fat_target": 70,
                "water_target_liters": 3.0,
                "diet": "vegetarian",
                "avoid": ["peanut"],
            },
        )
        self.assertEqual(p_res.status_code, 200)
        p_body = p_res.get_json()
        self.assertEqual(p_body["state"]["targets"], {"calories": 2250.0, "protein": 155.0})
        self.assertEqual(p_body["state"]["preferences"]["diet"], "vegetarian")
        self.assertEqual(p_body["profile"]["water_target_liters"], 3.0)

        # 6. Recommendations endpoint populates multiple validated options (Featured + Alternatives)
        rec_res = self.client.post("/api/nutritrack/recommendations", json={"filter": "high-protein"}).get_json()
        self.assertEqual(rec_res["status"], "success")
        self.assertGreaterEqual(len(rec_res["state"]["recommendations"]), 2)

    def test_external_food_api_lookup_and_multi_unit_quick_add(self) -> None:
        from nutritrack.ca1.food_database import FOOD_DB

        # 1. Verify Quick Add UI exposes quantity input and multi-unit measurement selector
        html = self.client.get("/").get_data(as_text=True)
        self.assertIn('id="diary-qty-value"', html)
        self.assertIn('id="diary-qty-unit"', html)
        for u in ("serving", "piece", "g", "kg", "ml", "l", "cup", "bowl", "tbsp", "tsp", "slice", "scoop"):
            self.assertIn(f'value="{u}"', html)

        # 2. Verify multi-unit Quick Add conversions on built-in foods
        # (a) "1 serving" on a weight-based food (paneer = 100 g = 265 kcal, 18.3 g protein)
        prev_paneer = self.client.get("/api/nutritrack/foods/preview?food=paneer&portion=1+serving").get_json()
        self.assertTrue(prev_paneer["ok"])
        self.assertEqual((prev_paneer["calories"], prev_paneer["protein"]), (265.0, 18.3))

        # (b) "150 g" on a piece-based food (boiled egg ~50 g/egg -> 3 eggs = 234 kcal, 18 g protein)
        prev_egg_g = self.client.get("/api/nutritrack/foods/preview?food=boiled+egg&portion=150+g").get_json()
        self.assertTrue(prev_egg_g["ok"])
        self.assertEqual((prev_egg_g["calories"], prev_egg_g["protein"]), (234.0, 18.0))

        # (c) "1 cup" on rice (200 g cooked rice = 260 kcal) and "1 bowl" on dal (200 g = 232 kcal)
        prev_rice_cup = self.client.get("/api/nutritrack/foods/preview?food=rice&portion=1+cup").get_json()
        self.assertTrue(prev_rice_cup["ok"])
        self.assertEqual(prev_rice_cup["calories"], 260.0)

        prev_dal_bowl = self.client.get("/api/nutritrack/foods/preview?food=dal&portion=1+bowl").get_json()
        self.assertTrue(prev_dal_bowl["ok"])
        self.assertEqual(prev_dal_bowl["calories"], 232.0)

        # 3. Unknown food defaults to asking user unless NUTRITRACK_ONLINE_LOOKUP=1 is opted in
        FOOD_DB.pop("avocado", None)
        prev_default = self.client.get("/api/nutritrack/foods/preview?food=avocado&portion=1+piece").get_json()
        self.assertFalse(prev_default["ok"])
        self.assertEqual(prev_default["reason"], "unknown_food")

        # When explicitly opted in, external lookup returns labelled values without silently saving to FOOD_DB
        fake_avocado = {
            "name": "Avocados, raw",
            "calories": 160.0,
            "protein": 2.0,
            "carbs": 8.5,
            "fat": 14.7,
            "portions": {
                "serving": 150.0,
                "piece": 150.0,
                "cup": 230.0,
                "bowl": 200.0,
                "tbsp": 15.0,
                "tsp": 5.0,
                "slice": 30.0,
                "scoop": 30.0,
            },
            "source": "USDA FoodData Central",
            "provenance": "usda_fdc",
        }
        self.addCleanup(FOOD_DB.pop, "avocado", None)
        self.client.post("/api/nutritrack/ai/mode", json={"mode": "gemini"})
        with patch.dict(os.environ, {"NUTRITRACK_ONLINE_LOOKUP": "1"}):
            os.environ.pop("NUTRITRACK_DISABLE_EXTERNAL_FOOD", None)
            with patch("nutritrack.core.nutrition.lookup_external_food", return_value=fake_avocado):
                # Preview 1 piece (150 g) of avocado
                prev_avo = self.client.get("/api/nutritrack/foods/preview?food=avocado&portion=1+piece").get_json()
                self.assertTrue(prev_avo["ok"])
                self.assertTrue(prev_avo["external"])
                self.assertEqual((prev_avo["calories"], prev_avo["protein"]), (240.0, 3.0))
                self.assertEqual(prev_avo["provenance"], "usda_fdc")

                # Log 50 g of avocado via Quick Add endpoint -> logs meal cleanly, never silently saves FOOD_DB
                add_avo = self.client.post(
                    "/api/nutritrack/meals",
                    json={"food": "avocado", "quantity": 50, "unit": "g", "meal_type": "breakfast"},
                ).get_json()
                self.assertEqual(add_avo["status"], "success")
                self.assertEqual(add_avo["state"]["consumed"]["calories"], 80.0)
                self.assertEqual(add_avo["state"]["consumed"]["protein"], 1.0)
                self.assertTrue(add_avo["state"]["meals"][-1]["external"])
                self.assertEqual(add_avo["state"]["meals"][-1]["external_source"], "USDA FoodData Central")
                self.assertNotIn("avocado", FOOD_DB)


if __name__ == "__main__":
    unittest.main()
