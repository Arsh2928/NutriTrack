"""Tests for per-user concurrency, cross-process file locking, idempotency, tail log I/O, caching, and API hardening."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from nutritrack.state import RunLog, StateManager
from nutritrack.web import app as web_app


class ConcurrencyIoAndValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = os.path.join(self._tmp.name, "data")
        self.log_dir = os.path.join(self._tmp.name, "logs")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        self._env = patch.dict(
            os.environ,
            {
                "NUTRITRACK_DATA_DIR": self.data_dir,
                "NUTRITRACK_LOG_DIR": self.log_dir,
                "NUTRITRACK_AI_MODE": "offline",
                "GEMINI_API_KEY": "",
                "GOOGLE_API_KEY": "",
                "GROQ_API_KEY": "",
            },
            clear=False,
        )
        self._env.start()
        web_app.reset_ca2_runtime()
        web_app.app.testing = True

    def tearDown(self) -> None:
        web_app.reset_ca2_runtime()
        self._env.stop()
        self._tmp.cleanup()

    def test_01_concurrent_same_user_meal_posts_preserve_every_entry(self) -> None:
        def _post_one(idx: int):
            c = web_app.app.test_client()
            resp = c.post(
                "/api/nutritrack/meals",
                json={"user_id": "conc_user", "food": "egg", "portion": "1 piece", "meal_type": "snack"},
            )
            return resp.status_code, resp.get_json()

        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(_post_one, range(8)))

        for code, body in results:
            self.assertEqual(code, 200)
            self.assertEqual(body["status"], "success")

        st = web_app.app.test_client().get("/api/nutritrack/state?user_id=conc_user").get_json()["state"]
        self.assertEqual(len(st["meals"]), 8)
        self.assertEqual(st["consumed"]["calories"], 624.0)

    def test_02_two_state_managers_reload_and_lock_cleanly(self) -> None:
        sm1 = StateManager(base_dir=self.data_dir, ca1_log_path=None)
        sm2 = StateManager(base_dir=self.data_dir, ca1_log_path=None)

        sm1.get("shared_u").add_meals(
            [{"food": "egg", "quantity_str": "1 piece", "calories": 78.0, "protein": 6.0, "carbs": 0.6, "fat": 5.3}],
            day="2026-11-16",
        )
        # sm2 must see sm1's disk write and merge safely without clobbering it
        sm2.get("shared_u").add_meals(
            [{"food": "banana", "quantity_str": "1 piece", "calories": 105.0, "protein": 1.3, "carbs": 27.0, "fat": 0.4}],
            day="2026-11-16",
        )
        snap = sm1.get("shared_u").snapshot(day="2026-11-16")
        self.assertEqual(len(snap["meals"]), 2)
        self.assertEqual(snap["consumed"]["calories"], 183.0)

    def test_03_chat_and_meal_idempotency_prevents_duplicate_logging(self) -> None:
        c = web_app.app.test_client()
        r1 = c.post(
            "/api/nutritrack/chat",
            json={"user_id": "idem_u", "message": "I ate 2 boiled eggs.", "request_id": "req-chat-001"},
        )
        self.assertEqual(r1.status_code, 200)
        r2 = c.post(
            "/api/nutritrack/chat",
            json={"user_id": "idem_u", "message": "I ate 2 boiled eggs.", "request_id": "req-chat-001"},
        )
        self.assertEqual(r2.status_code, 200)
        self.assertTrue(r2.get_json().get("duplicate"))

        m1 = c.post(
            "/api/nutritrack/meals",
            json={"user_id": "idem_u", "food": "banana", "portion": "1 piece", "request_id": "req-meal-002"},
        )
        m2 = c.post(
            "/api/nutritrack/meals",
            json={"user_id": "idem_u", "food": "banana", "portion": "1 piece", "request_id": "req-meal-002"},
        )
        self.assertEqual(m1.status_code, 200)
        self.assertEqual(m2.status_code, 200)
        self.assertTrue(m2.get_json().get("duplicate"))

        snap = c.get("/api/nutritrack/state?user_id=idem_u").get_json()["state"]
        self.assertEqual(len(snap["meals"]), 2)
        self.assertEqual(snap["consumed"]["calories"], 261.0)

    def test_04_run_log_tail_read_and_size_rotation(self) -> None:
        log_path = Path(self.log_dir) / "tail_test.jsonl"
        rl = RunLog(path=log_path, max_bytes=4096)
        for i in range(60):
            rl.append(
                run_id=f"r-{i}",
                user_id="system" if i % 5 == 0 else "demo_user",
                agent="Meal Intake Agent",
                task=f"log task {i} " + ("x" * 40),
                tool="fast_log_meal",
                status="success",
            )
        # Rotation created .1 backup cleanly
        self.assertTrue(log_path.with_name("tail_test.jsonl.1").exists())

        recent = rl.read(limit=5, user_id="demo_user", exclude_user_ids=("system",))
        self.assertEqual(len(recent), 5)
        self.assertTrue(all(e["user_id"] == "demo_user" for e in recent))
        self.assertEqual(recent[-1]["run_id"], "r-59")

    def test_05_insights_cache_invalidates_after_meal_mutation(self) -> None:
        c = web_app.app.test_client()
        g1 = c.get("/api/nutritrack/insights?user_id=cache_u&days=7").get_json()
        g2 = c.get("/api/nutritrack/insights?user_id=cache_u&days=7").get_json()
        self.assertEqual(g1["trends"]["days_with_data"], 0)
        self.assertEqual(g2["trends"]["days_with_data"], 0)

        c.post("/api/nutritrack/meals", json={"user_id": "cache_u", "food": "egg", "portion": "2 pieces"})
        g3 = c.get("/api/nutritrack/insights?user_id=cache_u&days=7").get_json()
        self.assertEqual(g3["trends"]["days_with_data"], 1)
        self.assertEqual(g3["trends"]["averages"]["calories"], 156.0)

    def test_06_api_rejects_malformed_json_arrays_bad_users_and_adds_security_headers(self) -> None:
        c = web_app.app.test_client()
        post_routes = [
            "/api/chat",
            "/api/nutritrack/ai/mode",
            "/api/nutritrack/chat",
            "/api/nutritrack/goals",
            "/api/nutritrack/preferences",
            "/api/nutritrack/recommendations",
            "/api/nutritrack/meals",
            "/api/nutritrack/profile",
            "/api/nutritrack/water",
            "/api/nutritrack/custom-food",
        ]
        for route in post_routes:
            with self.subTest(route=route):
                r_arr = c.post(route, json=[1, 2, 3])
                self.assertEqual(r_arr.status_code, 400)
                r_bad = c.post(route, data="{not json", content_type="application/json")
                self.assertEqual(r_bad.status_code, 400)

        self.assertEqual(c.get("/api/nutritrack/state?user_id=../evil").status_code, 400)
        self.assertEqual(c.post("/api/nutritrack/chat", json={"message": "a" * 1005}).status_code, 400)
        self.assertEqual(c.post("/api/nutritrack/water", json={"liters": -1}).status_code, 400)
        self.assertEqual(c.post("/api/nutritrack/water", json={"date": "bad-date", "liters": 1.0}).status_code, 400)

        r404 = c.get("/api/nutritrack/nonexistent-route")
        self.assertEqual(r404.status_code, 404)
        self.assertEqual(r404.get_json()["error_code"], "not_found")
        self.assertEqual(r404.headers.get("X-Content-Type-Options"), "nosniff")
        self.assertEqual(r404.headers.get("X-Frame-Options"), "SAMEORIGIN")


if __name__ == "__main__":
    unittest.main()
