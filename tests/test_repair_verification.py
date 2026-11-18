"""Comprehensive regression suite covering all repaired Master Audit bugs."""

from __future__ import annotations

from datetime import date, timedelta
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nutritrack.agents import NutriTrackOrchestrator
from nutritrack.ai.config import AIProviderConfig, ai_mode_scope
from nutritrack.core import external_nutrition
from nutritrack.core.quantity import parse_item_text
from nutritrack.web import app as web_app


class MasterRepairVerificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmp.name) / "data"
        self.logs_dir = Path(self._tmp.name) / "logs"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._env_patch = patch.dict(
            os.environ,
            {
                "NUTRITRACK_DATA_DIR": str(self.data_dir),
                "NUTRITRACK_LOG_DIR": str(self.logs_dir),
                "NUTRITRACK_CUSTOM_FOODS": str(self.data_dir / "custom_foods.json"),
            },
            clear=False,
        )
        self._env_patch.start()
        web_app.app.testing = True
        web_app.app.config["REQUIRE_AUTH"] = False
        web_app.reset_ca2_runtime()
        self.client = web_app.app.test_client()

    def tearDown(self) -> None:
        web_app.reset_ca2_runtime()
        self._env_patch.stop()
        self._tmp.cleanup()

    def test_manual_three_modes_and_offline_network_isolation(self) -> None:
        # Even if all cloud keys and legacy modes are present, default first-run mode MUST be 'offline'
        with patch.dict(
            os.environ,
            {
                "NUTRITRACK_DATA_DIR": str(self.data_dir),
                "GEMINI_API_KEY": "AIzaSyNeverAutoSelectGemini123456",
                "GROQ_API_KEY": "gsk_NeverAutoSelectGroq123456",
                "GITHUB_TOKEN": "ghp_LegacyTokenIgnored123456",
            },
            clear=True,
        ):
            cfg = AIProviderConfig.from_env(persisted_dir=self.data_dir)
            self.assertEqual(cfg.selected_mode, "offline")

        # Switching UI mode to offline immediately disables external nutrition even when env has gemini + ONLINE_LOOKUP=1
        with patch.dict(
            os.environ,
            {
                "NUTRITRACK_AI_MODE": "gemini",
                "NUTRITRACK_ONLINE_LOOKUP": "1",
                "GITHUB_TOKEN": "ghp_LegacyTokenIgnored123456",
            },
            clear=False,
        ):
            web_app.reset_ca2_runtime()
            res_set = self.client.post("/api/nutritrack/ai/mode", json={"mode": "offline"})
            self.assertEqual(res_set.status_code, 200)
            self.assertEqual(res_set.get_json()["selected_mode"], "offline")

            # With outbound networking completely poisoned, offline operations must succeed with ZERO network attempts
            with patch("socket.create_connection", side_effect=AssertionError("Network socket called in offline mode!")), \
                 patch("urllib.request.urlopen", side_effect=AssertionError("urlopen called in offline mode!")):
                self.assertFalse(external_nutrition.external_enabled())
                r_prev = self.client.get("/api/nutritrack/foods/preview?food=dragonfruit&portion=100+g")
                self.assertEqual(r_prev.status_code, 200)
                self.assertFalse(r_prev.get_json()["ok"])
                r_chat = self.client.post("/api/nutritrack/chat", json={"message": "Log 2 boiled eggs"})
                self.assertEqual(r_chat.status_code, 200)
                self.assertEqual(r_chat.get_json()["status"], "success")

        # Legacy /api/chat never uses GitHub Models and rejects legacy modes explicitly
        r_bad_legacy = self.client.post("/api/chat", json={"message": "hi", "mode": "github"})
        self.assertEqual(r_bad_legacy.status_code, 400)
        r_legacy_ok = self.client.post("/api/chat", json={"message": "Add 2 boiled eggs"})
        self.assertEqual(r_legacy_ok.status_code, 200)
        self.assertEqual(r_legacy_ok.get_json()["agent_mode"], "rule_based")

    def test_mode_persistence_failure_returns_500_without_silent_memory_mutation(self) -> None:
        self.assertEqual(self.client.get("/api/nutritrack/ai/mode").get_json()["selected_mode"], "offline")
        with patch("nutritrack.ai.provider_manager.save_persisted_mode", side_effect=OSError("read-only disk")):
            res = self.client.post("/api/nutritrack/ai/mode", json={"mode": "gemini"})
            self.assertEqual(res.status_code, 500)
            self.assertEqual(res.get_json()["error_code"], "config_persist_failed")
        self.assertEqual(self.client.get("/api/nutritrack/ai/mode").get_json()["selected_mode"], "offline")

    def test_fast_chat_routing_and_quantity_date_parsing_repairs(self) -> None:
        # 7up parsing + fast deterministic chat path (zero NutritionManagerAgent delegation)
        parsed_7up = parse_item_text("1 can of 7up")
        self.assertEqual(parsed_7up.value, 1.0)
        self.assertEqual(parsed_7up.food_text, "7up")
        self.assertEqual(parsed_7up.problems, [])

        # Fractions and dates in arbitrary positions
        parsed_frac = parse_item_text("1 1/2 cups rice")
        self.assertEqual(parsed_frac.value, 1.5)
        self.assertEqual(parsed_frac.unit, "cup")

        with patch.object(NutriTrackOrchestrator, "run_turn", side_effect=AssertionError("Simple turn should not invoke MAF orchestrator!")):
            # Unregistered '7up' resolves cleanly to 'unknown_food' (never 'ambiguous_quantity') without invoking MAF
            r_7up_unk = self.client.post("/api/nutritrack/chat", json={"message": "Log 1 can of 7up"})
            self.assertEqual(r_7up_unk.status_code, 200)
            b_7up_unk = r_7up_unk.get_json()
            self.assertEqual(b_7up_unk["error_code"], "unknown_food")
            self.assertEqual(b_7up_unk["execution_mode"], "fast_deterministic")

            # After registering custom nutrition for '7up', logging 'Log 330 ml 7up' succeeds fast!
            r_cf = self.client.post(
                "/api/nutritrack/custom-food",
                json={"name": "7up", "calories": 140, "protein": 0, "carbs": 38, "fat": 0, "unit": "ml", "base_amount": 330, "serving_size": "1 can (330 ml)"},
            )
            self.assertEqual(r_cf.status_code, 200)
            r_7up = self.client.post("/api/nutritrack/chat", json={"message": "Log 330 ml 7up"})
            self.assertEqual(r_7up.status_code, 200)
            b_7up = r_7up.get_json()
            self.assertEqual(b_7up["status"], "success")
            self.assertEqual(b_7up["execution_mode"], "fast_deterministic")

            # Explicit yesterday date placed anywhere
            r_yday = self.client.post("/api/nutritrack/chat", json={"message": "Yesterday I had 2 boiled eggs for breakfast"})
            self.assertEqual(r_yday.status_code, 200)
            self.assertEqual(r_yday.get_json()["status"], "success")
            yday_iso = (date.today() - timedelta(days=1)).isoformat()
            self.assertEqual(r_yday.get_json()["date"], yday_iso)

            # Yesterday's possessive + suffix position
            r_yday2 = self.client.post("/api/nutritrack/chat", json={"message": "Log 1 banana for yesterday's dinner"})
            self.assertEqual(r_yday2.status_code, 200)
            self.assertEqual(r_yday2.get_json()["date"], yday_iso)

            # Simple daily & weekly analytics locally
            r_q1 = self.client.post("/api/nutritrack/chat", json={"message": "What did I eat today?"})
            self.assertEqual(r_q1.status_code, 200)
            self.assertEqual(r_q1.get_json()["execution_mode"], "fast_deterministic")

            r_q2 = self.client.post("/api/nutritrack/chat", json={"message": "Weekly calories summary"})
            self.assertEqual(r_q2.status_code, 200)
            self.assertEqual(r_q2.get_json()["execution_mode"], "fast_deterministic")

            # Remove last meal deterministically
            r_rem = self.client.post("/api/nutritrack/chat", json={"message": "Remove my last meal"})
            self.assertEqual(r_rem.status_code, 200)
            self.assertTrue(r_rem.get_json().get("removed"))
            st_now = self.client.get("/api/nutritrack/state").get_json()["state"]
            self.assertEqual(st_now["meals"], [])

            # Invalid calendar date rejected cleanly without crashing
            r_bad_date = self.client.post("/api/nutritrack/chat", json={"message": "Log 2 eggs on 2026-02-31"})
            self.assertIn(r_bad_date.get_json()["error_code"], ("invalid_date", "invalid_input"))

    def test_durable_operations_replay_conflict_and_restart(self) -> None:
        r1 = self.client.post(
            "/api/nutritrack/chat",
            json={"user_id": "op_user", "request_id": "dur-101", "message": "I ate 2 bananas"},
        )
        self.assertEqual(r1.status_code, 200)
        self.assertEqual(len(r1.get_json()["state"]["meals"]), 1)
        self.assertEqual(r1.get_json()["state"]["consumed"]["calories"], 210.0)

        # Operation status endpoint reports completion
        op_res = self.client.get("/api/nutritrack/operations/dur-101?user_id=op_user")
        self.assertEqual(op_res.status_code, 200)
        self.assertEqual(op_res.get_json()["operation_status"], "succeeded")

        # Simulate full process restart -> same request_id still replays cleanly
        web_app.reset_ca2_runtime()
        r_dup = self.client.post(
            "/api/nutritrack/chat",
            json={"user_id": "op_user", "request_id": "dur-101", "message": "I ate 2 bananas"},
        )
        self.assertEqual(r_dup.status_code, 200)
        self.assertTrue(r_dup.get_json().get("duplicate"))
        self.assertEqual(len(self.client.get("/api/nutritrack/state?user_id=op_user").get_json()["state"]["meals"]), 1)

        # Different payload reusing same request_id returns 409 Conflict
        r_conflict = self.client.post(
            "/api/nutritrack/chat",
            json={"user_id": "op_user", "request_id": "dur-101", "message": "I ate 5 eggs"},
        )
        self.assertEqual(r_conflict.status_code, 409)

    def test_profile_water_limits_rollback_and_http_validation(self) -> None:
        # Malformed water_logs rejected BEFORE touching disk
        self.assertEqual(
            self.client.post("/api/nutritrack/profile", json={"water_logs": "corrupt"}).status_code,
            400,
        )
        self.assertEqual(
            self.client.post("/api/nutritrack/profile", json={"water_logs": {"2026-99-99": 2.0}}).status_code,
            400,
        )
        self.assertEqual(
            self.client.post("/api/nutritrack/profile", json={"water_logs": {date.today().isoformat(): "abc"}}).status_code,
            400,
        )
        # Subsequent GET /state still healthy 200
        self.assertEqual(self.client.get("/api/nutritrack/state").status_code, 200)

        # Atomic rollback: valid weight_kg paired with invalid calories=99999 must not persist weight_kg=88
        r_fail = self.client.post("/api/nutritrack/profile", json={"weight_kg": 88.0, "calories": 99999})
        self.assertEqual(r_fail.status_code, 400)
        prof_after = self.client.get("/api/nutritrack/profile").get_json()["profile"]
        self.assertNotEqual(prof_after["weight_kg"], 88.0)

        # Water limit consistency: 12.0 L accepted and preserved without clamping to 10.0 L; 15.1 L rejected
        r_w12 = self.client.post("/api/nutritrack/water", json={"liters": 12.0})
        self.assertEqual(r_w12.status_code, 200)
        self.assertEqual(r_w12.get_json()["water_today_liters"], 12.0)
        self.assertEqual(self.client.post("/api/nutritrack/water", json={"liters": 15.1}).status_code, 400)

        # Payload size (413), wrong content-type (415), invalid meal_type (400)
        r_413 = self.client.post(
            "/api/nutritrack/chat",
            data=json.dumps({"message": "a" * 300_000}),
            content_type="application/json",
        )
        self.assertEqual(r_413.status_code, 413)
        r_415 = self.client.post("/api/nutritrack/chat", data="message=hello", content_type="text/plain")
        self.assertEqual(r_415.status_code, 415)
        r_mt = self.client.post("/api/nutritrack/meals", json={"food": "egg", "portion": "1 piece", "meal_type": "midnight_binge"})
        self.assertEqual(r_mt.status_code, 400)

    def test_insights_cache_invalidates_across_midnight_rollover(self) -> None:
        from datetime import datetime as _real_dt

        class _FakeDateDay1(date):
            @classmethod
            def today(cls) -> "_FakeDateDay1":
                return cls(2026, 11, 16)

        class _FakeDtDay1(_real_dt):
            @classmethod
            def now(cls, tz=None):
                return cls(2026, 11, 16, 23, 59, 0)

        class _FakeDateDay2(date):
            @classmethod
            def today(cls) -> "_FakeDateDay2":
                return cls(2026, 11, 17)

        class _FakeDtDay2(_real_dt):
            @classmethod
            def now(cls, tz=None):
                return cls(2026, 11, 17, 0, 1, 0)

        sm = web_app._get_ca2_runtime()[0]
        st_roll = sm.get("roll_u")
        with patch.object(sm, "_clock", _FakeDtDay1.now), \
             patch.object(st_roll, "_clock", _FakeDtDay1.now), \
             patch("nutritrack.web.app.date", _FakeDateDay1), \
             patch("nutritrack.tools.analysis_tools.date", _FakeDateDay1):
            r1 = self.client.get("/api/nutritrack/insights?user_id=roll_u&days=7")
            self.assertEqual(r1.status_code, 200)
            self.assertEqual(r1.get_json()["trends"]["period"]["end"], "2026-11-16")

        with patch.object(sm, "_clock", _FakeDtDay2.now), \
             patch.object(st_roll, "_clock", _FakeDtDay2.now), \
             patch("nutritrack.web.app.date", _FakeDateDay2), \
             patch("nutritrack.tools.analysis_tools.date", _FakeDateDay2):
            r2 = self.client.get("/api/nutritrack/insights?user_id=roll_u&days=7")
            self.assertEqual(r2.status_code, 200)
            self.assertEqual(r2.get_json()["trends"]["period"]["end"], "2026-11-17")

    def test_auth_csrf_idor_and_per_user_custom_food_isolation(self) -> None:
        # Register Alice and Bob in separate clients
        c_alice = web_app.app.test_client()
        c_bob = web_app.app.test_client()
        c_anon = web_app.app.test_client()

        reg_a = c_alice.post("/api/nutritrack/auth/register", json={"user_id": "alice", "password": "alice-password-123"})
        self.assertEqual(reg_a.status_code, 201)
        alice_csrf = reg_a.get_json()["csrf_token"]

        reg_b = c_bob.post("/api/nutritrack/auth/register", json={"user_id": "bob", "password": "bob-password-123"})
        self.assertEqual(reg_b.status_code, 201)
        bob_csrf = reg_b.get_json()["csrf_token"]

        # Unauthenticated requests must be rejected with 401 once accounts exist
        self.assertEqual(c_anon.get("/api/nutritrack/state?user_id=alice").status_code, 401)
        self.assertEqual(c_anon.post("/api/nutritrack/chat", json={"message": "hi"}).status_code, 401)

        # Cookie-authenticated POST without CSRF token must return 403
        no_csrf = c_alice.post("/api/nutritrack/custom-food", json={"name": "alice_bar", "calories": 210, "protein": 20})
        self.assertEqual(no_csrf.status_code, 403)

        # Valid CSRF creates Alice's custom food
        cf_a = c_alice.post(
            "/api/nutritrack/custom-food",
            headers={"X-CSRF-Token": alice_csrf},
            json={"name": "alice_secret_shake", "calories": 320, "protein": 30},
        )
        self.assertEqual(cf_a.status_code, 200)

        # Alice attempting to read/mutate Bob's resources returns 403 IDOR rejection
        self.assertEqual(c_alice.get("/api/nutritrack/state?user_id=bob").status_code, 403)
        self.assertEqual(
            c_alice.post(
                "/api/nutritrack/water",
                headers={"X-CSRF-Token": alice_csrf},
                json={"user_id": "bob", "liters": 2.0},
            ).status_code,
            403,
        )

        # Bob cannot see, preview, or delete Alice's custom food
        bob_foods = c_bob.get("/api/nutritrack/foods").get_json()["foods"]
        self.assertFalse(any(f["name"] == "alice_secret_shake" for f in bob_foods))
        bob_del = c_bob.delete(
            "/api/nutritrack/custom-food/alice_secret_shake",
            headers={"X-CSRF-Token": bob_csrf},
        )
        self.assertEqual(bob_del.status_code, 404)

        # Alice can delete her own custom food cleanly
        alice_del = c_alice.delete(
            "/api/nutritrack/custom-food/alice_secret_shake",
            headers={"X-CSRF-Token": alice_csrf},
        )
        self.assertEqual(alice_del.status_code, 200)

    def test_frontend_js_syntax_and_security_invariants(self) -> None:
        html = (PROJECT_ROOT / "nutritrack" / "web" / "static" / "index.html").read_text(encoding="utf-8")
        self.assertIn('id="target-water"', html)
        self.assertIn('max="15"', html)
        self.assertIn("getStableRequestId", html)
        self.assertIn("/api/nutritrack/operations/", html)
        self.assertIn("apiFetch", html)
        self.assertNotIn("innerHTML", html)
        self.assertNotIn("outerHTML", html)
        self.assertNotIn("document.write", html)

        scripts = re.findall(r"<script>([\s\S]*?)</script>", html)
        self.assertEqual(len(scripts), 1)
        js_code = scripts[0]

        # Run node --check if Node.js is available on PATH
        node_bin = None
        for cand in ("node", "node.exe"):
            try:
                chk = subprocess.run([cand, "--version"], capture_output=True, text=True, timeout=5)
                if chk.returncode == 0:
                    node_bin = cand
                    break
            except Exception:
                continue
        if node_bin:
            js_tmp = Path(self._tmp.name) / "app.js"
            js_tmp.write_text(js_code, encoding="utf-8")
            syn = subprocess.run([node_bin, "--check", str(js_tmp)], capture_output=True, text=True, timeout=10)
            self.assertEqual(syn.returncode, 0, f"JS syntax check failed:\n{syn.stderr}")


if __name__ == "__main__":
    unittest.main()
