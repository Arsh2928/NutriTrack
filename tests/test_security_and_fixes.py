"""Regression tests for security redaction, exclusions, and CA2 audit fixes."""

from __future__ import annotations

import json
import os
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.support import IsolatedCase, ToolCase


class TestSecurityExclusionsAndRedaction(IsolatedCase):
    def test_gitignore_excludes_env_data_contents_and_zips(self):
        gi = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
        lines = [line.strip() for line in gi.splitlines() if line.strip() and not line.strip().startswith("#")]
        self.assertIn(".env", lines)
        self.assertIn("data/*", lines)
        self.assertIn("!data/.gitkeep", lines)
        self.assertIn("*.zip", lines)
        self.assertIn("__pycache__/", lines)
        self.assertFalse((PROJECT_ROOT / "scripts.zip").exists(), "scripts.zip leaked .env/data and must be removed")

    def test_sanitize_and_redact_strip_url_keys_and_tokens(self):
        from nutritrack.ai.exceptions import sanitize_error_text
        from nutritrack.state.run_log import redact

        secrets = [
            "SUPERSECRETTOKEN12345",
            "MY_USDA_KEY_VALUE_999",
            "AnotherSecretValue999",
            "SecretHeaderValue12345",
            "my-secret-jwt-token-xyz",
            "AIzaSyFakeKeyPattern1234567890",
            "gsk_FakeGroqPattern1234567890",
        ]
        samples = [
            f"GET https://generativelanguage.googleapis.com/v1beta/models/gemini:generateContent?key={secrets[0]} failed",
            f"GET https://api.nal.usda.gov/fdc/v1/foods/search?query=apple&api_key={secrets[1]}",
            f"https://example.com/path?foo=bar&key={secrets[2]}&baz=1",
            f"headers={{'x-api-key': '{secrets[3]}'}}",
            f"Authorization: Bearer {secrets[4]}",
            f"token={secrets[5]}",
            f"groq={secrets[6]}",
        ]
        for s, secret in zip(samples, secrets):
            cleaned_ai = sanitize_error_text(s)
            cleaned_log = redact(s)
            self.assertNotIn(secret, cleaned_ai, f"sanitize_error_text leaked secret in: {cleaned_ai}")
            self.assertNotIn(secret, cleaned_log, f"run_log.redact leaked secret in: {cleaned_log}")

    def test_execute_agent_redacts_exception_urls_and_keys(self):
        from nutritrack.agents.base import execute_agent
        from nutritrack.agents.runtime import LLMUnavailableError
        from nutritrack.state.nutrition_state import StateManager
        from nutritrack.state.run_log import RunLog
        from nutritrack.tools import new_context

        class _ExplodingAgent:
            name = "Meal Intake Agent"

            def __init__(self, exc):
                self._exc = exc

            async def run(self, _task, session=None):
                raise self._exc

        state = StateManager()
        ctx = new_context(state, RunLog(), user_id="sec_user", run_id="run_sec_1")

        res1 = execute_agent(
            _ExplodingAgent(LLMUnavailableError("GET https://gemini.googleapis.com/x?key=SECRET_GEM_QUERY_TOKEN99 404")),
            "meal_intake",
            ctx,
            "log 1 banana",
        )
        self.assertEqual(res1["status"], "failed")
        self.assertNotIn("SECRET_GEM_QUERY_TOKEN99", res1["summary"])
        self.assertNotIn("SECRET_GEM_QUERY_TOKEN99", res1["error_message"])

        res2 = execute_agent(
            _ExplodingAgent(RuntimeError("HTTPSConnectionPool: ?api_key=SECRET_USDA_QUERY_TOKEN88 failed")),
            "meal_intake",
            ctx,
            "log 1 apple",
        )
        self.assertEqual(res2["status"], "failed")
        self.assertNotIn("SECRET_USDA_QUERY_TOKEN88", res2["summary"])
        self.assertNotIn("SECRET_USDA_QUERY_TOKEN88", res2["error_message"])


class TestToolAndWebRedaction(ToolCase):
    def test_delegate_to_specialist_redacts_exception_text(self):
        from nutritrack.tools import bind
        from nutritrack.tools.manager_tools import delegate_to_specialist

        def _boom(_task):
            raise RuntimeError("Upstream URL https://api.example/run?key=SECRET_DELEGATE_TOKEN_777 timed out")

        self.ctx.specialists["meal_intake"] = _boom
        payload = bind(delegate_to_specialist, self.ctx)("meal_intake", "log 1 banana")
        self.assertEqual(payload["status"], "failed")
        self.assertNotIn("SECRET_DELEGATE_TOKEN_777", payload["message"])
        log_text = self.ctx.run_log.path.read_text(encoding="utf-8")
        self.assertNotIn("SECRET_DELEGATE_TOKEN_777", log_text)

    def test_legacy_chat_fallback_trace_redacts_exception(self):
        from nutritrack.web import app as web_app

        web_app.app.testing = True
        client = web_app.app.test_client()
        with patch.object(web_app.agent, "chat_capture", side_effect=RuntimeError("https://host?key=SECRET_WEB_TRACE_555")):
            resp = client.post("/api/chat", json={"message": "hi", "mode": "gemini"})
        body = resp.get_json()
        self.assertEqual(resp.status_code, 200)
        self.assertNotIn("SECRET_WEB_TRACE_555", body.get("trace", ""))


class TestExternalNutritionPolicy(IsolatedCase):
    def test_no_cloud_ai_nutrition_generation_helper(self):
        from nutritrack.core import external_nutrition

        self.assertFalse(hasattr(external_nutrition, "_lookup_cloud_ai"))
        self.assertFalse(hasattr(external_nutrition, "_query_cloud_endpoint"))
        src = (PROJECT_ROOT / "nutritrack" / "core" / "external_nutrition.py").read_text(encoding="utf-8")
        self.assertNotIn("cloud_ai_nutrition", src)

    def test_external_lookup_is_opt_in_by_default_and_never_uses_url_key(self):
        from nutritrack.core import external_nutrition
        from nutritrack.core.nutrition import resolve_item

        self.assertFalse(external_nutrition.external_enabled())
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NUTRITRACK_ONLINE_LOOKUP", None)
            with patch.dict(sys.modules):
                sys.modules.pop("pytest", None)
                self.assertFalse(external_nutrition.external_enabled())
        res_default = resolve_item("dragonfruit", 100, "g")
        self.assertFalse(res_default["ok"])
        self.assertEqual(res_default["reason"], "unknown_food")
        self.assertIn("question", res_default)

        # Opt-in requires explicit non-offline mode AND NUTRITRACK_ONLINE_LOOKUP=1
        from nutritrack.ai.config import ai_mode_scope

        with patch.dict(os.environ, {"NUTRITRACK_ONLINE_LOOKUP": "1"}, clear=False):
            os.environ.pop("NUTRITRACK_DISABLE_EXTERNAL_FOOD", None)
            with ai_mode_scope("offline"):
                self.assertFalse(external_nutrition.external_enabled())
            with ai_mode_scope("gemini"):
                self.assertTrue(external_nutrition.external_enabled())
        with patch.dict(os.environ, {"NUTRITRACK_ONLINE_LOOKUP": "1", "NUTRITRACK_DISABLE_EXTERNAL_FOOD": "1"}):
            with ai_mode_scope("gemini"):
                self.assertFalse(external_nutrition.external_enabled())

        # USDA lookup must pass api_key in X-Api-Key header, never URL query string
        captured_urls = []

        def fake_get(url, headers=None):
            captured_urls.append((url, headers or {}))
            if "/foods/search" in url:
                return {"foods": [{"description": "Dragonfruit, raw", "fdcId": 42, "foodNutrients": [
                    {"nutrientNumber": "208", "value": 60.0},
                    {"nutrientNumber": "203", "value": 1.2},
                    {"nutrientNumber": "205", "value": 13.0},
                    {"nutrientNumber": "204", "value": 0.4},
                ]}]}
            return {"foodPortions": []}

        with patch.dict(os.environ, {"USDA_API_KEY": "MY_USDA_SECRET_HDR_KEY"}):
            with ai_mode_scope("gemini"), patch.object(external_nutrition, "_http_get_json", side_effect=fake_get):
                found = external_nutrition._lookup_usda("dragonfruit")
        self.assertIsNotNone(found)
        for url, hdrs in captured_urls:
            self.assertNotIn("api_key=", url)
            self.assertNotIn("MY_USDA_SECRET_HDR_KEY", url)
            self.assertEqual(hdrs.get("X-Api-Key"), "MY_USDA_SECRET_HDR_KEY")

    def test_opted_in_external_result_is_labelled_and_never_silently_saved(self):
        from nutritrack.ai.config import ai_mode_scope
        from nutritrack.ca1.food_database import FOOD_DB
        from nutritrack.core import external_nutrition
        from nutritrack.core.nutrition import resolve_item

        fake_item = {
            "name": "Dragonfruit, raw",
            "calories": 60.0,
            "protein": 1.2,
            "carbs": 13.0,
            "fat": 0.4,
            "portions": {"serving": 100.0, "piece": 200.0},
            "source": "USDA FoodData Central",
            "provenance": "usda_fdc",
        }
        FOOD_DB.pop("dragonfruit", None)
        with patch.dict(os.environ, {"NUTRITRACK_ONLINE_LOOKUP": "1"}):
            os.environ.pop("NUTRITRACK_DISABLE_EXTERNAL_FOOD", None)
            with ai_mode_scope("gemini"), patch("nutritrack.core.nutrition.lookup_external_food", return_value=fake_item):
                res = resolve_item("dragonfruit", 100, "g")
        self.assertTrue(res["ok"])
        self.assertTrue(res.get("external"))
        self.assertEqual(res.get("provenance"), "usda_fdc")
        self.assertEqual(res.get("external_source"), "USDA FoodData Central")
        self.assertTrue(any("external" in n.lower() and "not saved" in n.lower() for n in res["notes"]))
        self.assertNotIn("dragonfruit", FOOD_DB)
        self.assertFalse(Path(os.environ["NUTRITRACK_CUSTOM_FOODS"]).exists())


class TestProviderFailoverAndLogSeparation(IsolatedCase):
    def test_corrupt_persisted_mode_file_is_quarantined_and_returns_none(self):
        from nutritrack.ai.config import load_persisted_mode

        tmp_dir = Path(self.tmp)
        mode_file = tmp_dir / "ai_mode_settings.json"
        mode_file.write_text("{broken json", encoding="utf-8")
        self.assertIsNone(load_persisted_mode(tmp_dir))
        self.assertFalse(mode_file.exists())
        corrupts = list(tmp_dir.glob("ai_mode_settings.json.corrupt-*"))
        self.assertEqual(len(corrupts), 1)

        mode_file.write_text(json.dumps({"selected_mode": "bogus_invalid"}), encoding="utf-8")
        self.assertIsNone(load_persisted_mode(tmp_dir))
        self.assertFalse(mode_file.exists())
        self.assertEqual(len(list(tmp_dir.glob("ai_mode_settings.json.corrupt-*"))), 2)

    def test_half_open_probe_cancelled_releases_probe_in_flight(self):
        import asyncio
        from nutritrack.agents.runtime import run_async
        from nutritrack.ai.config import AIProviderConfig
        from nutritrack.ai.exceptions import ProviderNetworkError
        from nutritrack.ai.failure_classifier import classify_provider_failure
        from nutritrack.ai.provider_manager import AIProviderManager
        from agent_framework import Message

        now = [1000.0]
        cfg = AIProviderConfig(
            selected_mode="gemini",
            gemini_api_key="g-fake-key-test",
            gemini_model="gemini-flash-lite-latest",
            groq_api_key="",
            max_retries=0,
            circuit_failure_threshold=1,
            circuit_cooldown_seconds=5.0,
        )

        class _CancelOnceFactory:
            def build_client(self, provider, _cfg):
                raise asyncio.CancelledError()

        mgr = AIProviderManager(config=cfg, factory=_CancelOnceFactory(), clock=lambda: now[0])
        mgr.circuits["gemini"].record_failure(
            classify_provider_failure(ProviderNetworkError("down", provider="gemini"), provider="gemini"),
            trip_immediately=True,
        )
        now[0] += 10.0  # Past cooldown -> HALF_OPEN

        with self.assertRaises(asyncio.CancelledError):
            run_async(mgr._invoke_cloud_provider("gemini", messages=[Message("user", ["hi"])], stream=False, options={}))

        self.assertFalse(mgr.circuits["gemini"].snapshot()["probe_in_flight"])
        self.assertFalse(mgr.recovery_manager.snapshot()["probe_in_progress"])
        self.assertFalse(mgr.health.recovery_attempting)

    def test_provider_events_go_to_separate_log_and_legacy_system_rows_filtered(self):
        from nutritrack.ai.config import AIProviderConfig
        from nutritrack.ai.provider_manager import AIProviderManager
        from nutritrack.state.run_log import RunLog
        from nutritrack.web import app as web_app

        logs_path = Path(self.tmp) / "logs"
        logs_path.mkdir(parents=True, exist_ok=True)
        agent_log = RunLog(path=logs_path / "agent_runs.jsonl")
        cfg = AIProviderConfig(selected_mode="offline")
        mgr = AIProviderManager(config=cfg, run_log=agent_log, persisted_dir=self.tmp)
        mgr.set_mode("offline", persist=False)

        agent_entries = agent_log.read()
        self.assertFalse(any(e.get("user_id") == "system" or e.get("agent") == "AI Provider Manager" for e in agent_entries))

        provider_log_path = logs_path / "ai_provider_events.jsonl"
        self.assertTrue(provider_log_path.exists())
        provider_entries = RunLog(path=provider_log_path).read()
        self.assertGreaterEqual(len(provider_entries), 1)
        required_keys = {"timestamp", "run_id", "user_id", "agent", "task", "tool", "status"}
        self.assertTrue(required_keys.issubset(provider_entries[-1].keys()))

        # Even if legacy agent_runs.jsonl contains historical system rows, API excludes them by default
        self.addCleanup(web_app.reset_ca2_runtime)
        with patch.dict(os.environ, {"NUTRITRACK_DATA_DIR": str(Path(self.tmp) / "data"), "NUTRITRACK_LOG_DIR": str(logs_path)}):
            web_app.reset_ca2_runtime()
            agent_log.append(
                run_id="ai-provider-router", user_id="system", agent="AI Provider Manager",
                task="legacy", tool="provider_selected", status="success",
            )
            agent_log.append(
                run_id="r-user-1", user_id="demo_user", agent="Meal Intake Agent",
                task="log", tool="add_meal_to_state", status="success",
            )
            web_app.app.testing = True
            client = web_app.app.test_client()
            runs_default = client.get("/api/nutritrack/runs?limit=10").get_json()["entries"]
            self.assertTrue(all(e.get("user_id") != "system" for e in runs_default))
            self.assertEqual(len(runs_default), 1)
            runs_all = client.get("/api/nutritrack/runs?limit=10&include_system=1").get_json()["entries"]
            self.assertEqual(len(runs_all), 2)
            state_runs = client.get("/api/nutritrack/state?user_id=demo_user").get_json()["recent_runs"]
            self.assertTrue(all(e.get("user_id") != "system" for e in state_runs))

    def test_data_only_honestly_labelled_in_api_and_log(self):
        from nutritrack.agents.nutrition_manager_agent import NutriTrackOrchestrator
        from nutritrack.agents.runtime import DeterministicNutriTrackChatClient
        from nutritrack.state.nutrition_state import StateManager
        from nutritrack.state.run_log import RunLog

        with patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaSyFakeConfiguredKeyForTestOnly123"}):
            sm = StateManager(base_dir=Path(self.tmp) / "data2", ca1_log_path=None)
            rl = RunLog(path=Path(self.tmp) / "logs2" / "agent_runs.jsonl")
            orch = NutriTrackOrchestrator(
                state_manager=sm, run_log=rl, client=DeterministicNutriTrackChatClient(), mode="offline",
            )
            res = orch.run_turn("I ate 2 boiled eggs")
        self.assertEqual(res["status"], "success")
        self.assertFalse(res.get("is_llm", True))
        self.assertTrue(res.get("data_only_active"))
        self.assertEqual(res.get("execution_mode"), "data_only_deterministic")
        self.assertFalse(res["mode_info"]["is_llm"])
        self.assertEqual(res["ai_status"]["active_provider"], "data_only")
        self.assertEqual(res["ai_status"]["effective_label"], "Offline Mode Active")
        entries = rl.read(run_id=res["run_id"])
        self.assertTrue(entries)
        self.assertTrue(all(e.get("execution_mode") == "data_only" for e in entries))


class TestAgentFrameworkUsage(IsolatedCase):
    def test_create_chat_client_safe_across_closed_loops_and_bounded(self):
        from nutritrack.agents import runtime
        from agent_framework import Message
        from openai.types.chat import ChatCompletion, ChatCompletionMessage
        from openai.types.chat.chat_completion import Choice

        fake_resp = ChatCompletion(
            id="chatcmpl-test-1",
            created=0,
            model="gemini-flash-lite-latest",
            object="chat.completion",
            choices=[
                Choice(
                    index=0,
                    finish_reason="stop",
                    message=ChatCompletionMessage(role="assistant", content="ok"),
                )
            ],
        )

        # Simulate live client backed by AsyncOpenAI that binds to active loop
        clients_created = []

        class _FakeAsyncOpenAI:
            def __init__(self, **_kwargs):
                self._loop = runtime.asyncio.get_running_loop()
                self.closed = False
                clients_created.append(self)

                async def _create(*_a, **_kw):
                    if self._loop.is_closed() or runtime.asyncio.get_running_loop() is not self._loop:
                        raise RuntimeError("Event loop is closed")
                    return fake_resp

                self.chat = type("Chat", (), {"completions": type("Comp", (), {"create": _create})()})()

            async def close(self):
                self.closed = True

        with patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaSyFakeKeyForLoopTest123456"}):
            with patch("nutritrack.agents.runtime.AsyncOpenAI", _FakeAsyncOpenAI, create=True):
                client, info = runtime.create_chat_client(mode="gemini")
        self.assertTrue(info["is_llm"])
        cfg = client.function_invocation_configuration
        self.assertFalse(cfg["allow_concurrent_invocation"])
        self.assertIsNotNone(cfg.get("max_iterations"))
        self.assertLessEqual(cfg["max_iterations"], 25)
        self.assertIsNotNone(cfg.get("max_function_calls"))
        self.assertIsNotNone(cfg.get("max_duration_seconds"))
        self.assertFalse(cfg.get("include_detailed_errors", True))

        # Running two separate asyncio.run() turns on the SAME client must NOT raise Event loop is closed
        with patch("nutritrack.agents.runtime.AsyncOpenAI", _FakeAsyncOpenAI, create=True):
            r1 = runtime.run_async(client.get_response([Message("user", ["hello 1"])]))
            r2 = runtime.run_async(client.get_response([Message("user", ["hello 2"])]))
        self.assertEqual(r1.text, "ok")
        self.assertEqual(r2.text, "ok")
        self.assertEqual(len(clients_created), 2)
        self.assertTrue(all(c.closed for c in clients_created))

    def test_all_maf_clients_bound_function_invocation_and_contextvars_propagate(self):
        import contextvars
        from nutritrack.agents.runtime import (
            DeterministicNutriTrackChatClient, FailingChatClient, ScriptedChatClient, run_async,
        )
        from nutritrack.ai.config import AIProviderConfig
        from nutritrack.ai.provider_manager import AIProviderManager

        mgr = AIProviderManager(config=AIProviderConfig(selected_mode="offline"), persisted_dir=self.tmp)
        for c in (DeterministicNutriTrackChatClient(), FailingChatClient(), ScriptedChatClient(), mgr.chat_client):
            cfg = c.function_invocation_configuration
            self.assertFalse(cfg["allow_concurrent_invocation"])
            self.assertIsNotNone(cfg.get("max_iterations"))
            self.assertIsNotNone(cfg.get("max_function_calls"))
            self.assertIsNotNone(cfg.get("max_duration_seconds"))

        var = contextvars.ContextVar("test_trace_id", default="unset")
        token = var.set("trace-xyz")
        try:
            async def _inner():
                async def _nested():
                    return var.get()
                return run_async(_nested())

            self.assertEqual(run_async(_inner()), "trace-xyz")
        finally:
            var.reset(token)


class TestCA1ThreadSafety(IsolatedCase):
    def test_chat_capture_does_not_swap_global_sys_stdout(self):
        import threading
        import time
        from nutritrack.ca1.agent import MealCaloricAgent

        agent = MealCaloricAgent(verbose=True)
        orig_stdout = sys.stdout
        seen_ids = set()
        errors = []
        stop = threading.Event()

        def observer():
            while not stop.is_set():
                seen_ids.add(id(sys.stdout))
                time.sleep(0.001)

        t = threading.Thread(target=observer, daemon=True)
        t.start()
        try:
            for _ in range(15):
                res = agent.chat_capture("help")
                if "NutriTrack" not in res["response"] and "Meal Calorie Helper" not in res["raw"] and "Logged" not in res["raw"] and "RESPONSE" not in res["raw"]:
                    errors.append(f"Unexpected empty capture: {res}")
                if not res["raw"]:
                    errors.append("Empty raw output")
        finally:
            stop.set()
            t.join(timeout=2)

        self.assertEqual(errors, [])
        self.assertEqual(seen_ids, {id(orig_stdout)}, "chat_capture mutated global sys.stdout across threads")


class TestWebValidationAndRollback(IsolatedCase):
    def setUp(self):
        super().setUp()
        from nutritrack.web import app as web_app

        self._web_app = web_app
        data_dir = Path(self.tmp) / "data"
        logs_dir = Path(self.tmp) / "logs"
        data_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        self.patch_env("NUTRITRACK_DATA_DIR", str(data_dir))
        self.patch_env("NUTRITRACK_LOG_DIR", str(logs_dir))
        web_app.reset_ca2_runtime()
        self.addCleanup(web_app.reset_ca2_runtime)
        self.client = web_app.app.test_client()

    def test_state_and_profile_reject_invalid_dates_with_400(self):
        r1 = self.client.get("/api/nutritrack/state?date=2099-01-01")
        self.assertEqual(r1.status_code, 400)
        self.assertEqual(r1.get_json()["status"], "failed")

        r2 = self.client.get("/api/nutritrack/state?date=not-a-date")
        self.assertEqual(r2.status_code, 400)
        self.assertEqual(r2.get_json()["status"], "failed")

        r3 = self.client.get("/api/nutritrack/profile?date=2099-01-01")
        self.assertEqual(r3.status_code, 400)

    def test_non_demo_user_gets_neutral_default_profile(self):
        resp = self.client.get("/api/nutritrack/profile?user_id=alice")
        self.assertEqual(resp.status_code, 200)
        prof = resp.get_json()["profile"]
        self.assertNotIn("Arshdeep", prof["name"])
        self.assertNotIn("arshdeep", prof["email"])
        self.assertEqual(prof["name"], "NutriTrack User")
        self.assertEqual(prof["email"], "")

    def test_numeric_endpoints_reject_nan_and_inf(self):
        # Raw JSON allows NaN/Infinity tokens or non-finite values in Python json; also test string "NaN"/"inf"
        for payload in ('{"calories": NaN}', '{"calories": Infinity}', '{"calories": "nan"}', '{"protein": "inf"}'):
            r = self.client.post("/api/nutritrack/goals", data=payload, content_type="application/json")
            self.assertEqual(r.status_code, 400, f"Expected 400 for {payload}, got {r.status_code}")

        for payload in ('{"liters": NaN}', '{"delta_liters": Infinity}'):
            r = self.client.post("/api/nutritrack/water", data=payload, content_type="application/json")
            self.assertEqual(r.status_code, 400, f"Expected 400 for {payload}")

        r_prof = self.client.post("/api/nutritrack/profile", data='{"weight_kg": Infinity}', content_type="application/json")
        self.assertEqual(r_prof.status_code, 400)

        r_cf = self.client.post(
            "/api/nutritrack/custom-food",
            data='{"name": "mystery bar", "calories": NaN, "protein": 10}',
            content_type="application/json",
        )
        self.assertEqual(r_cf.status_code, 400)

    def test_insights_passes_period_end_date_to_profile(self):
        from datetime import date, timedelta

        yday = (date.today() - timedelta(days=1)).isoformat()
        w = self.client.post("/api/nutritrack/water", json={"date": yday, "liters": 1.75})
        self.assertEqual(w.status_code, 200)

        resp = self.client.get(f"/api/nutritrack/insights?end_date={yday}")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(body["profile"]["active_date"], yday)
        self.assertEqual(body["profile"]["water_today_liters"], 1.75)

    def test_recommendations_does_not_use_stale_pool_when_turn_fails(self):
        from unittest.mock import patch
        from nutritrack.agents import NutriTrackOrchestrator
        from nutritrack.agents.runtime import ScriptedChatClient, text_response, tool_call_response
        from nutritrack.state import RunLog, StateManager
        from nutritrack.tools import recommendation_tools

        sm = StateManager(base_dir=Path(self.tmp) / "data", ca1_log_path=None)
        rl = RunLog(path=Path(self.tmp) / "logs" / "agent_runs.jsonl")
        st = sm.get("demo_user")
        called = []

        def partial_then_fail(task_msg):
            called.append(True)
            st.pool["meta"] = {"remaining_calories": 1800.0, "remaining_protein": 120.0}
            st.pool["candidates"] = {
                "c01": {
                    "id": "c01",
                    "name": "2 pieces egg",
                    "items": [{"food": "egg", "quantity": 2, "unit": "piece", "quantity_str": "2 pieces"}],
                    "estimate": {"calories": 156.0, "protein": 12.0, "carbs": 1.2, "fat": 10.6},
                }
            }
            st.pool["ranked"] = ["c01"]
            return {"status": "failed", "code": "llm_unavailable", "message": "Timed out mid-turn"}

        client = ScriptedChatClient(
            scripts={
                "nutrition_manager": [
                    tool_call_response([("delegate_to_specialist", {"specialist": "meal_recommendation", "task": "recommend a meal"})]),
                    text_response("Failed to generate recommendations due to provider timeout."),
                ]
            }
        )
        orch = NutriTrackOrchestrator(
            state_manager=sm,
            run_log=rl,
            client=client,
            mode="llm",
            require_llm=True,
            specialist_overrides={"meal_recommendation": partial_then_fail},
        )
        self._web_app.set_ca2_orchestrator(orch)

        with patch.object(
            recommendation_tools,
            "validate_recommendation_fit",
            wraps=recommendation_tools.validate_recommendation_fit,
        ) as mock_validate:
            resp = self.client.post("/api/nutritrack/recommendations", json={"filter": "all"})
            mock_validate.assert_not_called()
        body = resp.get_json()
        self.assertEqual(called, [True])
        self.assertEqual(body["activated_specialists"], ["meal_recommendation"])
        self.assertEqual(body["status"], "failed")
        self.assertEqual(st.snapshot()["recommendations"], [])

    def test_edit_meal_restores_original_on_orchestrator_failure(self):
        from nutritrack.agents import FailingChatClient, NutriTrackOrchestrator
        from nutritrack.state import RunLog, StateManager
        from nutritrack.tools.base import LLMUnavailableError

        # First log 2 eggs using deterministic client
        add_resp = self.client.post(
            "/api/nutritrack/meals",
            json={"food": "boiled egg", "portion": "2 pieces", "meal_type": "breakfast"},
        )
        self.assertEqual(add_resp.status_code, 200)
        meals = add_resp.get_json()["state"]["meals"]
        self.assertEqual(len(meals), 1)
        meal_id = meals[0]["meal_id"]

        # Swap orchestrator to FailingChatClient so the replacement turn fails
        sm, rl, _ = self._web_app._get_ca2_runtime()
        failing = FailingChatClient(error=LLMUnavailableError("LLM timeout during edit"))
        orch = NutriTrackOrchestrator(state_manager=sm, run_log=rl, client=failing, mode="llm", require_llm=True)
        self._web_app.set_ca2_orchestrator(orch)

        edit_resp = self.client.put(
            f"/api/nutritrack/meals/{meal_id}",
            json={"food": "banana", "portion": "1 piece", "meal_type": "breakfast"},
        )
        self.assertNotEqual(edit_resp.status_code, 200)

        # Original meal must still be preserved cleanly
        st_after = self.client.get("/api/nutritrack/state").get_json()["state"]
        self.assertEqual(len(st_after["meals"]), 1)
        self.assertEqual(st_after["meals"][0]["food"], "egg")
        self.assertEqual(st_after["consumed"]["calories"], 156.0)

    def test_delete_missing_meal_returns_404(self):
        resp = self.client.delete("/api/nutritrack/meals/m-does-not-exist")
        self.assertEqual(resp.status_code, 404)
        body = resp.get_json()
        self.assertEqual(body["status"], "failed")
        self.assertEqual(body["error_code"], "not_found")


class TestOfflineWebUIAndBindings(unittest.TestCase):
    def test_index_html_has_no_external_cdns_or_innerhtml(self):
        html = (PROJECT_ROOT / "nutritrack" / "web" / "static" / "index.html").read_text(encoding="utf-8")
        self.assertNotIn("cdn.tailwindcss.com", html)
        self.assertNotIn("fonts.googleapis.com", html)
        self.assertNotIn("fonts.gstatic.com", html)
        self.assertNotIn("chart.js", html.lower())
        self.assertNotIn("innerHTML", html)
        self.assertNotIn("outerHTML", html)
        self.assertNotIn("document.write", html)

    def test_index_html_fixes_date_insights_and_data_only_ui_bugs(self):
        html = (PROJECT_ROOT / "nutritrack" / "web" / "static" / "index.html").read_text(encoding="utf-8")
        self.assertNotIn("toISOString().slice(0, 10)", html, "Must format local calendar date, not UTC ISO")
        self.assertNotIn("trends.start_date", html, "Weekly trends use trends.period.start")
        self.assertNotIn("trends.end_date", html, "Weekly trends use trends.period.end")
        self.assertNotIn("ms.percentages", html, "calculate_macro_split returns percent, not percentages")
        self.assertNotIn("st.state", html, "Typo st.state in renderDashboardWeeklyChart")
        self.assertNotIn("cloud_ai_nutrition", html)
        self.assertIn("ms.percent", html)
        self.assertIn("trends.period", html)
        self.assertIn("dash-feedback-mode", html)
        self.assertIn("data_only_active", html)
        # All Material Symbol spans must be converted to inline SVGs offline
        self.assertIn("hydrateMaterialIcons", html)
        self.assertIn(".bg-primary", html)
        # Strict 3-mode AI selector (zero Automatic/Online choices in UI)
        self.assertNotIn('value="automatic"', html)
        self.assertNotIn('value="online"', html)
        self.assertNotIn('data-ai-mode="automatic"', html)
        self.assertNotIn('data-ai-mode="online"', html)
        self.assertIn('data-ai-mode="gemini"', html)
        self.assertIn('data-ai-mode="groq"', html)
        self.assertIn('data-ai-mode="offline"', html)

    def test_ca1_error_paths_redact_secrets(self):
        import tempfile
        from unittest.mock import patch
        from nutritrack.ca1 import daily_store, llm_agent, tools as ca1_tools

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(daily_store, "_STORE_FILE", str(Path(tmp) / "daily_log.json")):
                with patch.object(ca1_tools.config, "API_PROVIDER", "openfoodfacts"), patch.object(
                    ca1_tools.config, "has_api_key", return_value=True
                ), patch.object(
                    ca1_tools, "_fetch_from_openfoodfacts", side_effect=RuntimeError("GET https://api.example.com?key=AIzaSyDummySecret12345678901234567890 failed")
                ):
                    from nutritrack.ai.config import ai_mode_scope
                    with ai_mode_scope("gemini"):
                        res = ca1_tools._fetch_from_api("dragonfruit")
                    self.assertEqual(res["status"], "api_error")
                    self.assertNotIn("AIzaSyDummySecret12345678901234567890", res["error"])
                    self.assertIn("REDACTED", res["error"])

                class _FakeCompletions:
                    def create(self, **kwargs):
                        raise RuntimeError("Upstream 401 token=ghp_DummySecretToken1234567890123456")

                class _FakeChat:
                    completions = _FakeCompletions()

                class _FakeClient:
                    chat = _FakeChat()

                with patch("nutritrack.ca1.llm_agent.llm_config.has_token", return_value=True), patch(
                    "nutritrack.ca1.llm_agent.OpenAI", return_value=_FakeClient()
                ):
                    agent = llm_agent.LLMMealAgent(verbose=False)
                    out = agent.chat_capture("hello")
                    self.assertNotIn("ghp_DummySecretToken1234567890123456", out["response"])
                    self.assertNotIn("ghp_DummySecretToken1234567890123456", out["trace"])
                    self.assertIn("REDACTED", out["response"])
