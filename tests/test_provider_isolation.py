"""15 Mandatory Provider Isolation Tests verifying strict Gemini / Groq / Offline separation."""

from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nutritrack.agents import DeterministicNutriTrackChatClient, NutriTrackOrchestrator
from nutritrack.agents.runtime import run_async
from nutritrack.ai import AIProviderConfig, AIProviderManager, ProviderFactory, normalize_ai_mode
from nutritrack.state import RunLog, StateManager


class StubProviderClient(DeterministicNutriTrackChatClient):
    def __init__(self, name: str, model: str, *, fail_fn=None) -> None:
        super().__init__()
        self.name = name
        self.model = model
        self.fail_fn = fail_fn
        self.calls = 0

    async def _inner_get_response(self, *, messages, stream, options, **kwargs):
        self.calls += 1
        if self.fail_fn is not None:
            exc = self.fail_fn(self.calls)
            if exc is not None:
                raise exc
        return await super()._inner_get_response(messages=messages, stream=stream, options=options, **kwargs)


class MandatoryProviderIsolation15Tests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.sm = StateManager(base_dir=self._tmp.name, ca1_log_path=None)
        self.rl = RunLog(path=f"{self._tmp.name}/runs.jsonl")
        self.clock_t = [1000.0]

        async def _sleep(d: float) -> None:
            self.clock_t[0] += d

        self.fake_sleep = _sleep

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _cfg(self, mode: str, *, max_retries: int = 2) -> AIProviderConfig:
        return AIProviderConfig(
            selected_mode=mode,
            gemini_api_key="AIzaSyIsolationGem1234567890123",
            gemini_model="gemini-explicit-pinned",
            groq_api_key="gsk_IsolationGroq123456789012345",
            groq_model="groq-explicit-pinned",
            max_retries=max_retries,
            retry_base_delay_seconds=0.25,
            retry_max_delay_seconds=2.0,
            circuit_failure_threshold=3,
            circuit_cooldown_seconds=60.0,
        )

    def test_01_gemini_mode_uses_gemini_only(self) -> None:
        cfg = self._cfg("gemini")
        gem = StubProviderClient("gemini", cfg.gemini_model)
        groq = StubProviderClient("groq", cfg.groq_model)
        mgr = AIProviderManager(config=cfg, run_log=self.rl, client_overrides={"gemini": gem, "groq": groq}, persisted_dir=self._tmp.name)
        res = NutriTrackOrchestrator(state_manager=self.sm, run_log=self.rl, ai_manager=mgr).run_turn("I ate 1 banana.")
        self.assertEqual(res["status"], "success")
        self.assertGreater(gem.calls, 0)
        self.assertEqual(groq.calls, 0)

    def test_02_gemini_failure_does_not_call_groq(self) -> None:
        cfg = self._cfg("gemini", max_retries=0)
        gem = StubProviderClient("gemini", cfg.gemini_model, fail_fn=lambda _: TimeoutError("Gemini timed out"))
        groq = StubProviderClient("groq", cfg.groq_model)
        mgr = AIProviderManager(config=cfg, run_log=self.rl, client_overrides={"gemini": gem, "groq": groq}, persisted_dir=self._tmp.name)
        res = NutriTrackOrchestrator(state_manager=self.sm, run_log=self.rl, ai_manager=mgr).run_turn("I ate 1 banana.")
        self.assertEqual(res["status"], "failed")
        self.assertGreaterEqual(gem.calls, 1)
        self.assertEqual(groq.calls, 0)

    def test_03_gemini_failure_preserves_gemini_mode(self) -> None:
        cfg = self._cfg("gemini", max_retries=0)
        gem = StubProviderClient("gemini", cfg.gemini_model, fail_fn=lambda _: ConnectionRefusedError("down"))
        mgr = AIProviderManager(config=cfg, run_log=self.rl, client_overrides={"gemini": gem}, persisted_dir=self._tmp.name)
        res = NutriTrackOrchestrator(state_manager=self.sm, run_log=self.rl, ai_manager=mgr).run_turn("How many calories left?")
        self.assertEqual(res["status"], "failed")
        self.assertEqual(mgr.config.selected_mode, "gemini")
        self.assertEqual(res["ai_status"]["selected_mode"], "gemini")
        self.assertFalse(res["ai_status"]["data_only_active"])

    def test_04_groq_mode_uses_groq_only(self) -> None:
        cfg = self._cfg("groq")
        gem = StubProviderClient("gemini", cfg.gemini_model)
        groq = StubProviderClient("groq", cfg.groq_model)
        mgr = AIProviderManager(config=cfg, run_log=self.rl, client_overrides={"gemini": gem, "groq": groq}, persisted_dir=self._tmp.name)
        res = NutriTrackOrchestrator(state_manager=self.sm, run_log=self.rl, ai_manager=mgr).run_turn("I ate 1 banana.")
        self.assertEqual(res["status"], "success")
        self.assertGreater(groq.calls, 0)
        self.assertEqual(gem.calls, 0)

    def test_05_groq_failure_does_not_call_gemini(self) -> None:
        cfg = self._cfg("groq", max_retries=0)
        gem = StubProviderClient("gemini", cfg.gemini_model)
        groq = StubProviderClient("groq", cfg.groq_model, fail_fn=lambda _: Exception("HTTP 503 Groq unavailable"))
        mgr = AIProviderManager(config=cfg, run_log=self.rl, client_overrides={"gemini": gem, "groq": groq}, persisted_dir=self._tmp.name)
        res = NutriTrackOrchestrator(state_manager=self.sm, run_log=self.rl, ai_manager=mgr).run_turn("I ate 1 banana.")
        self.assertEqual(res["status"], "failed")
        self.assertEqual(gem.calls, 0)

    def test_06_groq_failure_preserves_groq_mode(self) -> None:
        cfg = self._cfg("groq", max_retries=0)
        groq = StubProviderClient("groq", cfg.groq_model, fail_fn=lambda _: Exception("HTTP 503 Groq down"))
        mgr = AIProviderManager(config=cfg, run_log=self.rl, client_overrides={"groq": groq}, persisted_dir=self._tmp.name)
        res = NutriTrackOrchestrator(state_manager=self.sm, run_log=self.rl, ai_manager=mgr).run_turn("How many calories?")
        self.assertEqual(res["status"], "failed")
        self.assertEqual(mgr.config.selected_mode, "groq")
        self.assertEqual(res["ai_status"]["selected_mode"], "groq")

    def test_07_offline_mode_zero_cloud_calls(self) -> None:
        cfg = self._cfg("offline")
        gem = StubProviderClient("gemini", cfg.gemini_model)
        groq = StubProviderClient("groq", cfg.groq_model)
        mgr = AIProviderManager(config=cfg, run_log=self.rl, client_overrides={"gemini": gem, "groq": groq}, persisted_dir=self._tmp.name)
        res = NutriTrackOrchestrator(state_manager=self.sm, run_log=self.rl, ai_manager=mgr).run_turn("I ate 2 eggs.")
        self.assertEqual(res["status"], "success")
        self.assertEqual(gem.calls, 0)
        self.assertEqual(groq.calls, 0)

    def test_08_automatic_and_online_modes_rejected_everywhere(self) -> None:
        for bad in ("automatic", "auto", "online", "cloud"):
            with self.assertRaises(ValueError):
                normalize_ai_mode(bad)

    def test_09_gemini_model_unchanged_without_auto_switching(self) -> None:
        from nutritrack.ai.config import ai_mode_scope
        from nutritrack.ai.provider_factory import _wrap_sdk_chat_completions

        seen_models: list[str] = []

        class FakeCompletions:
            async def create(self, *args, **kwargs):
                seen_models.append(kwargs.get("model"))
                raise Exception("HTTP 404 model_not_found")

        fake_sdk = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
        _wrap_sdk_chat_completions(fake_sdk, "gemini", "gemini-explicit-pinned")
        with ai_mode_scope("gemini"), self.assertRaises(Exception):
            run_async(fake_sdk.chat.completions.create(model="gemini-explicit-pinned", messages=[]))
        self.assertEqual(seen_models, ["gemini-explicit-pinned"])

    def test_10_groq_model_unchanged_without_auto_switching(self) -> None:
        from nutritrack.ai.config import ai_mode_scope
        from nutritrack.ai.provider_factory import _wrap_sdk_chat_completions

        seen_models: list[str] = []

        class FakeCompletions:
            async def create(self, *args, **kwargs):
                seen_models.append(kwargs.get("model"))
                raise Exception("HTTP 429 rate limit reached for model")

        fake_sdk = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
        _wrap_sdk_chat_completions(fake_sdk, "groq", "groq-explicit-pinned")
        with ai_mode_scope("groq"), self.assertRaises(Exception):
            run_async(fake_sdk.chat.completions.create(model="groq-explicit-pinned", messages=[]))
        self.assertEqual(seen_models, ["groq-explicit-pinned"])

    def test_11_model_discovery_and_rotation_helpers_absent(self) -> None:
        import nutritrack.ai.provider_factory as pf

        self.assertFalse(hasattr(pf, "_FALLBACK_MODELS"))
        self.assertFalse(hasattr(pf, "_RESOLVED_MODELS"))
        self.assertFalse(hasattr(pf, "_is_model_rotatable_error"))
        self.assertIsNotNone(ProviderFactory)

    def test_12_rate_limit_retries_bounded_and_daily_quota_zero_retries(self) -> None:
        cfg_rl = self._cfg("groq", max_retries=2)
        groq_rl = StubProviderClient("groq", cfg_rl.groq_model, fail_fn=lambda _: Exception("HTTP 429 rate_limit retry in 1s"))
        mgr_rl = AIProviderManager(config=cfg_rl, run_log=self.rl, clock=lambda: self.clock_t[0], async_sleep=self.fake_sleep, client_overrides={"groq": groq_rl}, persisted_dir=self._tmp.name)
        NutriTrackOrchestrator(state_manager=self.sm, run_log=self.rl, ai_manager=mgr_rl).run_turn("I ate 1 banana.")
        self.assertEqual(groq_rl.calls, 3)  # 1 initial + 2 retries max

        cfg_q = self._cfg("gemini", max_retries=2)
        gem_q = StubProviderClient("gemini", cfg_q.gemini_model, fail_fn=lambda _: Exception("HTTP 429 RESOURCE_EXHAUSTED GenerateRequestsPerDay"))
        mgr_q = AIProviderManager(config=cfg_q, run_log=self.rl, clock=lambda: self.clock_t[0], async_sleep=self.fake_sleep, client_overrides={"gemini": gem_q}, persisted_dir=self._tmp.name)
        NutriTrackOrchestrator(state_manager=self.sm, run_log=self.rl, ai_manager=mgr_q).run_turn("I ate 1 banana.")
        self.assertEqual(gem_q.calls, 1)  # Zero retries on quota exhaustion

    def test_13_malformed_tool_call_400_and_401_and_404_fail_fast(self) -> None:
        for err_text in (
            "HTTP 400 tool call validation failed: tool_use_failed",
            "HTTP 401 Unauthorized API_KEY_INVALID",
            "HTTP 404 model_not_found",
        ):
            with self.subTest(err_text=err_text):
                cfg = self._cfg("groq", max_retries=2)
                groq = StubProviderClient("groq", cfg.groq_model, fail_fn=lambda _: Exception(err_text))
                mgr = AIProviderManager(config=cfg, run_log=self.rl, async_sleep=self.fake_sleep, client_overrides={"groq": groq}, persisted_dir=self._tmp.name)
                res = NutriTrackOrchestrator(state_manager=self.sm, run_log=self.rl, ai_manager=mgr).run_turn("I ate 1 apple.")
                self.assertEqual(res["status"], "failed")
                self.assertEqual(groq.calls, 1)
                self.assertEqual(mgr.config.selected_mode, "groq")

    def test_14_flask_chat_returns_503_on_provider_failure_and_manual_retry_works(self) -> None:
        from nutritrack.web import app as web_app

        data_dir = os.path.join(self._tmp.name, "d")
        log_dir = os.path.join(self._tmp.name, "l")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        with patch.dict(
            os.environ,
            {
                "NUTRITRACK_DATA_DIR": data_dir,
                "NUTRITRACK_LOG_DIR": log_dir,
                "NUTRITRACK_AI_MODE": "gemini",
                "GEMINI_API_KEY": "",
                "GOOGLE_API_KEY": "",
                "GROQ_API_KEY": "",
            },
            clear=False,
        ):
            web_app.reset_ca2_runtime()
            client = web_app.app.test_client()
            resp = client.post("/api/nutritrack/chat", json={"message": "What should I eat?"})
            self.assertEqual(resp.status_code, 503)
            body = resp.get_json()
            self.assertEqual(body["status"], "failed")
            self.assertEqual(body["ai_status"]["selected_mode"], "gemini")

    def test_15_manual_switching_persists_cleanly(self) -> None:
        cfg = self._cfg("offline")
        mgr = AIProviderManager(config=cfg, run_log=self.rl, persisted_dir=self._tmp.name)
        for mode in ("gemini", "groq", "offline"):
            st = mgr.set_mode(mode, persist=True)
            self.assertEqual(st["selected_mode"], mode)
            self.assertFalse(st["fallback_active"])
        self.assertEqual(mgr.get_status()["selected_mode"], "offline")

    def test_16_offline_mode_succeeds_even_with_all_network_apis_blocked(self) -> None:
        from nutritrack.tools.fast_log import fast_log_meal

        with patch.dict(os.environ, {"NUTRITRACK_AI_MODE": "offline", "NUTRITRACK_ONLINE_LOOKUP": "1"}):
            with patch("urllib.request.urlopen", side_effect=AssertionError("Network prohibited in offline mode")), \
                 patch("openai.AsyncOpenAI", side_effect=AssertionError("Cloud SDK prohibited in offline mode")):
                cfg = self._cfg("offline")
                mgr = AIProviderManager(config=cfg, run_log=self.rl, persisted_dir=self._tmp.name)
                orch = NutriTrackOrchestrator(state_manager=self.sm, run_log=self.rl, ai_manager=mgr)
                res = orch.run_turn("I ate 2 boiled eggs and 100g oats for breakfast on 2026-11-16")
                self.assertEqual(res["status"], "success")
                self.assertEqual(res["date"], "2026-11-16")
                fast = fast_log_meal(self.sm, self.rl, message="1 banana", date="2026-11-16")
                self.assertEqual(fast["status"], "success")
                self.assertEqual(fast["state"]["consumed"]["calories"], 631.0)


if __name__ == "__main__":
    unittest.main()
