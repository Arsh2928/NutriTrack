"""Automated tests for strict single-provider failure isolation (no cross-provider fallback)."""

from __future__ import annotations

import tempfile
import unittest

from nutritrack.agents import DeterministicNutriTrackChatClient, NutriTrackOrchestrator
from nutritrack.ai import STATE_CLOSED, STATE_OPEN, AIProviderConfig, AIProviderManager
from nutritrack.state import RunLog, StateManager


class ControllableClock:
    def __init__(self, start: float = 2000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class ScriptedIsolationCloudClient(DeterministicNutriTrackChatClient):
    """MAF chat client whose failures can be injected per call to verify strict provider isolation."""

    def __init__(self, provider_name: str, model_name: str, *, error_fn=None) -> None:
        super().__init__()
        self.provider_name = provider_name
        self.model = model_name
        self.error_fn = error_fn
        self.call_count = 0

    async def _inner_get_response(self, *, messages, stream, options, **kwargs):
        self.call_count += 1
        if self.error_fn is not None:
            err = self.error_fn(self.call_count, messages, options)
            if err is not None:
                raise err
        return await super()._inner_get_response(
            messages=messages,
            stream=stream,
            options=options,
            **kwargs,
        )


class ProviderIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.clock = ControllableClock()
        self.slept: list[float] = []

        async def _fake_sleep(delay: float) -> None:
            self.slept.append(delay)
            self.clock.advance(delay)

        self.fake_sleep = _fake_sleep
        self.sm = StateManager(base_dir=self._tmp.name, ca1_log_path=None)
        self.rl = RunLog(path=f"{self._tmp.name}/runs.jsonl")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _make_config(self, mode: str, max_retries: int = 2) -> AIProviderConfig:
        return AIProviderConfig(
            selected_mode=mode,
            gemini_api_key="AIzaSyTestGeminiKey000000000000000",
            gemini_model="gemini-flash-lite-latest",
            groq_api_key="gsk_TestGroqKey0000000000000000000",
            groq_model="openai/gpt-oss-120b",
            max_retries=max_retries,
            retry_base_delay_seconds=0.5,
            retry_max_delay_seconds=4.0,
            circuit_failure_threshold=3,
            circuit_cooldown_seconds=60.0,
        )

    def test_gemini_rate_limit_retries_bounded_never_calls_groq_and_preserves_mode(self) -> None:
        """Gemini 429 rate-limit retries within bounds, never touches Groq, never switches mode."""
        cfg = self._make_config(mode="gemini", max_retries=2)
        gem_client = ScriptedIsolationCloudClient(
            "gemini",
            cfg.gemini_model,
            error_fn=lambda idx, m, o: Exception("HTTP 429 Too Many Requests: RPM rate limit reached, retry in 1s"),
        )
        groq_client = ScriptedIsolationCloudClient("groq", cfg.groq_model)

        mgr = AIProviderManager(
            config=cfg,
            run_log=self.rl,
            clock=self.clock,
            async_sleep=self.fake_sleep,
            persisted_dir=self._tmp.name,
            client_overrides={"gemini": gem_client, "groq": groq_client},
        )
        orch = NutriTrackOrchestrator(state_manager=self.sm, run_log=self.rl, ai_manager=mgr)

        res = orch.run_turn("I ate 2 boiled eggs.")
        self.assertEqual(res["status"], "failed")
        self.assertEqual(res["error_code"], "llm_unavailable")
        self.assertEqual(gem_client.call_count, 3)  # 1 attempt + 2 retries
        self.assertEqual(groq_client.call_count, 0)
        self.assertEqual(mgr.circuits["groq"].state, STATE_CLOSED)

        ai_status = res["ai_status"]
        self.assertEqual(ai_status["selected_mode"], "gemini")
        self.assertEqual(ai_status["active_provider"], "gemini")
        self.assertFalse(ai_status["fallback_active"])
        self.assertFalse(ai_status["data_only_active"])
        self.assertEqual(ai_status["provider_status"]["gemini"], "rate_limited")
        self.assertIn("Google Gemini", res["error_message"])
        self.assertEqual(res["state"]["meals"], [])

    def test_gemini_daily_quota_fails_immediately_with_zero_retries_and_no_groq(self) -> None:
        """Gemini daily quota (429 RESOURCE_EXHAUSTED) must NOT retry and must NOT invoke Groq."""
        cfg = self._make_config(mode="gemini", max_retries=2)
        gem_client = ScriptedIsolationCloudClient(
            "gemini",
            cfg.gemini_model,
            error_fn=lambda idx, m, o: Exception(
                "HTTP 429 RESOURCE_EXHAUSTED: You exceeded your current quota (GenerateRequestsPerDay)."
            ),
        )
        groq_client = ScriptedIsolationCloudClient("groq", cfg.groq_model)

        mgr = AIProviderManager(
            config=cfg,
            run_log=self.rl,
            clock=self.clock,
            async_sleep=self.fake_sleep,
            persisted_dir=self._tmp.name,
            client_overrides={"gemini": gem_client, "groq": groq_client},
        )
        orch = NutriTrackOrchestrator(state_manager=self.sm, run_log=self.rl, ai_manager=mgr)

        res = orch.run_turn("Set my daily calorie goal to 2200 kcal.")
        self.assertEqual(res["status"], "failed")
        self.assertEqual(gem_client.call_count, 1)
        self.assertEqual(len(self.slept), 0)
        self.assertEqual(groq_client.call_count, 0)
        self.assertEqual(mgr.circuits["gemini"].state, STATE_OPEN)
        self.assertEqual(res["ai_status"]["selected_mode"], "gemini")
        self.assertEqual(res["ai_status"]["provider_status"]["gemini"], "quota_exhausted")

    def test_groq_success_and_groq_failure_never_call_gemini_or_switch_mode(self) -> None:
        """Groq mode uses Groq exclusively on success and never touches Gemini or switches mode on failure."""
        cfg = self._make_config(mode="groq", max_retries=1)
        gem_client = ScriptedIsolationCloudClient("gemini", cfg.gemini_model)
        groq_client = ScriptedIsolationCloudClient("groq", cfg.groq_model)

        mgr = AIProviderManager(
            config=cfg,
            run_log=self.rl,
            clock=self.clock,
            async_sleep=self.fake_sleep,
            persisted_dir=self._tmp.name,
            client_overrides={"gemini": gem_client, "groq": groq_client},
        )
        orch = NutriTrackOrchestrator(state_manager=self.sm, run_log=self.rl, ai_manager=mgr)

        ok_res = orch.run_turn("I ate 150g paneer.")
        self.assertEqual(ok_res["status"], "success")
        self.assertEqual(gem_client.calls if hasattr(gem_client, "calls") else gem_client.call_count, 0)
        self.assertGreater(groq_client.call_count, 0)
        self.assertEqual(ok_res["ai_status"]["selected_mode"], "groq")
        self.assertEqual(ok_res["ai_status"]["active_provider"], "groq")
        self.assertFalse(ok_res["ai_status"]["fallback_active"])

        # Now force Groq HTTP 400 malformed tool call -> zero retries, no Gemini call, mode stays groq
        groq_before = groq_client.call_count
        groq_client.error_fn = lambda idx, m, o: Exception(
            "Error code: 400 - {'error': {'message': 'tool call validation failed', 'code': 'tool_use_failed'}}"
        )
        fail_res = orch.run_turn("Give me a meal recommendation.")
        self.assertEqual(fail_res["status"], "failed")
        self.assertEqual(fail_res["error_code"], "llm_unavailable")
        self.assertEqual(groq_client.call_count, groq_before + 1)
        self.assertEqual(gem_client.call_count, 0)
        self.assertEqual(fail_res["ai_status"]["selected_mode"], "groq")
        self.assertFalse(fail_res["ai_status"]["fallback_active"])
        self.assertIn("malformed_tool_call", fail_res["error_message"])
        # Meal logged before the failed turn remains intact
        self.assertEqual(len(fail_res["state"]["meals"]), 1)

    def test_401_and_404_fail_fast_without_retries_or_provider_switching(self) -> None:
        """HTTP 401 and 404 fail fast (1 call) with circuit trip, preserving selected mode."""
        for status_msg in ("HTTP 401 Unauthorized: API_KEY_INVALID", "HTTP 404 model_not_found: does not exist"):
            with self.subTest(err=status_msg):
                cfg = self._make_config(mode="gemini", max_retries=2)
                gem_client = ScriptedIsolationCloudClient(
                    "gemini",
                    cfg.gemini_model,
                    error_fn=lambda idx, m, o, msg=status_msg: Exception(msg),
                )
                groq_client = ScriptedIsolationCloudClient("groq", cfg.groq_model)
                mgr = AIProviderManager(
                    config=cfg,
                    run_log=self.rl,
                    clock=self.clock,
                    async_sleep=self.fake_sleep,
                    persisted_dir=self._tmp.name,
                    client_overrides={"gemini": gem_client, "groq": groq_client},
                )
                orch = NutriTrackOrchestrator(state_manager=self.sm, run_log=self.rl, ai_manager=mgr)
                res = orch.run_turn("I ate 1 banana.")
                self.assertEqual(res["status"], "failed")
                self.assertEqual(gem_client.call_count, 1)
                self.assertEqual(groq_client.call_count, 0)
                self.assertEqual(mgr.circuits["gemini"].state, STATE_OPEN)
                self.assertEqual(res["ai_status"]["selected_mode"], "gemini")


if __name__ == "__main__":
    unittest.main()
