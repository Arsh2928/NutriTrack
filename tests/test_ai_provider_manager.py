"""Automated tests for AIProviderManager, FailureClassifier, RetryPolicy, CircuitBreaker, and Config."""

from __future__ import annotations

import tempfile
import unittest

from nutritrack.agents import DeterministicNutriTrackChatClient, NutriTrackOrchestrator
from nutritrack.ai import (
    STATE_CLOSED,
    STATE_HALF_OPEN,
    STATE_OPEN,
    AIProviderConfig,
    AIProviderManager,
    CircuitBreaker,
    RetryPolicy,
    classify_provider_failure,
    is_model_supported,
)
from nutritrack.state import RunLog, StateManager


class ControllableClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class MockCloudClient(DeterministicNutriTrackChatClient):
    """MAF chat client simulating Gemini or Groq responses and injectable failures."""

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


class AIProviderManagerUnitTests(unittest.TestCase):
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

    def test_gemini_mode_calls_gemini_only(self) -> None:
        cfg = AIProviderConfig(
            selected_mode="gemini",
            gemini_api_key="AIzaSyFakeGeminiKeyForUnitTesting123456",
            gemini_model="gemini-flash-lite-latest",
            groq_api_key="gsk_FakeGroqKeyForUnitTesting123456789",
            groq_model="openai/gpt-oss-120b",
        )
        gem_client = MockCloudClient("gemini", cfg.gemini_model)
        groq_client = MockCloudClient("groq", cfg.groq_model)

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
        self.assertEqual(res["status"], "success")
        self.assertGreater(gem_client.call_count, 0)
        self.assertEqual(groq_client.call_count, 0)

        ai_status = res["ai_status"]
        self.assertEqual(ai_status["selected_mode"], "gemini")
        self.assertEqual(ai_status["active_provider"], "gemini")
        self.assertFalse(ai_status["fallback_active"])
        self.assertEqual(ai_status["effective_label"], "Gemini Connected")
        self.assertEqual(ai_status["provider_status"]["gemini"], "healthy")
        self.assertEqual(res["state"]["consumed"]["calories"], 156.0)

    def test_circuit_breaker_open_half_open_and_close(self) -> None:
        cb = CircuitBreaker("gemini", failure_threshold=3, cooldown_seconds=60.0, clock=self.clock)
        self.assertEqual(cb.state, STATE_CLOSED)

        transient_fail = classify_provider_failure(TimeoutError("Request timed out"), provider="gemini")
        cb.record_failure(transient_fail)
        self.assertEqual(cb.state, STATE_CLOSED)
        cb.record_failure(transient_fail)
        self.assertEqual(cb.state, STATE_CLOSED)
        res3 = cb.record_failure(transient_fail)
        self.assertTrue(res3["opened_now"])
        self.assertEqual(cb.state, STATE_OPEN)

        allowed, reason = cb.allow_request()
        self.assertFalse(allowed)
        self.assertEqual(reason, "circuit_open")

        self.clock.advance(61.0)
        self.assertEqual(cb.state, STATE_HALF_OPEN)
        allowed1, reason1 = cb.allow_request()
        self.assertTrue(allowed1)
        self.assertEqual(reason1, "half_open_probe")

        allowed2, reason2 = cb.allow_request()
        self.assertFalse(allowed2)
        self.assertEqual(reason2, "half_open_probe_in_flight")

        cb.record_failure(transient_fail)
        self.assertEqual(cb.state, STATE_OPEN)

        self.clock.advance(61.0)
        self.assertEqual(cb.state, STATE_HALF_OPEN)
        ok_probe, _ = cb.allow_request()
        self.assertTrue(ok_probe)
        rec = cb.record_success()
        self.assertTrue(rec["recovered"])
        self.assertEqual(cb.state, STATE_CLOSED)

    def test_invalid_api_key_isolated_and_secrets_redacted(self) -> None:
        secret_gemini_key = "AIzaSySecretKeyThatMustNeverAppearInLogs99999"
        secret_groq_key = "gsk_SecretGroqKeyThatMustNeverAppearInLogs88888"

        cfg = AIProviderConfig(
            selected_mode="gemini",
            gemini_api_key=secret_gemini_key,
            groq_api_key=secret_groq_key,
            max_retries=2,
        )

        def _raise_401(call_idx, messages, options):
            raise Exception(f"HTTP 401 Unauthorized: API_KEY_INVALID for key {secret_gemini_key}")

        gem_client = MockCloudClient("gemini", cfg.gemini_model, error_fn=_raise_401)
        groq_client = MockCloudClient("groq", cfg.groq_model)

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
        self.assertEqual(res["ai_status"]["selected_mode"], "gemini")
        self.assertEqual(res["ai_status"]["provider_status"]["gemini"], "config_error")

        status_dump = str(res["ai_status"]) + str(mgr.get_providers_diagnostics()) + str(res.get("error_message"))
        self.assertNotIn(secret_gemini_key, status_dump)
        self.assertNotIn(secret_groq_key, status_dump)
        raw_log = open(f"{self._tmp.name}/runs.jsonl", encoding="utf-8").read()
        self.assertNotIn(secret_gemini_key, raw_log)
        self.assertNotIn(secret_groq_key, raw_log)

    def test_all_failure_classifications_including_malformed_tool_call_and_retry_budget(self) -> None:
        f_net = classify_provider_failure(ConnectionRefusedError("Network is unreachable"), provider="gemini")
        self.assertEqual(f_net.category, "internet_disconnected")
        self.assertFalse(f_net.retryable)
        self.assertFalse(f_net.fallback_eligible)

        f_dns = classify_provider_failure(OSError("gaierror: [Errno 11001] getaddrinfo failed"), provider="gemini")
        self.assertEqual(f_dns.category, "dns_failure")
        self.assertFalse(f_dns.retryable)

        f_ct = classify_provider_failure(TimeoutError("ConnectTimeout: connection timed out"), provider="gemini")
        self.assertEqual(f_ct.category, "connection_timeout")
        self.assertTrue(f_ct.retryable)

        f_rt = classify_provider_failure(TimeoutError("ReadTimeout: response timed out"), provider="gemini")
        self.assertEqual(f_rt.category, "response_timeout")
        self.assertTrue(f_rt.retryable)

        f_rl = classify_provider_failure(Exception("HTTP 429 Too Many Requests: retry in 2s"), provider="gemini")
        self.assertEqual(f_rl.category, "rate_limit_temporary")
        self.assertTrue(f_rl.retryable)
        self.assertEqual(f_rl.retry_after_seconds, 2.0)

        f_qe = classify_provider_failure(
            Exception("HTTP 429 RESOURCE_EXHAUSTED: GenerateRequestsPerDay daily quota exceeded"),
            provider="gemini",
        )
        self.assertEqual(f_qe.category, "quota_exhausted_daily")
        self.assertFalse(f_qe.retryable)
        self.assertTrue(f_qe.is_quota_exhausted)

        f_groq_rl = classify_provider_failure(
            Exception("HTTP 429 rate_limit_exceeded: Please try again in 45s"),
            provider="groq",
        )
        policy = RetryPolicy(max_retries=2, base_delay_seconds=1.0, max_delay_seconds=8.0)
        decision = policy.evaluate(0, f_groq_rl)
        self.assertFalse(decision.should_retry)
        self.assertIn("exceeding max_delay_seconds", decision.reason)

        f_503 = classify_provider_failure(Exception("HTTP 503 Service Unavailable"), provider="gemini")
        self.assertEqual(f_503.category, "server_error")
        self.assertTrue(f_503.retryable)

        f_401 = classify_provider_failure(Exception("HTTP 401 Invalid API key"), provider="groq")
        self.assertEqual(f_401.category, "invalid_api_key")
        self.assertFalse(f_401.retryable)

        ok_m, _ = is_model_supported("gemini", "gemini-pro")
        self.assertFalse(ok_m)
        f_mod = classify_provider_failure(Exception("HTTP 404 model_not_found: decommissioned"), provider="groq")
        self.assertEqual(f_mod.category, "unsupported_model")
        self.assertFalse(f_mod.retryable)

        f_tool = classify_provider_failure(
            Exception("Error code: 400 - tool call validation failed (tool_use_failed / failed_generation)"),
            provider="groq",
        )
        self.assertEqual(f_tool.category, "malformed_tool_call")
        self.assertFalse(f_tool.retryable)
        self.assertFalse(f_tool.fallback_eligible)

        f_400 = classify_provider_failure(Exception("HTTP 400 BadRequestError: invalid_request"), provider="gemini")
        self.assertEqual(f_400.category, "invalid_request")
        self.assertFalse(f_400.retryable)
        self.assertFalse(f_400.fallback_eligible)


if __name__ == "__main__":
    unittest.main()
