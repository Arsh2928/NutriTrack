"""Automated tests for independent per-provider circuit recovery and manual retry probes."""

from __future__ import annotations

import tempfile
import unittest

from nutritrack.agents import DeterministicNutriTrackChatClient, NutriTrackOrchestrator
from nutritrack.ai import STATE_CLOSED, STATE_HALF_OPEN, STATE_OPEN, AIProviderConfig, AIProviderManager
from nutritrack.state import RunLog, StateManager


class ControllableClock:
    def __init__(self, start: float = 5000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class ToggleableCloudClient(DeterministicNutriTrackChatClient):
    def __init__(self, provider_name: str, model_name: str, *, online: bool = True, failure_exc: Exception | None = None) -> None:
        super().__init__()
        self.provider_name = provider_name
        self.model = model_name
        self.online = online
        self.failure_exc = failure_exc or ConnectionRefusedError("Network is unreachable")
        self.call_count = 0

    async def _inner_get_response(self, *, messages, stream, options, **kwargs):
        self.call_count += 1
        if not self.online:
            raise self.failure_exc
        return await super()._inner_get_response(messages=messages, stream=stream, options=options, **kwargs)


class ProviderRecoveryIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.clock = ControllableClock()

        async def _fake_sleep(delay: float) -> None:
            self.clock.advance(delay)

        self.fake_sleep = _fake_sleep
        self.sm = StateManager(base_dir=self._tmp.name, ca1_log_path=None)
        self.rl = RunLog(path=f"{self._tmp.name}/runs.jsonl")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_gemini_circuit_blocks_while_open_then_recovers_after_cooldown_without_touching_groq(self) -> None:
        """Gemini circuit opens on failure, blocks calls before cooldown, recovers in HALF_OPEN -> CLOSED."""
        cfg = AIProviderConfig(
            selected_mode="gemini",
            gemini_api_key="AIzaSyRecoveryTestKey111111111111",
            groq_api_key="gsk_RecoveryTestKey2222222222222",
            max_retries=0,
            circuit_failure_threshold=1,
            circuit_cooldown_seconds=60.0,
            health_check_interval_seconds=60.0,
        )
        gem_client = ToggleableCloudClient("gemini", cfg.gemini_model, online=False)
        groq_client = ToggleableCloudClient("groq", cfg.groq_model, online=True)

        mgr = AIProviderManager(
            config=cfg,
            run_log=self.rl,
            clock=self.clock,
            async_sleep=self.fake_sleep,
            persisted_dir=self._tmp.name,
            client_overrides={"gemini": gem_client, "groq": groq_client},
        )
        orch = NutriTrackOrchestrator(state_manager=self.sm, run_log=self.rl, ai_manager=mgr)

        # Turn 1: Gemini network down -> circuit trips OPEN, selected_mode remains gemini, Groq untouched
        r1 = orch.run_turn("I ate 2 boiled eggs.")
        self.assertEqual(r1["status"], "failed")
        self.assertEqual(gem_client.call_count, 1)
        self.assertEqual(groq_client.call_count, 0)
        self.assertEqual(mgr.circuits["gemini"].state, STATE_OPEN)
        self.assertEqual(mgr.circuits["groq"].state, STATE_CLOSED)
        self.assertEqual(r1["ai_status"]["selected_mode"], "gemini")

        # Turn 2 before cooldown: circuit blocks extra wasted call
        self.clock.advance(15.0)
        r2 = orch.run_turn("I ate 2 boiled eggs.")
        self.assertEqual(r2["status"], "failed")
        self.assertEqual(gem_client.call_count, 1)
        self.assertEqual(groq_client.call_count, 0)

        # Turn 3 after cooldown expires and Gemini is back -> HALF_OPEN probe closes circuit
        self.clock.advance(50.0)
        self.assertEqual(mgr.circuits["gemini"].state, STATE_HALF_OPEN)
        gem_client.online = True

        r3 = orch.run_turn("I ate 1 banana.")
        self.assertEqual(r3["status"], "success")
        self.assertEqual(groq_client.call_count, 0)
        self.assertEqual(mgr.circuits["gemini"].state, STATE_CLOSED)
        self.assertEqual(r3["ai_status"]["effective_label"], "Gemini Connected")
        self.assertFalse(r3["ai_status"]["fallback_active"])
        self.assertEqual(r3["state"]["consumed"]["calories"], 105.0)

    def test_manual_retry_probe_checks_only_selected_provider_and_manual_mode_switch_is_isolated(self) -> None:
        """Manual retry_providers probes only the selected provider; switching mode does not mutate circuits."""
        cfg = AIProviderConfig(
            selected_mode="groq",
            gemini_api_key="AIzaSyRecoveryTestKey333333333333",
            groq_api_key="gsk_RecoveryTestKey4444444444444",
            max_retries=0,
            circuit_failure_threshold=1,
            circuit_cooldown_seconds=60.0,
        )
        gem_client = ToggleableCloudClient("gemini", cfg.gemini_model, online=True)
        groq_client = ToggleableCloudClient("groq", cfg.groq_model, online=False)

        mgr = AIProviderManager(
            config=cfg,
            run_log=self.rl,
            clock=self.clock,
            async_sleep=self.fake_sleep,
            persisted_dir=self._tmp.name,
            client_overrides={"gemini": gem_client, "groq": groq_client},
        )
        orch = NutriTrackOrchestrator(state_manager=self.sm, run_log=self.rl, ai_manager=mgr)

        fail_res = orch.run_turn("What is my calorie goal?")
        self.assertEqual(fail_res["status"], "failed")
        self.assertEqual(mgr.circuits["groq"].state, STATE_OPEN)
        self.assertEqual(gem_client.call_count, 0)

        # Manual retry in Groq mode probes ONLY Groq (never Gemini)
        groq_client.online = True
        retried = mgr.retry_providers(force=True)
        self.assertEqual(gem_client.call_count, 0)
        self.assertGreater(groq_client.call_count, 1)
        self.assertEqual(mgr.circuits["groq"].state, STATE_CLOSED)
        self.assertEqual(retried["selected_mode"], "groq")
        self.assertEqual(retried["effective_label"], "Groq Connected")

        # Switching to offline does NOT probe either cloud provider
        gem_before = gem_client.call_count
        groq_before = groq_client.call_count
        mgr.set_mode("offline", persist=False)
        off_retried = mgr.retry_providers(force=True)
        self.assertEqual(off_retried["selected_mode"], "offline")
        self.assertEqual(gem_client.call_count, gem_before)
        self.assertEqual(groq_client.call_count, groq_before)


if __name__ == "__main__":
    unittest.main()
