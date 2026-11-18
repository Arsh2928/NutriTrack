"""Strict 3-mode AI isolation, persistence, model enforcement, and Flask API tests."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nutritrack.agents import DeterministicNutriTrackChatClient, NutriTrackOrchestrator
from nutritrack.ai import (
    DATA_ONLY_BANNER,
    VALID_AI_MODES,
    AIProviderConfig,
    AIProviderManager,
    load_persisted_mode,
    normalize_ai_mode,
)
from nutritrack.core import external_nutrition
from nutritrack.state import RunLog, StateManager


class StaticProviderClient(DeterministicNutriTrackChatClient):
    def __init__(self, provider_name: str, model_name: str, *, error: Exception | None = None) -> None:
        super().__init__()
        self.provider_name = provider_name
        self.model = model_name
        self.error = error
        self.calls = 0

    async def _inner_get_response(self, *, messages, stream, options, **kwargs):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return await super()._inner_get_response(messages=messages, stream=stream, options=options, **kwargs)


class StrictThreeModeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.sm = StateManager(base_dir=self._tmp.name, ca1_log_path=None)
        self.rl = RunLog(path=f"{self._tmp.name}/runs.jsonl")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_01_valid_modes_are_strictly_gemini_groq_offline(self) -> None:
        """Automatic, Online, and hidden modes must be completely absent and rejected."""
        self.assertEqual(VALID_AI_MODES, ("gemini", "groq", "offline"))
        self.assertEqual(normalize_ai_mode("gemini"), "gemini")
        self.assertEqual(normalize_ai_mode("  GROQ "), "groq")
        self.assertEqual(normalize_ai_mode("Offline"), "offline")
        self.assertEqual(normalize_ai_mode(None), "offline")

        for banned in ("automatic", "auto", "online", "cloud", "data_only", "hybrid", "", "invalid"):
            with self.subTest(banned=banned):
                with self.assertRaises(ValueError):
                    normalize_ai_mode(banned)

    def test_02_from_env_ignores_legacy_automatic_persisted_or_env_modes(self) -> None:
        """Legacy 'automatic'/'online' in persisted settings or env must quarantine/ignore cleanly."""
        persisted = Path(self._tmp.name) / "ai_mode_settings.json"
        persisted.write_text(json.dumps({"selected_mode": "automatic"}), encoding="utf-8")

        self.assertIsNone(load_persisted_mode(self._tmp.name))
        self.assertFalse(persisted.exists())

        with patch.dict(
            os.environ,
            {
                "NUTRITRACK_AI_MODE": "automatic",
                "GEMINI_API_KEY": "AIzaSyEnvDefaultTest1234567890",
                "GROQ_API_KEY": "gsk_EnvDefaultTest1234567890",
            },
            clear=True,
        ):
            cfg = AIProviderConfig.from_env(persisted_dir=self._tmp.name)
            self.assertEqual(cfg.selected_mode, "offline")

        with patch.dict(
            os.environ,
            {
                "NUTRITRACK_AI_MODE": "gemini",
                "GEMINI_API_KEY": "AIzaSyEnvDefaultTest1234567890",
            },
            clear=True,
        ):
            cfg_explicit = AIProviderConfig.from_env(persisted_dir=self._tmp.name)
            self.assertEqual(cfg_explicit.selected_mode, "gemini")

        with patch.dict(os.environ, {"NUTRITRACK_AI_MODE": "online"}, clear=True):
            cfg_no_keys = AIProviderConfig.from_env(persisted_dir=self._tmp.name)
            self.assertEqual(cfg_no_keys.selected_mode, "offline")

    def test_03_offline_mode_zero_cloud_calls_and_blocks_external_food_api(self) -> None:
        """Offline mode executes deterministically with zero Gemini, Groq, or external food lookups."""
        cfg = AIProviderConfig(
            selected_mode="offline",
            gemini_api_key="AIzaSyOfflineBlockKey123456789",
            groq_api_key="gsk_OfflineBlockKey1234567890",
        )
        gem_client = StaticProviderClient("gemini", cfg.gemini_model)
        groq_client = StaticProviderClient("groq", cfg.groq_model)
        mgr = AIProviderManager(
            config=cfg,
            run_log=self.rl,
            persisted_dir=self._tmp.name,
            client_overrides={"gemini": gem_client, "groq": groq_client},
        )
        orch = NutriTrackOrchestrator(state_manager=self.sm, run_log=self.rl, ai_manager=mgr)

        res = orch.run_turn("I ate 2 boiled eggs.")
        self.assertEqual(res["status"], "success")
        self.assertEqual(gem_client.calls, 0)
        self.assertEqual(groq_client.calls, 0)
        self.assertEqual(res["ai_status"]["selected_mode"], "offline")
        self.assertEqual(res["ai_status"]["effective_label"], "Offline Mode Active")
        self.assertFalse(res["ai_status"]["fallback_active"])
        self.assertEqual(res["ai_status"]["banner"], DATA_ONLY_BANNER)
        self.assertFalse(res["is_llm"])

        coach = orch.run_turn("Design me a personalized keto recipe philosophy for travel.")
        self.assertEqual(coach["status"], "success")
        self.assertIn("OFFLINE MODE ACTIVE", coach["response"])
        self.assertEqual(gem_client.calls, 0)
        self.assertEqual(groq_client.calls, 0)

        with patch.dict(os.environ, {"NUTRITRACK_AI_MODE": "offline", "NUTRITRACK_ONLINE_LOOKUP": "1"}):
            self.assertFalse(external_nutrition.external_enabled())

    def test_04_model_rotation_removed_and_explicit_models_used_unchanged(self) -> None:
        """Model discovery/fallback dicts must not exist and configured models are sent verbatim."""
        from nutritrack.ai import provider_factory

        self.assertFalse(hasattr(provider_factory, "_FALLBACK_MODELS"))
        self.assertFalse(hasattr(provider_factory, "_RESOLVED_MODELS"))
        self.assertFalse(hasattr(provider_factory, "_is_model_rotatable_error"))

        cfg = AIProviderConfig(
            selected_mode="gemini",
            gemini_api_key="AIzaSyExactModelKey12345678901",
            gemini_model="gemini-custom-explicit-v1",
            groq_api_key="gsk_ExactModelKey123456789012",
            groq_model="groq-custom-explicit-v1",
        )
        factory = provider_factory.ProviderFactory()
        gem_maf = factory.build_client("gemini", cfg)
        groq_maf = factory.build_client("groq", cfg)
        self.assertEqual(gem_maf.model, "gemini-custom-explicit-v1")
        self.assertEqual(groq_maf.model, "groq-custom-explicit-v1")

    def test_05_flask_ai_mode_api_enforces_three_modes_and_persists(self) -> None:
        """Flask endpoints expose Gemini, Groq, Offline only and reject automatic/online with 400."""
        from nutritrack.web import app as web_app

        data_dir = os.path.join(self._tmp.name, "data")
        log_dir = os.path.join(self._tmp.name, "logs")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        env_patch = {
            "NUTRITRACK_DATA_DIR": data_dir,
            "NUTRITRACK_LOG_DIR": log_dir,
            "NUTRITRACK_AI_MODE": "offline",
            "GEMINI_API_KEY": "",
            "GOOGLE_API_KEY": "",
            "GROQ_API_KEY": "",
        }
        with patch.dict(os.environ, env_patch, clear=False):
            web_app.reset_ca2_runtime()
            client = web_app.app.test_client()

            for bad_mode in ("automatic", "auto", "online", "cloud", "invalid"):
                bad_resp = client.post("/api/nutritrack/ai/mode", json={"mode": bad_mode})
                self.assertEqual(bad_resp.status_code, 400, f"Expected 400 for {bad_mode}")

            for valid_mode in ("gemini", "groq", "offline"):
                ok_resp = client.post("/api/nutritrack/ai/mode", json={"mode": valid_mode})
                self.assertEqual(ok_resp.status_code, 200)
                body = ok_resp.get_json()
                self.assertEqual(body["selected_mode"], valid_mode)
                self.assertFalse(body["fallback_active"])

            diag = client.get("/api/nutritrack/ai/providers").get_json()
            self.assertFalse(diag["fallback_active"])
            self.assertNotIn("secondary_fallback", str(diag))


if __name__ == "__main__":
    unittest.main()
