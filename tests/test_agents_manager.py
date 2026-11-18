"""Tests for Phase 5: Nutrition Manager Agent and Manager-Worker Delegation Orchestration."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from agent_framework import Agent, FunctionTool

from nutritrack.agents import (
    DeterministicNutriTrackChatClient,
    FailingChatClient,
    NutriTrackOrchestrator,
    create_chat_client,
    create_nutrition_manager_agent,
    detect_llm_config,
)
from nutritrack.state import RunLog, StateManager
from nutritrack.tools import new_context
from nutritrack.tools.base import LLMUnavailableError


class NutritionManagerOrchestrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.sm = StateManager(base_dir=self._tmp.name, ca1_log_path=None)
        self.rl = RunLog(path=f"{self._tmp.name}/runs.jsonl")
        self.orch = NutriTrackOrchestrator(state_manager=self.sm, run_log=self.rl)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_manager_builds_maf_agent_with_five_manager_tools(self) -> None:
        ctx = new_context(self.sm, self.rl, task="inspect")
        manager = create_nutrition_manager_agent(ctx, client=DeterministicNutriTrackChatClient())
        self.assertIsInstance(manager, Agent)
        self.assertEqual(manager.name, "Nutrition Manager Agent")
        tools = manager.default_options.get("tools") or []
        self.assertEqual(len(tools), 5)
        for t in tools:
            self.assertIsInstance(t, FunctionTool)
        self.assertEqual(
            {t.name for t in tools},
            {
                "read_shared_state",
                "delegate_to_specialist",
                "validate_agent_results",
                "update_workflow_status",
                "append_run_log",
            },
        )

    def test_five_core_scenarios_and_selective_delegation(self) -> None:
        # Scenario 1: Log meal intake -> activates intake, analysis, budget (NOT recommendation)
        r1 = self.orch.run_turn("I ate 3 boiled eggs and 2 bananas.")
        self.assertEqual(r1["status"], "success")
        self.assertEqual(r1["workflow_status"], "completed")
        self.assertEqual(r1["activated_specialists"], ["meal_intake", "nutrition_analysis", "goal_budget"])
        self.assertNotIn("meal_recommendation", r1["activated_specialists"])
        self.assertEqual(r1["state"]["consumed"]["calories"], 444.0)
        self.assertEqual(len(r1["state"]["meals"]), 2)
        self.assertIn("validate_agent_results", r1["manager_tools"])

        # Scenario 2: Query consumed intake -> activates nutrition_analysis ONLY
        r2 = self.orch.run_turn("How many calories and protein did I eat today?")
        self.assertEqual(r2["status"], "success")
        self.assertEqual(r2["activated_specialists"], ["nutrition_analysis"])
        self.assertIn("444", r2["response"])
        self.assertIn("20.6", r2["response"])

        # Scenario 3: Update goals -> activates goal_budget ONLY and persists
        r3 = self.orch.run_turn("Set my daily calorie goal to 2100 kcal and protein goal to 140 g.")
        self.assertEqual(r3["status"], "success")
        self.assertEqual(r3["activated_specialists"], ["goal_budget"])
        self.assertEqual(r3["state"]["targets"], {"calories": 2100.0, "protein": 140.0})
        self.assertEqual(r3["state"]["remaining"], {"calories": 1656.0, "protein": 119.4})

        # Scenario 4: Recommend next meal -> activates goal_budget + meal_recommendation ONLY
        r4 = self.orch.run_turn("What meal should I eat next to hit my remaining calorie and protein target?")
        self.assertEqual(r4["status"], "success")
        self.assertEqual(r4["activated_specialists"], ["goal_budget", "meal_recommendation"])
        self.assertNotIn("meal_intake", r4["activated_specialists"])
        self.assertTrue(r4["state"]["recommendations"])
        self.assertLessEqual(r4["state"]["recommendations"][0]["calories"], r4["state"]["remaining"]["calories"])

        # Scenario 5: Weekly nutrition insights -> activates nutrition_analysis ONLY
        r5 = self.orch.run_turn("Show my weekly nutrition insights.")
        self.assertEqual(r5["status"], "success")
        self.assertEqual(r5["activated_specialists"], ["nutrition_analysis"])
        self.assertIn("1 of 7 days", r5["response"])

    def test_non_food_query_skips_all_specialists_and_stays_clean(self) -> None:
        r = self.orch.run_turn("What is the weather in Mumbai today?")
        self.assertEqual(r["status"], "success")
        self.assertEqual(r["workflow_status"], "completed")
        self.assertEqual(r["activated_specialists"], [])
        self.assertEqual(r["state"]["meals"], [])

    def test_state_persists_across_orchestrator_restarts(self) -> None:
        self.orch.run_turn("Set my daily calorie goal to 2000 kcal and protein goal to 135 g.")
        self.orch.run_turn("I ate 2 boiled eggs.")

        # Re-open from disk with fresh StateManager + RunLog + Orchestrator
        sm2 = StateManager(base_dir=self._tmp.name, ca1_log_path=None)
        rl2 = RunLog(path=f"{self._tmp.name}/runs.jsonl")
        orch2 = NutriTrackOrchestrator(state_manager=sm2, run_log=rl2)
        res = orch2.run_turn("How many calories and protein did I eat today?")
        self.assertEqual(res["state"]["targets"], {"calories": 2000.0, "protein": 135.0})
        self.assertEqual(res["state"]["consumed"]["calories"], 156.0)

    def test_specialist_llm_failure_stops_workflow_honestly_without_fabricating(self) -> None:
        # Failing client fails ONLY when meal_recommendation agent runs
        failing = FailingChatClient(
            error=LLMUnavailableError("upstream timeout"),
            fail_for_agents={"meal_recommendation"},
            fallback_client=DeterministicNutriTrackChatClient(),
        )
        res = self.orch.run_turn(
            "What meal should I eat next to hit my remaining calorie and protein target?",
            client=failing,
        )
        self.assertEqual(res["status"], "failed")
        self.assertEqual(res["workflow_status"], "failed")
        self.assertEqual(res["error_code"], "llm_unavailable")
        self.assertIn("llm_unavailable", res["response"])
        self.assertEqual(res["state"]["recommendations"], [])
        self.assertTrue(any(e["status"] == "failed" for e in res["run_log_entries"]))

    def test_manager_llm_failure_handles_gracefully(self) -> None:
        res = self.orch.run_turn(
            "I ate 1 banana",
            client=FailingChatClient(error=LLMUnavailableError("API unreachable")),
        )
        self.assertEqual(res["status"], "failed")
        self.assertEqual(res["workflow_status"], "failed")
        self.assertEqual(res["error_code"], "llm_unavailable")
        self.assertEqual(res["state"]["meals"], [])

    def test_runtime_config_and_require_llm_guard(self) -> None:
        with patch.dict(os.environ, {"NUTRITRACK_DATA_DIR": self._tmp.name}, clear=True):
            cfg = detect_llm_config()
            self.assertEqual(cfg["mode"], "offline_demo")
            self.assertFalse(cfg["is_llm"])
            with self.assertRaises(LLMUnavailableError):
                create_chat_client(require_llm=True)

        with patch.dict(
            os.environ,
            {"NUTRITRACK_DATA_DIR": self._tmp.name, "GEMINI_API_KEY": "AIzaSyTestRedactedKey1234567890123456"},
            clear=True,
        ):
            cfg_no_mode = detect_llm_config()
            self.assertEqual(cfg_no_mode["mode"], "offline_demo")
            self.assertFalse(cfg_no_mode["is_llm"])

        with patch.dict(
            os.environ,
            {
                "NUTRITRACK_DATA_DIR": self._tmp.name,
                "NUTRITRACK_AI_MODE": "gemini",
                "GEMINI_API_KEY": "AIzaSyTestRedactedKey1234567890123456",
            },
            clear=True,
        ):
            client, info = create_chat_client()
            self.assertEqual(info["mode"], "llm")
            self.assertEqual(info["provider"], "gemini")
            self.assertTrue(info["is_llm"])
            # Make sure RunLog never contains the test key if logged
            self.orch.run_turn("I ate 1 egg")
            raw_log = open(f"{self._tmp.name}/runs.jsonl", encoding="utf-8").read()
            self.assertNotIn("AIzaSyTestRedactedKey1234567890123456", raw_log)


if __name__ == "__main__":
    unittest.main()
