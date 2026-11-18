"""Tests for Phase 4: the four specialist Microsoft Agent Framework agent modules."""

from __future__ import annotations

import json
import tempfile
import unittest

from agent_framework import Agent, FunctionTool

from nutritrack.agents import (
    DeterministicNutriTrackChatClient,
    FailingChatClient,
    ScriptedChatClient,
    create_goal_budget_agent,
    create_meal_intake_agent,
    create_meal_recommendation_agent,
    create_nutrition_analysis_agent,
    run_goal_budget,
    run_meal_intake,
    run_meal_recommendation,
    run_nutrition_analysis,
    text_response,
    tool_call_response,
)
from nutritrack.state import RunLog, StateManager
from nutritrack.tools import new_context
from nutritrack.tools.base import LLMUnavailableError


class SpecialistAgentsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.sm = StateManager(base_dir=self._tmp.name, ca1_log_path=None)
        self.rl = RunLog(path=f"{self._tmp.name}/runs.jsonl")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _ctx(self, task: str = "test", user_id: str = "demo_user"):
        return new_context(self.sm, self.rl, user_id=user_id, task=task)

    def test_each_specialist_builds_real_maf_agent_with_distinct_instructions_and_tools(self) -> None:
        ctx = self._ctx()
        client = DeterministicNutriTrackChatClient()
        builders = [
            (create_meal_intake_agent, "Meal Intake Agent", 5),
            (create_nutrition_analysis_agent, "Nutrition Analysis Agent", 4),
            (create_goal_budget_agent, "Goal and Budget Agent", 4),
            (create_meal_recommendation_agent, "Meal Recommendation Agent", 4),
        ]
        instructions = set()
        for factory, expected_name, min_tools in builders:
            agent = factory(ctx, client=client)
            self.assertIsInstance(agent, Agent)
            self.assertEqual(agent.name, expected_name)
            self.assertGreaterEqual(len(agent.default_options.get("tools") or []), min_tools)
            for t in agent.default_options.get("tools") or []:
                self.assertIsInstance(t, FunctionTool)
            instructions.add(agent.default_options.get("instructions"))
        self.assertEqual(len(instructions), 4)

    def test_meal_intake_logs_removes_and_flags_unknown_food_honestly(self) -> None:
        ctx = self._ctx("Log breakfast")
        res = run_meal_intake(ctx, "I ate 3 boiled eggs and 2 bananas for breakfast")
        self.assertEqual(res["status"], "success")
        tools = [t["tool"] for t in res["tools_called"]]
        self.assertIn("parse_meal_entry", tools)
        self.assertIn("lookup_food_nutrition", tools)
        self.assertIn("add_meal_to_state", tools)
        snap = self.sm.get("demo_user").snapshot()
        self.assertEqual(len(snap["meals"]), 2)
        self.assertEqual(snap["consumed"]["calories"], 444.0)

        # Removal uses parse + remove tools and updates shared state
        ctx_rm = self._ctx("Remove banana")
        res_rm = run_meal_intake(ctx_rm, "remove banana")
        self.assertEqual(res_rm["status"], "success")
        snap_after = self.sm.get("demo_user").snapshot()
        self.assertEqual(len(snap_after["meals"]), 1)
        self.assertEqual(snap_after["meals"][0]["food"], "egg")

        # Unknown food is NOT invented; agent asks follow-up question
        ctx_unk = self._ctx("Log unknown")
        res_unk = run_meal_intake(ctx_unk, "I ate 1 dragonfruit")
        self.assertEqual(res_unk["status"], "needs_input")
        self.assertTrue(res_unk["open_questions"])
        self.assertEqual(len(self.sm.get("demo_user").snapshot()["meals"]), 1)

    def test_meal_intake_works_with_scripted_maf_client(self) -> None:
        ctx = self._ctx("Scripted intake")
        scripted = ScriptedChatClient(
            scripts={
                "meal_intake": [
                    tool_call_response([("parse_meal_entry", {"text": "2 bananas"})], start_index=1),
                    tool_call_response(
                        [("add_meal_to_state", {"items": [{"food": "banana", "quantity": 2}]})],
                        start_index=2,
                    ),
                    text_response("Logged 2 bananas via scripted MAF turn."),
                ]
            }
        )
        res = run_meal_intake(ctx, "2 bananas", client=scripted)
        self.assertEqual(res["status"], "success")
        self.assertEqual([t["tool"] for t in res["tools_called"]], ["parse_meal_entry", "add_meal_to_state"])
        self.assertEqual(self.sm.get("demo_user").snapshot()["consumed"]["calories"], 210.0)

    def test_nutrition_analysis_agent_handles_daily_and_partial_weekly_data(self) -> None:
        st = self.sm.get("demo_user")
        st.set_target("calories", 2200)
        st.set_target("protein", 140)
        st.add_meals(
            [{"food": "egg", "quantity_str": "2 pieces", "calories": 156, "protein": 12, "carbs": 1.2, "fat": 10.6}],
            day="2026-11-10",
        )
        st.add_meals(
            [{"food": "chicken breast", "quantity_str": "200 g", "calories": 330, "protein": 62, "carbs": 0, "fat": 7.2}],
            day="2026-11-11",
        )

        ctx = self._ctx("Weekly insights 2026-11-11")
        res = run_nutrition_analysis(ctx, "Show my weekly nutrition insights up to 2026-11-11")
        self.assertEqual(res["status"], "success")
        tools = [t["tool"] for t in res["tools_called"]]
        self.assertGreaterEqual(len(tools), 3)
        self.assertIn("calculate_daily_totals", tools)
        self.assertIn("get_weekly_trends", tools)
        self.assertIn("compare_intake_with_targets", tools)
        # Fewer than 7 days must be stated honestly, never fabricated as 0
        self.assertIn("2 of 7 days", res["summary"])
        self.assertIn("NOT counted as zero", res["summary"])

    def test_goal_budget_agent_updates_targets_and_checks_proposed_meal(self) -> None:
        ctx = self._ctx("Set targets")
        res = run_goal_budget(ctx, "Set my daily calorie goal to 2100 kcal and protein goal to 140 g.")
        self.assertEqual(res["status"], "success")
        tools = [t["tool"] for t in res["tools_called"]]
        self.assertEqual(tools, ["set_calorie_target", "set_protein_target", "calculate_remaining_targets"])
        snap = self.sm.get("demo_user").snapshot()
        self.assertEqual(snap["targets"], {"calories": 2100.0, "protein": 140.0})

        # Now check a proposed meal against remaining budget
        ctx2 = self._ctx("Check meal fit")
        res2 = run_goal_budget(ctx2, "Does 2 bananas fit my budget?")
        self.assertEqual(res2["status"], "success")
        self.assertEqual(
            [t["tool"] for t in res2["tools_called"]],
            ["calculate_remaining_targets", "check_meal_against_budget"],
        )
        self.assertIn("Fits", res2["summary"])

    def test_meal_recommendation_agent_executes_all_four_tools_and_respects_diet(self) -> None:
        ctx = self._ctx("Recommend vegetarian meal")
        res = run_meal_recommendation(ctx, "Recommend a vegetarian meal fitting my remaining targets.")
        self.assertEqual(res["status"], "success")
        tools = [t["tool"] for t in res["tools_called"]]
        self.assertEqual(
            tools,
            [
                "get_recommendation_candidates",
                "estimate_meal_nutrition",
                "rank_meal_candidates",
                "validate_recommendation_fit",
            ],
        )
        snap = self.sm.get("demo_user").snapshot()
        self.assertTrue(snap["recommendations"])
        rec_foods = " ".join(it["food"].lower() for it in snap["recommendations"][0]["items"])
        self.assertNotIn("chicken", rec_foods)
        self.assertNotIn("egg", rec_foods)
        self.assertIn("estimates", res["summary"].lower())

    def test_specialist_reports_llm_unavailable_cleanly_and_writes_run_log(self) -> None:
        ctx = self._ctx("Broken LLM")
        res = run_meal_intake(
            ctx,
            "I ate 2 eggs",
            client=FailingChatClient(error=LLMUnavailableError("connection refused")),
        )
        self.assertEqual(res["status"], "failed")
        self.assertEqual(res["error_code"], "llm_unavailable")
        entries = self.rl.read(run_id=ctx.run_id)
        self.assertTrue(any(e["status"] == "failed" and e["error"]["code"] == "llm_unavailable" for e in entries))
        # Ensure nothing fake was written to meals
        self.assertEqual(self.sm.get("demo_user").snapshot()["meals"], [])


if __name__ == "__main__":
    unittest.main()
