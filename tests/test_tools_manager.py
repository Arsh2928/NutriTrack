"""Phase 3 tests: tool plumbing (binding, logging, error conversion) and the Nutrition Manager's tools.

The specialist "runners" used here are TEST DOUBLES for the language-model agents that Phase 4-5 add.
They call the REAL tools, so every log line, state change and failure in these tests is genuine; only the
"thinking" between tool calls is scripted.
"""
import inspect
import unittest
from unittest import mock

from nutritrack.state import NutritionState
from nutritrack.tools import AGENTS, TOOLS_BY_AGENT, LLMUnavailableError, agent_tool, bind, tools_for
from nutritrack.tools.base import ok
from tests.support import ToolCase, make_entry

# The tools named in the CA2 brief, per agent.
BRIEF_TOOLS = {
    "meal_intake": {"parse_meal_entry", "lookup_food_nutrition", "add_meal_to_state", "remove_meal_from_state"},
    "nutrition_analysis": {"calculate_daily_totals", "calculate_macro_split", "get_weekly_trends", "compare_intake_with_targets"},
    "goal_budget": {"set_calorie_target", "set_protein_target", "calculate_remaining_targets", "check_meal_against_budget"},
    "meal_recommendation": {"get_recommendation_candidates", "estimate_meal_nutrition", "rank_meal_candidates",
                            "validate_recommendation_fit"},
    "nutrition_manager": {"read_shared_state", "delegate_to_specialist", "validate_agent_results",
                          "update_workflow_status", "append_run_log"},
}


def item(food, quantity=None, unit=None):
    return {"food": food, "quantity": quantity, "unit": unit}


# ===========================================================================
class TestToolInventory(ToolCase):

    def test_every_tool_named_in_the_brief_exists_on_the_right_agent(self):
        for agent, expected in BRIEF_TOOLS.items():
            with self.subTest(agent=agent):
                have = {t.__name__ for t in TOOLS_BY_AGENT[agent]}
                self.assertTrue(expected <= have, f"missing: {expected - have}")
        self.assertEqual({t.__name__ for t in TOOLS_BY_AGENT["meal_intake"]} - BRIEF_TOOLS["meal_intake"], {"register_custom_food"})
        for agent in ("nutrition_analysis", "goal_budget", "meal_recommendation", "nutrition_manager"):
            self.assertEqual({t.__name__ for t in TOOLS_BY_AGENT[agent]}, BRIEF_TOOLS[agent])

    def test_every_agent_has_at_least_two_tools_and_each_tool_belongs_to_exactly_one_agent(self):
        seen = {}
        for agent, tools in TOOLS_BY_AGENT.items():
            self.assertGreaterEqual(len(tools), 2, agent)
            for t in tools:
                self.assertEqual(t.agent_key, agent)
                self.assertNotIn(t.__name__, seen, "tool shared between agents")
                seen[t.__name__] = agent
        self.assertEqual(set(AGENTS), set(TOOLS_BY_AGENT))
        self.assertEqual(len(seen), 22)

    def test_agents_receive_tools_without_the_context_parameter(self):
        for agent in TOOLS_BY_AGENT:
            for original, bound in zip(TOOLS_BY_AGENT[agent], tools_for(agent, self.ctx)):
                with self.subTest(tool=original.__name__):
                    params = list(inspect.signature(bound).parameters)
                    self.assertNotIn("ctx", params)
                    self.assertEqual(params, list(inspect.signature(original).parameters)[1:])
                    self.assertEqual((bound.__name__, bound.__doc__), (original.__name__, original.__doc__))
                    self.assertNotIn("ctx", bound.__annotations__)
                    for annotation in bound.__annotations__.values():
                        self.assertNotIsInstance(annotation, str, "annotations must be real types (schema generation)")

    def test_every_tool_is_documented_for_the_model(self):
        for tools in TOOLS_BY_AGENT.values():
            for t in tools:
                with self.subTest(tool=t.__name__):
                    doc = t.__doc__ or ""
                    self.assertGreater(len(doc), 80)
                    if len(inspect.signature(t).parameters) > 1:
                        self.assertIn("Args:", doc)

    def test_a_bound_tool_runs_against_its_own_context(self):
        mine, other = self.new_ctx(user_id="demo_user"), self.new_ctx(user_id="alice")
        bound_mine = {t.__name__: t for t in tools_for("meal_intake", mine)}
        bound_other = {t.__name__: t for t in tools_for("meal_intake", other)}
        bound_mine["add_meal_to_state"]([item("egg", 3)])
        self.assertEqual(self.manager.get("demo_user").day_totals()["calories"], 234.0)
        self.assertEqual(self.manager.get("alice").day_totals()["calories"], 0.0)
        bound_other["add_meal_to_state"]([item("banana", 1)])
        self.assertEqual(self.manager.get("alice").day_totals()["calories"], 105.0)


# ===========================================================================
class TestToolLoggingAndSafety(ToolCase):

    def test_every_call_writes_exactly_one_complete_log_entry(self):
        ctx = self.new_ctx(task="log my breakfast")
        self.call("lookup_food_nutrition", "egg", 3, ctx=ctx)
        entries = self.logged(run_id=ctx.run_id)
        self.assertEqual(len(entries), 1)
        e = entries[0]
        self.assertEqual((e["agent"], e["tool"], e["status"], e["task"], e["user_id"]),
                         ("Meal Intake Agent", "lookup_food_nutrition", "success", "log my breakfast", "demo_user"))
        self.assertIn('"food":"egg"', e["input_summary"])
        self.assertIn("234", e["output_summary"])
        self.assertGreaterEqual(e["duration_ms"], 0)
        self.assertNotIn("state_manager", e["input_summary"], "the context object must never be logged")

    def test_failed_calls_log_the_error_code_and_message(self):
        self.call("lookup_food_nutrition", "idli", 2)
        e = self.logged()[0]
        self.assertEqual((e["status"], e["error"]["code"]), ("failed", "unknown_food"))
        self.assertIn("idli", e["error"]["message"])

    def test_each_call_is_attributed_to_the_agent_that_owns_the_tool(self):
        self.call("calculate_daily_totals"); self.call("calculate_remaining_targets")
        self.call("get_recommendation_candidates", remaining_calories=500); self.call("read_shared_state")
        self.assertEqual([e["agent"] for e in self.logged()],
                         ["Nutrition Analysis Agent", "Goal and Budget Agent", "Meal Recommendation Agent", "Nutrition Manager Agent"])

    def test_the_outcome_of_each_call_is_recorded_in_the_shared_state(self):
        self.call("lookup_food_nutrition", "egg", 1)
        self.call("lookup_food_nutrition", "idli", 1)
        rec = self.state.get_agent_results()["meal_intake"]
        self.assertEqual((rec["tool_calls"], rec["last_tool"], rec["status"], rec["code"], rec["run_id"]),
                         (2, "lookup_food_nutrition", "failed", "unknown_food", self.ctx.run_id))

    def test_secrets_passed_to_a_tool_never_reach_the_log(self):
        token = "ghp_" + "Zy9Xw8Vu7T" * 4
        self.call("parse_meal_entry", f"3 eggs api_key={token}")
        raw = self.run_log.path.read_text(encoding="utf-8")
        self.assertNotIn(token, raw)
        self.assertIn("***", raw)

    def test_a_crash_inside_a_tool_becomes_a_failed_result_not_an_exception(self):
        with mock.patch.object(NutritionState, "day_totals", side_effect=RuntimeError("boom")):
            res = self.call("calculate_daily_totals")
        self.assertEqual((res["status"], res["code"]), ("failed", "internal_error"))
        self.assertIn("RuntimeError: boom", res["message"])
        self.assertEqual(self.logged(status="failed")[0]["error"]["code"], "internal_error")

    def test_wrong_arguments_become_a_failed_result(self):
        res = self.call("calculate_daily_totals", bogus=1)
        self.assertEqual((res["status"], res["code"]), ("failed", "bad_arguments"))
        self.assertEqual(self.call("add_meal_to_state")["code"], "bad_arguments")
        self.assertEqual(len(self.logged(status="failed")), 2)

    def test_a_failed_save_is_reported_and_leaves_the_state_unchanged(self):
        self.call("add_meal_to_state", [item("egg", 1)])
        before = self.state.snapshot()["meals"]
        with mock.patch.object(self.state, "_persist", side_effect=OSError("disk full")):
            res = self.call("add_meal_to_state", [item("banana", 2)], ctx=self.new_ctx())
        self.assertEqual((res["status"], res["code"]), ("failed", "state_update_failed"))
        self.assertFalse(res["needs_user_input"])
        self.assertIn("disk full", res["message"])
        self.assertEqual(self.state.snapshot()["meals"], before)
        self.assertEqual(self.logged(status="failed")[0]["error"]["code"], "state_update_failed")

    def test_a_tool_returning_something_else_than_an_envelope_is_caught(self):
        @agent_tool("meal_intake")
        def broken_tool(ctx):
            """Returns the wrong thing on purpose."""
            return "not a dict"
        res = broken_tool(self.ctx)
        self.assertEqual((res["status"], res["code"]), ("failed", "invalid_tool_result"))

    def test_a_broken_run_log_never_breaks_a_tool(self):
        from pathlib import Path
        from nutritrack.state import RunLog
        blocker = Path(self.tmp) / "file"; blocker.write_text("x")
        ctx = self._new_context(self.manager, RunLog(blocker / "agent_runs.jsonl"), task="t")
        with self.assertLogs("nutritrack.state.run_log", level="ERROR"):
            res = self.call("lookup_food_nutrition", "egg", 1, ctx=ctx)
        self.assertEqual(res["status"], "success")

    def test_no_tool_ever_raises_whatever_it_is_given(self):
        """Fuzz: all 22 tools, 11 kinds of garbage. Every call must return a result envelope."""
        garbage = [(), (None,), (123,), ("",), ([],), ({},), ("x" * 5000,), (None, None), ([{"food": None}],),
                   ("a", "b", "c"), (float("nan"),)]
        for name, tool in self.tools.items():
            for args in garbage:
                with self.subTest(tool=name, args=str(args)[:25]):
                    res = tool(self.new_ctx(), *args)
                    self.assertIn(res["status"], ("success", "failed"))
                    self.assertIn("code", res)
                    self.assertIsInstance(res["message"], str)


# ===========================================================================
class TestManagerTools(ToolCase):

    def test_read_shared_state(self):
        res = self.call("read_shared_state")
        self.assertEqual(set(res["state"]), {"targets", "consumed", "remaining", "workflow_status"})
        self.assertEqual(res["state"]["remaining"], {"calories": 2300.0, "protein": 150.0})
        self.call("add_meal_to_state", [item("egg", 3)])
        full = self.call("read_shared_state", ["meals", "preferences", "recommendations", "weekly_history", "agent_results"])
        self.assertEqual((len(full["state"]["meals"]), full["meal_count"]), (1, 1))
        self.assertEqual(full["targets_source"]["calories"], "demo_default")
        self.assertEqual(self.call("read_shared_state", date="yesterday")["meal_count"], 0)

    def test_read_shared_state_rejects_unknown_sections_and_dates(self):
        self.assertEqual(self.call("read_shared_state", ["secrets"])["code"], "invalid_input")
        self.assertEqual(self.call("read_shared_state", date="2099-01-01")["code"], "invalid_input")

    def test_update_workflow_status_walks_through_the_workflow(self):
        for status in ("planning", "delegating", "validating", "compiling", "completed"):
            self.assertEqual(self.call("update_workflow_status", status, "step")["current"], status)
        self.assertEqual([h["to"] for h in self.state.get_workflow()["history"]],
                         ["planning", "delegating", "validating", "compiling", "completed"])
        res = self.call("update_workflow_status", "exploding")
        self.assertEqual((res["status"], res["code"]), ("failed", "invalid_input"))
        self.assertEqual(self.state.snapshot()["workflow_status"], "completed")

    def test_append_run_log_writes_exactly_one_entry(self):
        res = self.call("append_run_log", "Nutrition Manager Agent", "plan", "plan", "success",
                        "request: log breakfast", "will delegate to meal_intake then goal_budget")
        self.assertEqual(res["code"], "logged")
        entries = self.logged(run_id=self.ctx.run_id)
        self.assertEqual(len(entries), 1, "the logging tool must not log itself a second time")
        self.assertEqual((entries[0]["agent"], entries[0]["tool"], entries[0]["status"]), ("Nutrition Manager Agent", "plan", "success"))

    def test_append_run_log_failed_entries_and_validation_and_secrets(self):
        self.call("append_run_log", "Nutrition Manager Agent", "compile", "compile", "failed", error_code="oops", error_message="bad")
        self.assertEqual(self.logged()[0]["error"], {"code": "oops", "message": "bad"})
        self.assertEqual(self.call("append_run_log", "a", "t", "x", "maybe")["code"], "invalid_input")
        token = "ghp_" + "Qq1Ww2Ee3R" * 4
        self.call("append_run_log", "a", "t", "x", "success", f"key {token}")
        self.assertNotIn(token, self.run_log.path.read_text(encoding="utf-8"))
        self.assertEqual(len(self.logged()), 2)


# ===========================================================================
class ManagerFlowCase(ToolCase):
    """Helpers for scripting a manager-worker run with real tools behind test-double specialists."""

    def runner(self, ctx, agent_key, script):
        bound = {t.__name__: t for t in tools_for(agent_key, ctx)}
        return lambda task: script(bound, task)

    def setup_run(self, **scripts):
        ctx = self.new_ctx(task="Add my breakfast and calculate how much protein I have consumed")
        ctx.specialists.update({key: self.runner(ctx, key, fn) for key, fn in scripts.items()})
        return ctx

    @staticmethod
    def intake(bound, task):
        parsed = bound["parse_meal_entry"](task)
        items = [{"food": i["food"], "quantity": i["quantity"], "unit": i["unit"]} for i in parsed["items"]]
        added = bound["add_meal_to_state"](items, meal_type=parsed["meal_type"])
        return added["message"]

    @staticmethod
    def analysis(bound, task):
        totals = bound["calculate_daily_totals"]()
        split = bound["calculate_macro_split"]()
        return f"{totals['message']} {split['message']}"

    @staticmethod
    def budget(bound, task):
        remaining = bound["calculate_remaining_targets"]()
        check = bound["check_meal_against_budget"]([item("banana", 1)])
        return f"{remaining['message']} {check['message']}"


class TestDelegation(ManagerFlowCase):

    def test_successful_delegation_records_the_result_and_the_workflow_status(self):
        ctx = self.setup_run(meal_intake=self.intake)
        res = self.call("delegate_to_specialist", "meal_intake", "Log: 3 boiled eggs and 2 bananas for breakfast", ctx=ctx)
        self.assertEqual((res["status"], res["specialist"]), ("success", "meal_intake"))
        self.assertIn("Saved 2 food(s), 444 kcal", res["specialist_response"])
        rec = self.state.get_agent_results()["meal_intake"]
        self.assertEqual((rec["specialist_status"], rec["run_id"]), ("success", ctx.run_id))
        self.assertIn("444 kcal", rec["summary"])
        self.assertEqual(self.state.snapshot()["workflow_status"], "delegating")
        self.assertEqual(self.state.day_totals()["calories"], 444.0)

    def test_specialists_can_be_named_loosely(self):
        ctx = self.setup_run(meal_intake=lambda b, t: "done", goal_budget=lambda b, t: "done")
        for name, key in (("Meal Intake Agent", "meal_intake"), ("meal-intake", "meal_intake"), ("intake", "meal_intake"),
                          ("Goal and Budget Agent", "goal_budget"), ("budget", "goal_budget")):
            self.assertEqual(self.call("delegate_to_specialist", name, "x", ctx=ctx)["specialist"], key)

    def test_unknown_specialist_and_bad_tasks_are_rejected(self):
        ctx = self.setup_run(meal_intake=lambda b, t: "done")
        self.assertEqual(self.call("delegate_to_specialist", "chef", "x", ctx=ctx)["code"], "unknown_specialist")
        self.assertEqual(self.call("delegate_to_specialist", "nutrition_manager", "x", ctx=ctx)["code"], "unknown_specialist")
        for bad in ("", "   ", None, "x" * 2001):
            self.assertEqual(self.call("delegate_to_specialist", "meal_intake", bad, ctx=ctx)["code"], "invalid_input")

    def test_a_specialist_without_a_runner_is_reported_unavailable(self):
        res = self.call("delegate_to_specialist", "goal_budget", "x")
        self.assertEqual((res["status"], res["code"]), ("failed", "specialist_unavailable"))
        self.assertEqual(self.state.get_agent_results()["goal_budget"]["specialist_status"], "failed")

    def test_a_crashing_specialist_is_a_failure_and_never_a_made_up_result(self):
        def crash(bound, task):
            bound["calculate_daily_totals"]()
            raise ValueError("model returned garbage")
        ctx = self.setup_run(nutrition_analysis=crash)
        res = self.call("delegate_to_specialist", "nutrition_analysis", "totals please", ctx=ctx)
        self.assertEqual((res["status"], res["code"]), ("failed", "specialist_failed"))
        self.assertNotIn("specialist_response", res)
        rec = self.state.get_agent_results()["nutrition_analysis"]
        self.assertEqual((rec["specialist_status"], rec["code"]), ("failed", "specialist_failed"))
        self.assertIn("garbage", rec["message"])
        self.assertEqual(self.logged(tool="delegate_to_specialist")[0]["status"], "failed")

    def test_an_unreachable_model_is_reported_as_llm_unavailable(self):
        for exc in (LLMUnavailableError("no route"), ConnectionError("refused"), TimeoutError("slow")):
            with self.subTest(exc=type(exc).__name__):
                def down(bound, task, exc=exc):
                    raise exc
                ctx = self.setup_run(meal_intake=down)
                res = self.call("delegate_to_specialist", "meal_intake", "x", ctx=ctx)
                self.assertEqual((res["status"], res["code"]), ("failed", "llm_unavailable"))

    def test_an_empty_answer_is_a_failure(self):
        for empty in ("", "   ", {"summary": ""}, {}):
            ctx = self.setup_run(meal_intake=lambda b, t, e=empty: e)
            self.assertEqual(self.call("delegate_to_specialist", "meal_intake", "x", ctx=ctx)["code"], "empty_response")

    def test_a_runner_may_answer_with_a_summary_dict(self):
        ctx = self.setup_run(meal_intake=lambda b, t: {"summary": "logged 3 eggs"})
        self.assertEqual(self.call("delegate_to_specialist", "meal_intake", "x", ctx=ctx)["specialist_response"], "logged 3 eggs")


class TestValidation(ManagerFlowCase):

    def _delegate(self, ctx, key, task="do it"):
        return self.call("delegate_to_specialist", key, task, ctx=ctx)

    def test_a_clean_run_validates(self):
        ctx = self.setup_run(meal_intake=self.intake, goal_budget=self.budget)
        self._delegate(ctx, "meal_intake", "3 boiled eggs and 2 bananas for breakfast")
        self._delegate(ctx, "goal_budget")
        res = self.call("validate_agent_results", ["meal_intake", "goal_budget"], ctx=ctx)
        self.assertEqual((res["status"], res["code"], res["valid"], res["issues"]), ("success", "valid", True, []))
        self.assertEqual(res["checked_specialists"], ["meal_intake", "goal_budget"])

    def test_a_specialist_that_reported_nothing_fails_validation(self):
        ctx = self.setup_run(meal_intake=self.intake)
        res = self.call("validate_agent_results", ["meal_intake"], ctx=ctx)
        self.assertEqual((res["status"], res["code"], res["valid"]), ("failed", "validation_failed", False))
        self.assertIn("reported no result", res["issues"][0])

    def test_a_result_left_over_from_an_earlier_run_is_not_accepted(self):
        old = self.setup_run(meal_intake=self.intake)
        self._delegate(old, "meal_intake", "2 eggs")
        fresh = self.new_ctx(task="a different request")                    # a NEW run, same user
        res = self.call("validate_agent_results", ["meal_intake"], ctx=fresh)
        self.assertFalse(res["valid"])
        self.assertIn("earlier run", res["issues"][0])

    def test_a_failed_specialist_fails_validation_with_its_reason(self):
        def crash(bound, task):
            raise RuntimeError("tool schema mismatch")
        ctx = self.setup_run(nutrition_analysis=crash)
        self._delegate(ctx, "nutrition_analysis")
        res = self.call("validate_agent_results", ["nutrition_analysis"], ctx=ctx)
        self.assertFalse(res["valid"])
        self.assertTrue(any("Nutrition Analysis Agent failed" in i and "tool schema mismatch" in i for i in res["issues"]))

    def test_a_specialist_that_never_finished_is_caught(self):
        ctx = self.new_ctx()
        self.state.update_agent_result("meal_intake", specialist_status="running", run_id=ctx.run_id)
        self.assertIn("never reported completion", self.call("validate_agent_results", ["meal_intake"], ctx=ctx)["issues"][0])

    def test_unknown_specialist_names_are_reported(self):
        res = self.call("validate_agent_results", ["chef"])
        self.assertFalse(res["valid"])
        self.assertIn("'chef' is not a known specialist", res["issues"][0])

    def test_user_input_needs_are_open_questions_not_failures(self):
        def intake_with_unknown(bound, task):
            bound["add_meal_to_state"]([item("egg", 2), item("idli", 2)])
            return "saved eggs; idli is unknown"
        ctx = self.setup_run(meal_intake=intake_with_unknown)
        self._delegate(ctx, "meal_intake")
        res = self.call("validate_agent_results", ["meal_intake"], ctx=ctx)
        self.assertEqual((res["status"], res["code"], res["valid"], res["needs_user_input"]), ("success", "valid_with_questions", True, True))
        self.assertEqual(res["issues"], [])
        self.assertEqual(self.state.day_totals()["calories"], 156.0)
        q = res["open_questions"][0]
        self.assertEqual((q["agent"], q["tool"], q["code"]), ("Meal Intake Agent", "add_meal_to_state", "partial"))
        self.assertIn("idli", q["message"])
        self.assertIn("calories", q["question"])                    # the question to put to the user
        self.assertEqual(len(self.state.get_open_questions(ctx.run_id)), 1)
        self.state.reset_run_data()
        self.assertEqual(self.state.get_open_questions(), [])

    def test_a_failed_lookup_in_the_log_is_surfaced_even_if_the_specialist_claims_success(self):
        def sloppy(bound, task):
            bound["lookup_food_nutrition"]("idli", 2)
            return "all done, everything logged"                                  # not true
        ctx = self.setup_run(meal_intake=sloppy)
        self._delegate(ctx, "meal_intake")
        res = self.call("validate_agent_results", ["meal_intake"], ctx=ctx)
        self.assertEqual([q["code"] for q in res["open_questions"]], ["unknown_food"])
        self.assertTrue(res["needs_user_input"])

    def test_a_system_failure_inside_a_specialist_fails_validation(self):
        def intake(bound, task):
            with mock.patch.object(self.state, "_persist", side_effect=OSError("disk full")):
                bound["add_meal_to_state"]([item("egg", 3)])
            return "I logged your eggs"                                            # the model's claim is false
        ctx = self.setup_run(meal_intake=intake)
        self._delegate(ctx, "meal_intake")
        res = self.call("validate_agent_results", ["meal_intake"], ctx=ctx)
        self.assertFalse(res["valid"])
        self.assertTrue(any("state_update_failed" in i for i in res["issues"]))
        self.assertEqual(self.state.get_day_meals(), [])

    def test_an_earlier_validation_failure_does_not_poison_a_later_check(self):
        ctx = self.setup_run(meal_intake=self.intake)
        self.assertFalse(self.call("validate_agent_results", ["meal_intake"], ctx=ctx)["valid"])
        self._delegate(ctx, "meal_intake", "3 boiled eggs")
        self.assertTrue(self.call("validate_agent_results", ["meal_intake"], ctx=ctx)["valid"])

    def test_state_corruption_is_detected(self):
        self.call("add_meal_to_state", [item("egg", 3)])
        meals = self.state._doc["days"][self.clock.today]["meals"]
        meals.append(dict(meals[0]))                                                # same meal_id twice
        res = self.call("validate_agent_results")
        self.assertFalse(res["valid"])
        self.assertTrue(any("Duplicate meal ids" in i for i in res["issues"]))

    def test_recommendations_that_exceed_what_is_left_are_caught_and_stale_ones_warn(self):
        rec = {"id": "c01", "name": "huge meal", "calories": 9999.0, "protein": 1.0}
        self.state.set_recommendations([rec])
        res = self.call("validate_agent_results")
        self.assertFalse(res["valid"])
        self.assertTrue(any("exceeds" in i for i in res["issues"]))
        self.state.set_recommendations([{**rec, "calories": 300.0}])
        self.call("add_meal_to_state", [item("egg", 1)])                            # remaining changes -> stale
        stale = self.call("validate_agent_results")
        self.assertTrue(stale["valid"])
        self.assertTrue(any("out of date" in w for w in stale["warnings"]))


# ===========================================================================
class TestManagerWorkerRun(ManagerFlowCase):
    """The example from the brief, end to end: 'Add my breakfast and calculate how much protein I have consumed.'"""

    def _run_manager(self, ctx, request, plan):
        """A scripted manager: exactly what the LLM manager will be asked to do in Phase 5."""
        manager = {t.__name__: t for t in tools_for("nutrition_manager", ctx)}
        manager["update_workflow_status"]("planning")
        manager["append_run_log"]("Nutrition Manager Agent", request, "plan", "success", request, f"delegate to {[p[0] for p in plan]}")
        results = {}
        for specialist, task in plan:
            results[specialist] = manager["delegate_to_specialist"](specialist, task)
        manager["update_workflow_status"]("validating")
        validation = manager["validate_agent_results"]([p[0] for p in plan])
        manager["update_workflow_status"]("compiling" if validation["valid"] else "failed")
        return results, validation

    def test_breakfast_then_protein_then_remaining(self):
        ctx = self.setup_run(meal_intake=self.intake, nutrition_analysis=self.analysis, goal_budget=self.budget)
        plan = [("meal_intake", "3 boiled eggs and 2 bananas for breakfast"),
                ("nutrition_analysis", "How much protein have I consumed today?"),
                ("goal_budget", "How much do I have left, and does a banana fit?")]
        results, validation = self._run_manager(ctx, "Add my breakfast and calculate how much protein I have consumed", plan)
        self.assertTrue(all(r["status"] == "success" for r in results.values()))
        self.assertTrue(validation["valid"], validation)
        self.assertEqual(self.state.snapshot()["workflow_status"], "compiling")
        self.assertEqual([h["to"] for h in self.state.get_workflow()["history"]],
                         ["planning", "delegating", "delegating", "delegating", "validating", "compiling"])
        # the specialists really read the SAME state the intake agent wrote
        self.assertIn("20.6 g protein", results["nutrition_analysis"]["specialist_response"])   # eggs 18.0 + bananas 2.6
        self.assertIn("1856 kcal left", results["goal_budget"]["specialist_response"])
        self.assertEqual(self.state.snapshot()["remaining"], {"calories": 1856.0, "protein": 129.4})

    def test_the_log_shows_which_agent_ran_which_tool_and_every_worker_used_two_or_more_tools(self):
        ctx = self.setup_run(meal_intake=self.intake, nutrition_analysis=self.analysis, goal_budget=self.budget)
        self._run_manager(ctx, "req", [("meal_intake", "2 eggs"), ("nutrition_analysis", "x"), ("goal_budget", "y")])
        entries = self.logged(run_id=ctx.run_id)
        by_agent = {}
        for e in entries:
            by_agent.setdefault(e["agent"], []).append(e["tool"])
        for agent in ("Meal Intake Agent", "Nutrition Analysis Agent", "Goal and Budget Agent"):
            self.assertGreaterEqual(len(set(by_agent[agent])), 2, f"{agent} used {by_agent.get(agent)}")
        self.assertEqual(by_agent["Meal Intake Agent"], ["parse_meal_entry", "add_meal_to_state"])
        self.assertEqual(by_agent["Nutrition Manager Agent"].count("delegate_to_specialist"), 3)
        self.assertNotIn("Meal Recommendation Agent", by_agent, "only the relevant specialists are delegated to")
        self.assertEqual({e["run_id"] for e in entries}, {ctx.run_id})
        self.assertEqual([e["timestamp"] for e in entries], sorted(e["timestamp"] for e in entries))

    def test_when_a_specialist_fails_the_manager_gets_a_failure_and_a_failed_validation(self):
        def broken_analysis(bound, task):
            bound["calculate_daily_totals"]()
            raise ConnectionError("model endpoint unreachable")
        ctx = self.setup_run(meal_intake=self.intake, nutrition_analysis=broken_analysis)
        results, validation = self._run_manager(ctx, "req", [("meal_intake", "3 boiled eggs"), ("nutrition_analysis", "totals")])
        self.assertEqual(results["meal_intake"]["status"], "success")
        self.assertEqual((results["nutrition_analysis"]["status"], results["nutrition_analysis"]["code"]), ("failed", "llm_unavailable"))
        self.assertFalse(validation["valid"])
        self.assertEqual(self.state.snapshot()["workflow_status"], "failed")
        self.assertEqual(self.state.day_totals()["calories"], 234.0, "the successful specialist's work is kept")
        self.assertTrue(self.logged(tool="delegate_to_specialist", status="failed"))

    def test_the_same_request_twice_in_one_run_logs_the_meal_once(self):
        ctx = self.setup_run(meal_intake=self.intake)
        self._run_manager(ctx, "req", [("meal_intake", "3 boiled eggs"), ("meal_intake", "3 boiled eggs")])
        self.assertEqual(self.state.day_totals()["meal_count"], 1)

    def test_a_consistent_state_validates_after_many_tool_operations(self):
        import random
        rng = random.Random(7)
        ctx = self.new_ctx()
        foods = ["egg", "banana", "roti", "rice", "oats"]
        for step in range(60):
            if rng.random() < 0.3 and self.state.get_day_meals():
                self.call("remove_meal_from_state", food_name=rng.choice(foods), ctx=self.new_ctx())
            else:
                food = rng.choice(foods)
                unit = None if food in ("egg", "banana", "roti") else "g"
                self.call("add_meal_to_state", [item(food, rng.randint(1, 3) if unit is None else rng.randint(20, 300), unit)],
                          ctx=self.new_ctx())
        self.state.update_agent_result("meal_intake", specialist_status="success", run_id=ctx.run_id)
        check = self.call("validate_agent_results", ["meal_intake"], ctx=ctx)
        self.assertTrue(check["valid"], check)


if __name__ == "__main__":
    unittest.main()
