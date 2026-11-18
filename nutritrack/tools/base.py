"""
base.py
-------
Plumbing shared by every agent tool.

NOTE: tool modules deliberately do NOT use `from __future__ import annotations`.
Agent frameworks read each tool's real type annotations to build its schema for
the model, so they must be real objects, not strings.

How a tool is built
    @agent_tool("meal_intake")
    def lookup_food_nutrition(ctx, food: str, quantity: Optional[float] = None, ...) -> dict:
        '''Docstring written for the LLM.'''
        ...

  * `ctx` (ToolContext) is passed explicitly -- there are no hidden globals or
    context variables. bind(tool, ctx) returns a copy WITHOUT the ctx parameter:
    that copy is what an agent is given, so the model never sees (or has to fill
    in) the user id, run id or state.
  * Every tool returns a dict envelope:
        {"status": "success" | "failed", "code": str, "message": str, ...}
    A failed result that the user can fix carries needs_user_input=True and a
    `question`. Exceptions never escape a tool: they become failed envelopes.
  * The decorator logs every call to logs/agent_runs.jsonl and records the
    outcome in the shared state (agent_results), so the Agent Activity Monitor
    shows what REALLY ran.
"""
import functools
import inspect
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional

from ..state import (RunLog, StateError, StateManager, StateUpdateError, new_run_id, summarize)

FUNCTION_INVOCATION_CONFIG: dict[str, Any] = {
    "allow_concurrent_invocation": False,
    "max_iterations": 15,
    "max_function_calls": 30,
    "max_duration_seconds": 90.0,
    "max_consecutive_errors_per_request": 3,
    "include_detailed_errors": False,
}

AGENTS = {
    "meal_intake": "Meal Intake Agent",
    "nutrition_analysis": "Nutrition Analysis Agent",
    "goal_budget": "Goal and Budget Agent",
    "meal_recommendation": "Meal Recommendation Agent",
    "nutrition_manager": "Nutrition Manager Agent",
}
SPECIALISTS = ("meal_intake", "nutrition_analysis", "goal_budget", "meal_recommendation")


class LLMUnavailableError(RuntimeError):
    """Raised by the orchestration layer when the model provider cannot be reached.
    delegate_to_specialist reports it as the failure code 'llm_unavailable'."""


@dataclass
class ToolContext:
    """Everything a tool needs, passed explicitly: who, which run, where the state and log are."""
    state_manager: StateManager
    run_log: RunLog
    user_id: str = "demo_user"
    run_id: str = field(default_factory=new_run_id)
    task: str = ""
    specialists: dict = field(default_factory=dict)       # key -> callable(task: str) -> str | dict
    execution_mode: Optional[str] = None
    provider_fn: Optional[Callable[[], str]] = None

    @property
    def current_execution_mode(self) -> Optional[str]:
        if self.provider_fn is not None:
            return self.provider_fn()
        return self.execution_mode

    @property
    def state(self):
        return self.state_manager.get(self.user_id)

    def with_task(self, task):
        return replace(self, task=task)


# ---------------------------------------------------------------------------
# Result envelope
# ---------------------------------------------------------------------------
def ok(message="", code="ok", **fields):
    return {"status": "success", "code": code, "message": message, **fields}


def fail(code, message, needs_user_input=False, **fields):
    return {"status": "failed", "code": code, "message": message,
            "needs_user_input": needs_user_input, **fields}


_BULKY = {"meals", "weekly_history", "daily", "calorie_series", "candidates", "estimates", "ranked", "items",
          "added", "removed", "specialist_response", "state", "agent_results", "open_questions", "failed_calls"}


def _output_summary(result):
    """Compact view of a result for the log: scalars and short values, no big lists."""
    brief = {}
    for key, value in result.items():
        if key in _BULKY and isinstance(value, (list, dict)):
            brief[key] = f"<{len(value)} {'entries' if isinstance(value, list) else 'fields'}>"
        else:
            brief[key] = value
    return brief


def _record(ctx, agent_key, tool_name, arguments, result, duration_ms, write_log, records_questions=True):
    status = result["status"]
    error = None if status == "success" else {"code": result.get("code"), "message": result.get("message")}
    if write_log:
        ctx.run_log.append(run_id=ctx.run_id, user_id=ctx.user_id, agent=AGENTS[agent_key], task=ctx.task,
                           tool=tool_name, status=status, input_summary=arguments,
                           output_summary=_output_summary(result), error=error, duration_ms=duration_ms,
                           execution_mode=ctx.current_execution_mode)
    try:                                                   # recording must never break a tool
        if records_questions and result.get("needs_user_input"):   # also true for a PARTIAL success (some foods unresolved)
            unresolved = result.get("unresolved") or [{}]
            ctx.state.add_open_question(ctx.run_id, AGENTS[agent_key], tool_name, result.get("code"),
                                        result.get("message"), result.get("question") or unresolved[0].get("question"))
        ctx.state.update_agent_result(
            agent_key, bump_tool_calls=True, agent=AGENTS[agent_key], run_id=ctx.run_id, last_tool=tool_name,
            status=status, code=result.get("code"), message=summarize(result.get("message"), 200), task=ctx.task)
    except Exception:
        pass


def agent_tool(agent_key, *, log=True, records_questions=True):
    """Mark a function as a tool of `agent_key`: validate arguments, convert exceptions
    into failed envelopes, log the call, record it in the shared state.

    records_questions=False is for a tool that only REPORTS open questions (validate_agent_results);
    without it the report would register the same question a second time."""
    if agent_key not in AGENTS:
        raise ValueError(f"unknown agent '{agent_key}'")

    def decorate(fn):
        signature = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(ctx, *args, **kwargs):
            started = time.perf_counter()
            arguments = {}
            try:
                bound = signature.bind(ctx, *args, **kwargs)
                arguments = {k: v for k, v in bound.arguments.items() if k != "ctx"}
            except TypeError as exc:
                result = fail("bad_arguments", f"{fn.__name__}: {exc}")
            else:
                try:
                    result = fn(ctx, *args, **kwargs)
                except StateError as exc:
                    result = fail("invalid_input", str(exc), needs_user_input=True)
                except StateUpdateError as exc:
                    result = fail("state_update_failed", str(exc))
                except Exception as exc:
                    result = fail("internal_error", f"{type(exc).__name__}: {exc}")
            if not (isinstance(result, dict) and result.get("status") in ("success", "failed")):
                result = fail("invalid_tool_result", f"{fn.__name__} returned something that is not a result envelope.")
            _record(ctx, agent_key, fn.__name__, arguments, result, (time.perf_counter() - started) * 1000, log,
                    records_questions)
            return result

        wrapper.agent_key = agent_key
        wrapper.tool_name = fn.__name__
        return wrapper
    return decorate


def bind(tool, ctx):
    """Return `tool` with `ctx` filled in and removed from the signature -- the form an agent receives."""
    signature = inspect.signature(tool)
    parameters = list(signature.parameters.values())[1:]

    @functools.wraps(tool)
    def bound(*args, **kwargs):
        return tool(ctx, *args, **kwargs)

    bound.__signature__ = signature.replace(parameters=parameters)
    bound.__annotations__ = {k: v for k, v in getattr(tool, "__annotations__", {}).items() if k != "ctx"}
    return bound
