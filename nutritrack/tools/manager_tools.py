"""
Nutrition Manager Agent tools.

    read_shared_state        look at the shared nutrition state
    delegate_to_specialist   hand a task to one specialist agent and record what happened
    validate_agent_results   check specialist results and the state before anything is shown to the user
    update_workflow_status   idle -> planning -> delegating -> validating -> compiling -> completed / failed
    append_run_log           write a manager-level entry (a plan, a decision) to the execution log

The manager never does the specialists' work. delegate_to_specialist runs a specialist
through `ctx.specialists[key]` -- a callable the orchestration layer registers for each worker
agent (Phase 5). A specialist that fails is reported as a failure; the manager is never handed
a made-up result.
"""
import math
import time
from typing import Optional

from ..state import StateError, WORKFLOW_STATUSES, summarize
from ..state.run_log import redact
from .base import AGENTS, SPECIALISTS, LLMUnavailableError, agent_tool, fail, ok

AGENT = "nutrition_manager"
STATE_SECTIONS = ("targets", "consumed", "remaining", "meals", "weekly_history", "recommendations",
                  "agent_results", "workflow_status", "preferences")
DEFAULT_SECTIONS = ("targets", "consumed", "remaining", "workflow_status")

# Failed tool calls that only mean "the user has to tell us something" -- not a system fault.
USER_INPUT_CODES = frozenset({
    "unknown_food", "low_confidence_match", "invalid_quantity", "implausible_quantity", "ambiguous_quantity",
    "missing_quantity", "unit_mismatch", "unsupported_unit", "no_food_name", "no_target", "no_food_found",
    "empty_input", "not_found", "invalid_input", "food_exists", "implausible_values", "no_items_resolved",
    "unresolved_items", "input_too_long"})

_ALIASES = {
    "intake": "meal_intake", "meal": "meal_intake", "logging": "meal_intake",
    "analysis": "nutrition_analysis", "analytics": "nutrition_analysis", "analyst": "nutrition_analysis",
    "goal": "goal_budget", "budget": "goal_budget", "goal_and_budget": "goal_budget", "goal_and_calorie_budget": "goal_budget",
    "recommendation": "meal_recommendation", "recommender": "meal_recommendation", "recommendations": "meal_recommendation",
}


def resolve_specialist(name):
    """'Meal Intake Agent' / 'meal-intake' / 'intake' -> 'meal_intake'; None if unknown."""
    if not isinstance(name, str):
        return None
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    if key.endswith("_agent"):
        key = key[:-6]
    key = _ALIASES.get(key, key)
    return key if key in SPECIALISTS else None


def _classify(exc):
    if isinstance(exc, (LLMUnavailableError, ConnectionError, TimeoutError)):
        return "llm_unavailable"
    return "specialist_failed"


@agent_tool(AGENT)
def read_shared_state(ctx, sections: Optional[list[str]] = None, date: Optional[str] = None) -> dict:
    """Read the shared nutrition state (read-only).

    Args:
        sections: any of targets, consumed, remaining, meals, weekly_history, recommendations, agent_results,
            workflow_status, preferences. Default: targets, consumed, remaining, workflow_status.
        date: "today" (default), "yesterday" or YYYY-MM-DD.
    """
    wanted = tuple(sections) if sections else DEFAULT_SECTIONS
    unknown = [s for s in wanted if s not in STATE_SECTIONS]
    if unknown:
        raise StateError(f"Unknown section(s): {', '.join(map(str, unknown))}. Choose from: {', '.join(STATE_SECTIONS)}.")
    snap = ctx.state.snapshot(date)
    return ok(f"State for {snap['date']}: {len(snap['meals'])} meal entr{'y' if len(snap['meals']) == 1 else 'ies'}, "
              f"workflow '{snap['workflow_status']}'.",
              date=snap["date"], state={s: snap[s] for s in wanted}, meal_count=len(snap["meals"]),
              targets_source=snap["targets_source"])


@agent_tool(AGENT)
def delegate_to_specialist(ctx, specialist: str, task: str) -> dict:
    """Give ONE task to ONE specialist agent and wait for its answer. Delegate only the specialists the
    request needs, one clear task at a time.

    Args:
        specialist: meal_intake, nutrition_analysis, goal_budget or meal_recommendation.
        task: a plain-language instruction containing everything the specialist needs, e.g.
            "Log: 3 boiled eggs and 2 bananas for breakfast".
    """
    key = resolve_specialist(specialist)
    if key is None:
        return fail("unknown_specialist", f"'{specialist}' is not a specialist. Choose from: {', '.join(SPECIALISTS)}.")
    if not isinstance(task, str) or not task.strip() or len(task) > 2000:
        return fail("invalid_input", "task must be a non-empty instruction of at most 2000 characters.")
    runner = ctx.specialists.get(key)
    if runner is None:
        ctx.state.update_agent_result(key, specialist_status="failed", run_id=ctx.run_id, agent=AGENTS[key],
                                      message="No runner is registered for this specialist.")
        return fail("specialist_unavailable", f"{AGENTS[key]} is not available right now.", specialist=key)
    ctx.state.set_workflow_status("delegating", key)
    ctx.state.update_agent_result(key, specialist_status="running", run_id=ctx.run_id, agent=AGENTS[key],
                                  delegated_task=summarize(task, 300))
    started = time.perf_counter()
    try:
        output = runner(task)
    except Exception as exc:
        code = _classify(exc)
        safe_exc = redact(f"{type(exc).__name__}: {exc}")
        ctx.state.update_agent_result(key, specialist_status="failed", run_id=ctx.run_id, code=code,
                                      message=summarize(safe_exc, 300),
                                      duration_ms=round((time.perf_counter() - started) * 1000, 1))
        return fail(code, f"{AGENTS[key]} could not complete the task ({safe_exc}).",
                    specialist=key)
    text = output.get("summary", "") if isinstance(output, dict) else str(output)
    if not isinstance(text, str) or not text.strip():
        ctx.state.update_agent_result(key, specialist_status="failed", run_id=ctx.run_id, code="empty_response",
                                      message="The specialist returned an empty answer.")
        return fail("empty_response", f"{AGENTS[key]} returned an empty answer.", specialist=key)
    ctx.state.update_agent_result(key, specialist_status="success", run_id=ctx.run_id, summary=summarize(text, 600),
                                  duration_ms=round((time.perf_counter() - started) * 1000, 1))
    return ok(f"{AGENTS[key]} finished.", specialist=key, specialist_response=text)


@agent_tool(AGENT, records_questions=False)
def validate_agent_results(ctx, expected_specialists: Optional[list[str]] = None) -> dict:
    """Check the specialists' results and the shared state before writing the final answer.
    Run this after delegating and before replying.

    It verifies that every expected specialist reported a result IN THIS RUN and did not fail, that the
    state's totals match its meals, that targets and remaining amounts agree, and that saved recommendations
    still fit. Tool calls that only need the user's input (unknown food, missing target...) are returned as
    open_questions; real failures make the validation fail.

    Args:
        expected_specialists: the specialists you delegated to this run, e.g. ["meal_intake", "goal_budget"].
    """
    st = ctx.state
    issues, warnings = [], []
    keys = []
    for name in expected_specialists or []:
        key = resolve_specialist(name)
        if key is None:
            issues.append(f"'{name}' is not a known specialist.")
        elif key not in keys:
            keys.append(key)

    results = st.get_agent_results()
    for key in keys:
        r = results.get(key)
        label = AGENTS[key]
        if r is None:
            issues.append(f"{label} reported no result.")
            continue
        if r.get("run_id") != ctx.run_id:
            issues.append(f"{label}'s recorded result belongs to an earlier run ({r.get('run_id')}), not this one.")
        if r.get("specialist_status") == "failed":
            issues.append(f"{label} failed: {r.get('message') or r.get('code')}.")
        elif r.get("specialist_status") == "running":
            issues.append(f"{label} never reported completion.")
        elif r.get("specialist_status") is None:
            warnings.append(f"{label} did not finish through delegate_to_specialist; only its tool calls are recorded.")

    # failed tool calls in THIS run, from the append-only log
    failed_calls = []
    open_questions = st.get_open_questions(ctx.run_id)         # recorded when each tool returned (covers partial successes)
    for entry in ctx.run_log.read(run_id=ctx.run_id):
        if entry.get("status") != "failed":
            continue
        error = entry.get("error") or {}
        item = {"agent": entry.get("agent"), "tool": entry.get("tool"), "code": error.get("code"),
                "message": error.get("message")}
        failed_calls.append(item)
        if error.get("code") in USER_INPUT_CODES:
            continue                                           # the user can fix it: reported as an open question
        if entry.get("tool") != "validate_agent_results":      # an earlier validation failure is not a new fault
            issues.append(f"{entry.get('agent')} / {entry.get('tool')} failed with '{error.get('code')}': {error.get('message')}")

    # state invariants (cheap, and they catch a corrupted state before the user sees it)
    snap = st.snapshot()
    meals = snap["meals"]
    recomputed = {k: round(math.fsum(m[k] for m in meals), 1) for k in ("calories", "protein", "carbs", "fat")}
    if recomputed != snap["consumed"]:
        issues.append(f"Consumed totals {snap['consumed']} do not match the meals {recomputed}.")
    if len({m["meal_id"] for m in meals}) != len(meals):
        issues.append("Duplicate meal ids found in today's meals.")
    for kind in ("calories", "protein"):
        target, rem = snap["targets"][kind], snap["remaining"][kind]
        if target is not None and rem != round(target - snap["consumed"][kind], 1):
            issues.append(f"Remaining {kind} ({rem}) does not equal target minus consumed.")
    if snap["recommendations"]:
        if snap["recommendations_stale"]:
            warnings.append("Saved recommendations were made for different remaining amounts and may be out of date.")
        else:
            left = snap["remaining"]["calories"]
            for rec in snap["recommendations"]:
                if left is not None and rec["calories"] > left:
                    issues.append(f"Recommendation '{rec['name']}' ({rec['calories']} kcal) exceeds the {left} kcal remaining.")

    if issues:
        return fail("validation_failed", "Validation found problems: " + " | ".join(issues[:5]), valid=False,
                    issues=issues, warnings=warnings, failed_calls=failed_calls, open_questions=open_questions)
    note = ""
    if open_questions:
        note = f" {len(open_questions)} item(s) still need the user's input."
    return ok("All checks passed." + note, code="valid_with_questions" if open_questions else "valid", valid=True,
              issues=[], warnings=warnings, failed_calls=failed_calls, open_questions=open_questions,
              needs_user_input=bool(open_questions), checked_specialists=keys)


@agent_tool(AGENT)
def update_workflow_status(ctx, status: str, detail: Optional[str] = None) -> dict:
    """Record where the workflow is. Typical order: planning -> delegating -> validating -> compiling ->
    completed (or failed). The UI shows this live.

    Args:
        status: idle, planning, delegating, validating, compiling, completed or failed.
        detail: short note, e.g. which specialist is running.
    """
    if status not in WORKFLOW_STATUSES:
        return fail("invalid_input", f"status must be one of: {', '.join(WORKFLOW_STATUSES)}.")
    change = ctx.state.set_workflow_status(status, detail)
    return ok(f"Workflow status: {change['previous']} -> {change['current']}.", previous=change["previous"],
              current=change["current"], detail=detail)


@agent_tool(AGENT, log=False)
def append_run_log(ctx, agent: str, task: str, tool: str, status: str, input_summary: str = "",
                   output_summary: str = "", error_code: Optional[str] = None,
                   error_message: Optional[str] = None) -> dict:
    """Append one entry to the execution log (logs/agent_runs.jsonl) for a manager-level step such as a plan or
    a decision. Specialist tool calls are logged automatically; do not repeat them. Secrets are masked.

    Args:
        agent: who did it, e.g. "Nutrition Manager Agent".
        task: what was being done.
        tool: the tool or step name, e.g. "plan".
        status: "success" or "failed".
        input_summary / output_summary: short text.
        error_code / error_message: when status is "failed".
    """
    if status not in ("success", "failed"):
        return fail("invalid_input", "status must be 'success' or 'failed'.")
    error = {"code": error_code or "error", "message": error_message or ""} if status == "failed" else None
    written = ctx.run_log.append(run_id=ctx.run_id, user_id=ctx.user_id, agent=agent, task=task, tool=tool,
                                 status=status, input_summary=input_summary, output_summary=output_summary,
                                 error=error, execution_mode=ctx.current_execution_mode)
    if not written:
        return fail("log_failed", "The entry could not be written to the run log.")
    return ok("Logged.", code="logged", run_id=ctx.run_id)
