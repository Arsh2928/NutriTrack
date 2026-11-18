"""Agent 5: Nutrition Manager Agent — Manager-Worker coordinator built on Microsoft Agent Framework."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

from agent_framework import Agent, AgentSession, BaseChatClient

from nutritrack.agents.base import AgentSpec, build_maf_agent, execute_agent
from nutritrack.agents.goal_budget_agent import make_goal_budget_runner
from nutritrack.agents.meal_intake_agent import make_meal_intake_runner
from nutritrack.agents.meal_recommendation_agent import make_meal_recommendation_runner
from nutritrack.agents.nutrition_analysis_agent import make_nutrition_analysis_runner
from nutritrack.agents.runtime import create_chat_client
from nutritrack.ai.config import normalize_ai_mode
from nutritrack.state import RunLog, StateManager
from nutritrack.tools import new_context
from nutritrack.tools.base import SPECIALISTS, ToolContext, summarize

if TYPE_CHECKING:
    from nutritrack.ai.provider_manager import AIProviderManager

AGENT_KEY = "nutrition_manager"
AGENT_NAME = "Nutrition Manager Agent"
AGENT_ROLE = (
    "Coordinates user nutrition requests through Manager-Worker delegation, validates specialist "
    "outputs against shared state, handles failures, and compiles final responses."
)

SYSTEM_INSTRUCTIONS = """[AGENT:nutrition_manager]
You are the Nutrition Manager Agent in NutriTrack, the central coordinator built on Microsoft Agent Framework.
You NEVER calculate nutrition values, parse foods, set targets, or recommend meals directly inside your own text.
Instead, you orchestrate specialist agents through Manager-Worker delegation and maintain shared state integrity.

Specialists you coordinate via `delegate_to_specialist`:
- `meal_intake`: log meals, remove/undo meals, resolve foods, register user-supplied custom foods.
- `nutrition_analysis`: daily totals, macro splits, 7-day/weekly trends, unusual days, target comparison.
- `goal_budget`: set calorie/protein goals, compute remaining budget, check whether a proposed meal fits.
- `meal_recommendation`: candidate filtering, nutrition estimation, ranking, fit validation for next meals.

Workflow:
1. Call `update_workflow_status(status="planning", detail=...)` and `read_shared_state(...)`.
2. Decide which specialist(s) the request genuinely requires. Do NOT invoke specialists unnecessary for the request.
   - For simple meal logging ("I ate 2 eggs", "Log 100g paneer"), call ONLY `meal_intake` (add_meal_to_state already returns daily_totals and remaining budget).
   - Call `nutrition_analysis` only when the user asks for totals, macros, trends, or weekly insights.
   - Call `goal_budget` only when changing targets or explicitly checking budget fit.
   - Call `meal_recommendation` only when asked what to eat or for meal suggestions.
3. For general greetings or non-food questions, respond politely explaining what NutriTrack can do without delegating food specialists.
4. Delegate sequentially via `delegate_to_specialist(specialist=..., task=...)` in causal order (`meal_intake` -> `nutrition_analysis` -> `goal_budget` -> `meal_recommendation` as applicable).
5. If any specialist fails (for example `llm_unavailable`), stop dependent delegations, call `append_run_log` with failure details, set `update_workflow_status(status="failed", ...)`, and report the failure honestly without inventing results.
6. Otherwise call `validate_agent_results(expected_specialists=...)`, re-read `read_shared_state(...)` as needed, log completion with `append_run_log`, call `update_workflow_status(status="completed", ...)`, and compile a clear grounded response including any open user questions.
"""

SPEC = AgentSpec(
    key=AGENT_KEY,
    name=AGENT_NAME,
    role=AGENT_ROLE,
    instructions=SYSTEM_INSTRUCTIONS,
)


def register_specialists(
    ctx: ToolContext,
    client: Optional[BaseChatClient] = None,
    overrides: Optional[dict[str, Callable[[str], Any]]] = None,
) -> dict[str, Callable[[str], Any]]:
    """Wire all four specialist MAF `Agent` runners into `ctx.specialists`."""
    resolved_client = client if client is not None else create_chat_client()[0]
    runners: dict[str, Callable[[str], Any]] = {
        "meal_intake": make_meal_intake_runner(ctx, client=resolved_client),
        "nutrition_analysis": make_nutrition_analysis_runner(ctx, client=resolved_client),
        "goal_budget": make_goal_budget_runner(ctx, client=resolved_client),
        "meal_recommendation": make_meal_recommendation_runner(ctx, client=resolved_client),
    }
    if overrides:
        for key, fn in overrides.items():
            if key in SPECIALISTS and fn is not None:
                runners[key] = fn
    ctx.specialists.update(runners)
    return ctx.specialists


def create_nutrition_manager_agent(
    ctx: ToolContext,
    client: Optional[BaseChatClient] = None,
) -> Agent:
    """Create the Microsoft Agent Framework `Agent` instance for the Nutrition Manager."""
    resolved_client = client if client is not None else create_chat_client()[0]
    return build_maf_agent(SPEC, ctx, resolved_client)


def _extract_delegated_specialists(manager_result: dict[str, Any]) -> list[str]:
    activated: list[str] = []
    for call in manager_result.get("tools_called") or []:
        if call.get("tool") == "delegate_to_specialist":
            args = call.get("arguments") or {}
            spec_key = args.get("specialist")
            if isinstance(spec_key, str) and spec_key not in activated:
                activated.append(spec_key)
    return activated


def _group_tools_by_agent(run_entries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in run_entries:
        agent_name = entry.get("agent") or "Unknown Agent"
        grouped.setdefault(agent_name, []).append(
            {
                "tool": entry.get("tool"),
                "status": entry.get("status"),
                "input_summary": entry.get("input_summary"),
                "output_summary": entry.get("output_summary"),
                "error": entry.get("error"),
                "timestamp": entry.get("timestamp"),
            }
        )
    return grouped


class NutriTrackOrchestrator:
    """End-to-end Manager-Worker multi-agent coordinator backed by `StateManager`, `RunLog`, and `AIProviderManager`."""

    def __init__(
        self,
        *,
        state_manager: Optional[StateManager] = None,
        run_log: Optional[RunLog] = None,
        ai_manager: Optional[AIProviderManager] = None,
        client: Optional[BaseChatClient] = None,
        mode: str = "",
        model: Optional[str] = None,
        require_llm: Optional[bool] = None,
        specialist_overrides: Optional[dict[str, Callable[[str], Any]]] = None,
    ) -> None:
        self.state_manager = state_manager or StateManager()
        self.run_log = run_log or RunLog()
        self._explicit_client = client
        self.mode = mode
        self.model = model
        self.require_llm = require_llm
        self.specialist_overrides = specialist_overrides or {}

        if ai_manager is not None:
            self.ai_manager = ai_manager
            if self.ai_manager.run_log is None:
                self.ai_manager.run_log = self.run_log
        else:
            from nutritrack.ai.provider_manager import AIProviderManager

            self.ai_manager = AIProviderManager(
                run_log=self.run_log,
                persisted_dir=(
                    getattr(self.state_manager, "base_dir", None)
                    or getattr(self.state_manager, "_base", None)
                ),
            )
            if mode and mode.strip() and (
                client is None or mode.strip().lower() in {"gemini", "groq", "offline"}
            ):
                self.ai_manager.set_mode(normalize_ai_mode(mode), persist=False)

    def get_mode_info(self) -> dict[str, Any]:
        """Describe current Cloud AI / deterministic execution mode without exposing secrets."""
        from nutritrack.ai.provider_status import DATA_ONLY_BANNER

        ai_status = self.ai_manager.get_status()
        if self._explicit_client is not None:
            _, info = create_chat_client(
                client=self._explicit_client,
                mode=self.mode,
                model=self.model,
                require_llm=self.require_llm,
            )
            if not info.get("is_llm"):
                ai_status = {
                    **ai_status,
                    "selected_mode": "offline" if info.get("mode") == "offline_demo" else ai_status.get("selected_mode", "offline"),
                    "active_provider": "data_only",
                    "effective_label": "Offline Mode Active",
                    "fallback_active": False,
                    "data_only_active": True,
                    "banner": DATA_ONLY_BANNER,
                }
            info["ai_status"] = ai_status
            return info

        active = ai_status["active_provider"]
        is_cloud = active in {"gemini", "groq"}
        return {
            "mode": "llm" if is_cloud else "offline_demo",
            "selected_mode": ai_status["selected_mode"],
            "provider": active if active != "data_only" else "local_deterministic",
            "active_provider": active,
            "effective_label": ai_status["effective_label"],
            "fallback_active": ai_status["fallback_active"],
            "last_switch_reason": ai_status["last_switch_reason"],
            "banner": ai_status["banner"],
            "model": self.ai_manager.active_model_name,
            "is_llm": is_cloud,
            "has_key": bool(
                self.ai_manager.config.gemini_configured or self.ai_manager.config.groq_configured
            ),
            "ai_status": ai_status,
        }

    def run_turn(
        self,
        message: str,
        *,
        user_id: str = "demo_user",
        date: Optional[str] = None,
        client: Optional[BaseChatClient] = None,
        specialist_overrides: Optional[dict[str, Callable[[str], Any]]] = None,
        session: Optional[AgentSession] = None,
    ) -> dict[str, Any]:
        """Run one complete Manager-Worker turn for `message` and return full state & execution trace."""
        if not isinstance(message, str) or not message.strip():
            mode_info = self.get_mode_info()
            return {
                "status": "failed",
                "error_code": "invalid_input",
                "error_message": "Please enter a non-empty message.",
                "run_id": None,
                "user_id": user_id,
                "date": date,
                "response": "Please enter a food entry, question, or nutrition goal.",
                "workflow_status": "idle",
                "activated_specialists": [],
                "agent_results": {},
                "open_questions": [],
                "tools_by_agent": {},
                "run_log_entries": [],
                "state": self.state_manager.get(user_id).snapshot(day=date),
                "mode_info": mode_info,
                "ai_status": mode_info.get("ai_status"),
            }

        explicit = client if client is not None else self._explicit_client
        if explicit is not None:
            resolved_client, _ = create_chat_client(
                client=explicit,
                mode=self.mode,
                model=self.model,
                require_llm=self.require_llm,
            )
        else:
            resolved_client = self.ai_manager.chat_client

        # Reset transient per-turn workflow status / prior turn agent_results before new turn
        state = self.state_manager.get(user_id)
        state.reset_run_data()

        clean_message = message.strip()
        effective_task = clean_message
        from nutritrack.core.quantity import extract_date_from_text

        msg_date_hint, msg_date_err, _ = extract_date_from_text(clean_message)
        if date and date not in clean_message and not msg_date_hint and not msg_date_err:
            effective_task = f"{clean_message} (date: {date})"

        ctx = new_context(
            self.state_manager,
            self.run_log,
            user_id=user_id,
            task=clean_message,
        )
        if explicit is not None:
            _, explicit_info = create_chat_client(
                client=explicit,
                mode=self.mode,
                model=self.model,
                require_llm=self.require_llm,
            )
            ctx.execution_mode = "llm" if explicit_info.get("is_llm") else "data_only"
        else:
            ctx.provider_fn = lambda: "llm" if self.ai_manager.active_provider in {"gemini", "groq"} else "data_only"
        merged_overrides = {**self.specialist_overrides, **(specialist_overrides or {})}
        register_specialists(ctx, client=resolved_client, overrides=merged_overrides)

        manager_agent = create_nutrition_manager_agent(ctx, client=resolved_client)
        manager_result = execute_agent(
            manager_agent,
            AGENT_KEY,
            ctx,
            effective_task,
            session=session,
        )

        # Ensure workflow state reflects failures or idle success cleanly
        wf = state.get_workflow()
        if manager_result["status"] == "failed" and wf.get("status") != "failed":
            state.set_workflow_status("failed", summarize(manager_result.get("error_message") or "Manager error", 160))
            wf = state.get_workflow()

        run_entries = self.run_log.read(run_id=ctx.run_id)
        agent_results = state.get_agent_results()
        open_questions = state.get_open_questions(run_id=ctx.run_id)
        activated_specialists = _extract_delegated_specialists(manager_result)

        # Determine overall turn status accurately
        failed_specialists = {
            k: v
            for k, v in agent_results.items()
            if k in SPECIALISTS and isinstance(v, dict) and v.get("specialist_status") == "failed"
        }
        status = manager_result["status"]
        error_code = manager_result.get("error_code")
        error_message = manager_result.get("error_message")

        if wf.get("status") == "failed" or failed_specialists:
            status = "failed"
            if not error_code and failed_specialists:
                first_fail = next(iter(failed_specialists.values()))
                error_code = first_fail.get("code") or "specialist_failed"
                error_message = first_fail.get("message") or manager_result.get("summary")
        elif open_questions:
            status = "needs_input"
        elif status == "success" and wf.get("status") == "idle":
            state.set_workflow_status("completed", "Completed turn")
            wf = state.get_workflow()

        snap_day = None
        if msg_date_hint:
            try:
                snap_day = state.resolve_date(msg_date_hint)
            except Exception:
                snap_day = date
        else:
            snap_day = date
        snapshot = state.snapshot(day=snap_day)
        fresh_mode_info = self.get_mode_info()
        ai_status = fresh_mode_info.get("ai_status") or self.ai_manager.get_status()
        is_llm = bool(fresh_mode_info.get("is_llm")) and ai_status.get("active_provider") != "data_only"
        return {
            "status": status,
            "error_code": error_code,
            "error_message": error_message,
            "run_id": ctx.run_id,
            "user_id": user_id,
            "date": snapshot["date"],
            "response": manager_result.get("summary", ""),
            "is_llm": is_llm,
            "data_only_active": not is_llm,
            "execution_mode": "llm" if is_llm else "data_only_deterministic",
            "workflow_status": wf.get("status", "idle"),
            "workflow": wf,
            "activated_specialists": activated_specialists,
            "manager_tools": manager_result.get("tool_names", []),
            "agent_results": agent_results,
            "open_questions": open_questions,
            "tools_by_agent": _group_tools_by_agent(run_entries),
            "run_log_entries": run_entries,
            "state": snapshot,
            "mode_info": fresh_mode_info,
            "ai_status": ai_status,
        }


def run_nutritrack_turn(
    message: str,
    *,
    user_id: str = "demo_user",
    date: Optional[str] = None,
    state_manager: Optional[StateManager] = None,
    run_log: Optional[RunLog] = None,
    ai_manager: Optional[AIProviderManager] = None,
    client: Optional[BaseChatClient] = None,
    mode: str = "",
) -> dict[str, Any]:
    """Convenience single-call helper running `NutriTrackOrchestrator.run_turn`."""
    orchestrator = NutriTrackOrchestrator(
        state_manager=state_manager,
        run_log=run_log,
        ai_manager=ai_manager,
        client=client,
        mode=mode,
    )
    return orchestrator.run_turn(message, user_id=user_id, date=date)
