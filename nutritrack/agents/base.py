"""Shared Microsoft Agent Framework scaffolding for NutriTrack agents."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable, Optional

from agent_framework import Agent, AgentResponse, AgentSession, BaseChatClient, tool as af_tool

from nutritrack.agents.runtime import run_async
from nutritrack.state.run_log import sanitize_error_text
from nutritrack.tools import tools_for
from nutritrack.tools.base import AGENTS, LLMUnavailableError, ToolContext, summarize


@dataclass(frozen=True)
class AgentSpec:
    """Declarative specification for one NutriTrack Microsoft Agent Framework agent."""

    key: str
    name: str
    role: str
    instructions: str


def build_maf_tools(agent_key: str, ctx: ToolContext) -> list[Any]:
    """Bind NutriTrack deterministic Python tools to `ctx` and wrap as MAF `FunctionTool`s."""
    return [af_tool(bound_fn) for bound_fn in tools_for(agent_key, ctx)]


def build_maf_agent(
    spec: AgentSpec,
    ctx: ToolContext,
    client: BaseChatClient,
    *,
    extra_tools: Optional[list[Any]] = None,
) -> Agent:
    """Construct a Microsoft Agent Framework `Agent` with its own instructions and Python tools."""
    bound_tools = build_maf_tools(spec.key, ctx)
    if extra_tools:
        bound_tools.extend(extra_tools)
    return Agent(
        client=client,
        instructions=spec.instructions,
        id=spec.key,
        name=spec.name,
        description=spec.role,
        tools=bound_tools,
    )


def _safe_parse_result(raw_result: Any) -> Any:
    if isinstance(raw_result, str):
        stripped = raw_result.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                return json.loads(stripped)
            except Exception:
                return raw_result
    return raw_result


def extract_tool_trace(response: AgentResponse) -> list[dict[str, Any]]:
    """Extract ordered tool invocations and results from a MAF `AgentResponse`."""
    calls_by_id: dict[str, dict[str, Any]] = {}
    ordered_calls: list[dict[str, Any]] = []

    for msg in getattr(response, "messages", []) or []:
        for content in getattr(msg, "contents", []) or []:
            ctype = getattr(content, "type", None)
            if ctype == "function_call":
                call_id = getattr(content, "call_id", None) or f"call-{len(ordered_calls) + 1}"
                args = getattr(content, "arguments", None)
                if isinstance(args, str):
                    args = _safe_parse_result(args)
                entry = {
                    "call_id": call_id,
                    "tool": getattr(content, "name", None) or "unknown_tool",
                    "arguments": args,
                    "status": "called",
                    "code": None,
                    "message": None,
                }
                calls_by_id[call_id] = entry
                ordered_calls.append(entry)
            elif ctype == "function_result":
                call_id = getattr(content, "call_id", None)
                parsed = _safe_parse_result(getattr(content, "result", None))
                entry = calls_by_id.get(call_id) if call_id else None
                if entry is None:
                    entry = {
                        "call_id": call_id or f"result-{len(ordered_calls) + 1}",
                        "tool": getattr(content, "tool_name", None) or "unknown_tool",
                        "arguments": None,
                    }
                    ordered_calls.append(entry)
                if isinstance(parsed, dict):
                    entry["status"] = parsed.get("status", "success")
                    entry["code"] = parsed.get("code")
                    entry["message"] = parsed.get("message")
                else:
                    raw_exc = getattr(content, "exception", None)
                    clean_exc = sanitize_error_text(raw_exc) if raw_exc else ""
                    entry["status"] = "success" if not raw_exc else "failed"
                    entry["code"] = "exception" if raw_exc else "ok"
                    entry["message"] = summarize(parsed or clean_exc, 200)
                if getattr(content, "exception", None):
                    entry["status"] = "failed"
                    entry["error"] = sanitize_error_text(getattr(content, "exception"))
    return ordered_calls


def _agent_open_questions(ctx: ToolContext, agent_name: str) -> list[dict[str, Any]]:
    questions = ctx.state.get_open_questions(run_id=ctx.run_id)
    return [q for q in questions if q.get("agent") == agent_name]


def _record_failed_execution(
    ctx: ToolContext,
    agent_key: str,
    agent_name: str,
    task: str,
    *,
    code: str,
    tool: str,
    exc: Exception,
    summary_template: str,
) -> dict[str, Any]:
    """Record a sanitized agent execution failure across shared state, run log, and return payload."""
    clean_exc = sanitize_error_text(exc)
    err_msg = sanitize_error_text(f"{type(exc).__name__}: {clean_exc}")
    ctx.state.update_agent_result(
        agent_key,
        specialist_status="failed",
        run_id=ctx.run_id,
        agent=agent_name,
        code=code,
        message=summarize(err_msg, 300),
    )
    ctx.run_log.append(
        run_id=ctx.run_id,
        user_id=ctx.user_id,
        agent=agent_name,
        task=ctx.task or task,
        tool=tool,
        status="failed",
        input_summary=summarize(task, 160),
        error={"code": code, "message": summarize(err_msg, 240)},
        execution_mode=ctx.current_execution_mode,
    )
    return {
        "agent_key": agent_key,
        "agent": agent_name,
        "run_id": ctx.run_id,
        "status": "failed",
        "code": code,
        "summary": summary_template.format(agent_name=agent_name, exc=clean_exc, err_msg=err_msg),
        "tools_called": [],
        "open_questions": _agent_open_questions(ctx, agent_name),
        "error_code": code,
        "error_message": err_msg,
    }


def execute_agent(
    agent: Agent,
    agent_key: str,
    ctx: ToolContext,
    task: str,
    *,
    session: Optional[AgentSession] = None,
    raise_on_llm_error: bool = False,
) -> dict[str, Any]:
    """Execute a specialist or manager MAF agent synchronously with structured output and logging."""
    agent_name = AGENTS.get(agent_key, agent.name or agent_key)
    if not isinstance(task, str) or not task.strip():
        msg = "Task instruction must be a non-empty string."
        ctx.state.update_agent_result(
            agent_key,
            specialist_status="failed",
            run_id=ctx.run_id,
            agent=agent_name,
            code="invalid_input",
            message=msg,
        )
        return {
            "agent_key": agent_key,
            "agent": agent_name,
            "run_id": ctx.run_id,
            "status": "failed",
            "code": "invalid_input",
            "summary": msg,
            "tools_called": [],
            "open_questions": [],
            "error_code": "invalid_input",
            "error_message": msg,
        }

    try:
        response: AgentResponse = run_async(agent.run(task.strip(), session=session))
    except LLMUnavailableError as exc:
        if raise_on_llm_error:
            raise
        return _record_failed_execution(
            ctx,
            agent_key,
            agent_name,
            task,
            code="llm_unavailable",
            tool="llm_call",
            exc=exc,
            summary_template="{agent_name} is unavailable because the LLM provider could not be reached ({exc}).",
        )
    except (ConnectionError, TimeoutError) as exc:
        if raise_on_llm_error:
            raise LLMUnavailableError(sanitize_error_text(exc)) from exc
        return _record_failed_execution(
            ctx,
            agent_key,
            agent_name,
            task,
            code="llm_unavailable",
            tool="llm_call",
            exc=exc,
            summary_template="{agent_name} could not connect to the LLM service ({exc}).",
        )
    except Exception as exc:
        failure = _record_failed_execution(
            ctx,
            agent_key,
            agent_name,
            task,
            code="agent_error",
            tool="agent_execution",
            exc=exc,
            summary_template="{agent_name} encountered an error ({err_msg}).",
        )
        if raise_on_llm_error:
            raise
        return failure

    tool_trace = extract_tool_trace(response)
    summary_text = (getattr(response, "text", None) or "").strip()
    questions = _agent_open_questions(ctx, agent_name)
    tool_names = [t["tool"] for t in tool_trace]

    if not summary_text:
        summary_text = f"{agent_name} completed using tools: {', '.join(tool_names) or 'none'}."

    status = "needs_input" if questions else "success"
    ctx.state.update_agent_result(
        agent_key,
        run_id=ctx.run_id,
        agent=agent_name,
        tools_invoked=tool_names,
        tool_call_count=len(tool_names),
        last_response=summarize(summary_text, 600),
    )

    return {
        "agent_key": agent_key,
        "agent": agent_name,
        "run_id": ctx.run_id,
        "status": status,
        "code": "needs_input" if questions else "ok",
        "summary": summary_text,
        "tools_called": tool_trace,
        "tool_names": tool_names,
        "open_questions": questions,
        "error_code": None,
        "error_message": None,
        "response": response,
    }


def make_specialist_runner(
    spec: AgentSpec,
    ctx: ToolContext,
    client: BaseChatClient,
) -> Callable[[str], dict[str, Any]]:
    """Create a synchronous runner callable compatible with `ctx.specialists[key]`."""
    agent = build_maf_agent(spec, ctx, client)

    def _runner(task: str) -> dict[str, Any]:
        return execute_agent(agent, spec.key, ctx, task, raise_on_llm_error=True)

    return _runner
