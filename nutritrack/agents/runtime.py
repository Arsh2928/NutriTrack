"""LLM client factory, async-to-sync bridge, error translation, and deterministic test/demo MAF clients."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import contextvars
from dataclasses import dataclass, field
from datetime import date, timedelta
import json
import os
import re
import sys
from typing import Any, Awaitable, Callable, Optional, Sequence, TypeVar

from agent_framework import (
    BaseChatClient,
    ChatContext,
    ChatResponse,
    Content,
    FunctionInvocationLayer,
    Message,
    chat_middleware,
)
from agent_framework.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
from openai import AsyncOpenAI

from nutritrack.core.quantity import extract_date_from_text, is_non_food_request, parse_item_text, split_meal_text
from nutritrack.paths import PROJECT_ROOT
from nutritrack.state.run_log import sanitize_error_text
from nutritrack.tools.base import FUNCTION_INVOCATION_CONFIG, LLMUnavailableError, summarize

T = TypeVar("T")

_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
_AGENT_TAG_RE = re.compile(r"\[AGENT:([a-z_]+)\]")


def load_env(*, force: bool = False) -> None:
    """Load `.env` from project root when not running inside isolated pytest/unittest suites."""
    if not force and ("pytest" in sys.modules or "PYTEST_CURRENT_TEST" in os.environ):
        return
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)


def detect_llm_config() -> dict[str, Any]:
    """Inspect explicit manual AI selection (never infer provider from API key presence)."""
    from nutritrack.ai.config import (
        DEFAULT_GEMINI_BASE_URL,
        DEFAULT_GEMINI_MODEL,
        DEFAULT_GROQ_BASE_URL,
        DEFAULT_GROQ_MODEL,
        load_persisted_mode,
        normalize_provider_model,
    )

    load_env()
    explicit_provider = os.environ.get("NUTRITRACK_LLM_PROVIDER", "").strip().lower()
    ai_mode = os.environ.get("NUTRITRACK_AI_MODE", "").strip().lower()
    persisted_mode = load_persisted_mode()
    gemini_key = (
        os.environ.get("GEMINI_API_KEY", "").strip()
        or os.environ.get("GOOGLE_API_KEY", "").strip()
    )
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    custom_model = os.environ.get("NUTRITRACK_MODEL", "").strip()
    gemini_model = normalize_provider_model("gemini", os.environ.get("GEMINI_MODEL", ""))
    groq_model = normalize_provider_model("groq", os.environ.get("GROQ_MODEL", ""))

    has_any_key = bool(gemini_key or groq_key)

    if explicit_provider in {"offline", "offline_demo", "deterministic", "mock", "local"}:
        chosen = "offline"
    elif explicit_provider in {"gemini", "groq"}:
        chosen = explicit_provider
    elif persisted_mode in {"gemini", "groq", "offline"}:
        chosen = persisted_mode
    elif ai_mode in {"gemini", "groq", "offline"}:
        chosen = ai_mode
    else:
        chosen = "offline"

    if chosen == "offline":
        return {
            "mode": "offline_demo",
            "provider": "local_deterministic",
            "model": "deterministic-tool-planner-v1",
            "base_url": None,
            "is_llm": False,
            "has_key": has_any_key,
        }

    if chosen == "gemini":
        return {
            "mode": "llm" if gemini_key else "offline_demo",
            "provider": "gemini" if gemini_key else "local_deterministic",
            "model": (custom_model or gemini_model) if gemini_key else "deterministic-tool-planner-v1",
            "base_url": DEFAULT_GEMINI_BASE_URL if gemini_key else None,
            "is_llm": bool(gemini_key),
            "has_key": bool(gemini_key),
        }

    return {
        "mode": "llm" if groq_key else "offline_demo",
        "provider": "groq" if groq_key else "local_deterministic",
        "model": (custom_model or groq_model) if groq_key else "deterministic-tool-planner-v1",
        "base_url": DEFAULT_GROQ_BASE_URL if groq_key else None,
        "is_llm": bool(groq_key),
        "has_key": bool(groq_key),
    }


def run_async(coro: Awaitable[T]) -> T:
    """Run an async Microsoft Agent Framework coroutine safely from sync code (Flask, CLI, Jupyter, or tools)."""

    async def _await() -> T:
        return await coro

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_await())

    ctx = contextvars.copy_context()
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(ctx.run, lambda: asyncio.run(_await())).result()


@chat_middleware
async def _translate_llm_errors_middleware(context: ChatContext, call_next: Callable[[], Awaitable[None]]) -> None:
    """Convert provider/network failures into `LLMUnavailableError` so failures never hide or invent results."""
    try:
        await call_next()
    except LLMUnavailableError:
        raise
    except Exception as exc:
        raise LLMUnavailableError(sanitize_error_text(f"LLM request failed ({type(exc).__name__}: {exc})")) from exc


class LoopSafeOpenAIChatClient(FunctionInvocationLayer, BaseChatClient):
    """MAF chat client creating a fresh `AsyncOpenAI` per call on the active event loop.

    Never retains `httpx.AsyncClient` sockets across `asyncio.run(...)` boundaries and bounds
    tool-call iteration loops cleanly via `FUNCTION_INVOCATION_CONFIG`.
    """

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        explicit_mode: bool = False,
    ) -> None:
        FunctionInvocationLayer.__init__(
            self,
            function_invocation_configuration=dict(FUNCTION_INVOCATION_CONFIG),
        )
        BaseChatClient.__init__(self)
        self.provider = provider
        self.model = model
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = float(timeout)
        self._explicit_mode = explicit_mode

    async def _inner_get_response(
        self,
        *,
        messages: Sequence[Message],
        stream: bool,
        options: dict[str, Any],
        **kwargs: Any,
    ) -> ChatResponse:
        from nutritrack.ai.config import ai_mode_scope, assert_network_allowed, is_turn_cancelled, remaining_turn_timeout

        if is_turn_cancelled():
            raise LLMUnavailableError("Request cancelled or exceeded server turn deadline.")

        if self._explicit_mode and self.provider in {"gemini", "groq"}:
            scope_cm = ai_mode_scope(self.provider)
        else:
            assert_network_allowed("ai", provider=self.provider)
            scope_cm = None

        async def _run_call() -> ChatResponse:
            if is_turn_cancelled():
                raise LLMUnavailableError("Request cancelled or exceeded server turn deadline.")
            assert_network_allowed("ai", provider=self.provider)
            effective_timeout = remaining_turn_timeout(self._timeout)
            client_kwargs: dict[str, Any] = {
                "api_key": self._api_key,
                "timeout": effective_timeout,
                "max_retries": 0,
            }
            if self._base_url:
                client_kwargs["base_url"] = self._base_url
            async_sdk_client = AsyncOpenAI(**client_kwargs)
            try:
                if self.provider in {"gemini", "groq"}:
                    from nutritrack.ai.provider_factory import _wrap_sdk_chat_completions

                    _wrap_sdk_chat_completions(async_sdk_client, self.provider, self.model)
                inner = OpenAIChatCompletionClient(
                    model=self.model,
                    async_client=async_sdk_client,
                    function_invocation_configuration={"enabled": False},
                )
                try:
                    return await inner._inner_get_response(
                        messages=messages,
                        stream=stream,
                        options=options,
                        **kwargs,
                    )
                except LLMUnavailableError:
                    raise
                except Exception as exc:
                    raise LLMUnavailableError(
                        sanitize_error_text(f"LLM request failed ({type(exc).__name__}: {exc})")
                    ) from exc
            finally:
                try:
                    await async_sdk_client.close()
                except Exception:
                    pass

        if scope_cm is not None:
            with scope_cm:
                return await _run_call()
        return await _run_call()


def create_chat_client(
    *,
    client: Optional[BaseChatClient] = None,
    mode: str = "",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    require_llm: Optional[bool] = None,
) -> tuple[BaseChatClient, dict[str, Any]]:
    """Build or reuse a Microsoft Agent Framework chat client plus descriptive mode metadata."""
    if client is not None:
        info = {
            "mode": "mock" if isinstance(client, (ScriptedChatClient, FailingChatClient)) else (
                "offline_demo" if isinstance(client, DeterministicNutriTrackChatClient) else "llm"
            ),
            "provider": type(client).__name__,
            "model": getattr(client, "model", None),
            "is_llm": not isinstance(client, DeterministicNutriTrackChatClient),
        }
        return client, info

    if require_llm is None:
        require_llm = os.environ.get("NUTRITRACK_REQUIRE_LLM", "").strip().lower() in {"1", "true", "yes"}

    cfg = detect_llm_config()
    chosen_mode = (mode or "").strip().lower()
    if chosen_mode in {"offline", "offline_demo", "deterministic"} and not require_llm:
        return DeterministicNutriTrackChatClient(), {
            "mode": "offline_demo",
            "provider": "local_deterministic",
            "model": "deterministic-tool-planner-v1",
            "is_llm": False,
            "has_key": cfg["has_key"],
        }

    resolved_key = api_key
    resolved_provider = cfg["provider"]
    resolved_model = model or cfg["model"]
    resolved_base_url = base_url or cfg["base_url"]

    if not resolved_key:
        if chosen_mode == "gemini":
            from nutritrack.ai.config import normalize_provider_model

            resolved_key = (
                os.environ.get("GEMINI_API_KEY", "").strip()
                or os.environ.get("GOOGLE_API_KEY", "").strip()
            )
            resolved_provider = "gemini"
            resolved_model = model or normalize_provider_model("gemini", os.environ.get("GEMINI_MODEL", ""))
            resolved_base_url = base_url or "https://generativelanguage.googleapis.com/v1beta/openai/"
            if not resolved_key:
                raise LLMUnavailableError("GEMINI_API_KEY is not configured for Gemini mode.")
        elif chosen_mode == "groq":
            from nutritrack.ai.config import normalize_provider_model

            resolved_key = os.environ.get("GROQ_API_KEY", "").strip()
            resolved_provider = "groq"
            resolved_model = model or normalize_provider_model("groq", os.environ.get("GROQ_MODEL", ""))
            resolved_base_url = base_url or "https://api.groq.com/openai/v1"
            if not resolved_key:
                raise LLMUnavailableError("GROQ_API_KEY is not configured for Groq mode.")
        elif cfg["is_llm"]:
            if cfg["provider"] == "gemini":
                resolved_key = (
                    os.environ.get("GEMINI_API_KEY", "").strip()
                    or os.environ.get("GOOGLE_API_KEY", "").strip()
                )
            elif cfg["provider"] == "groq":
                resolved_key = os.environ.get("GROQ_API_KEY", "").strip()

    if resolved_key and resolved_provider in {"gemini", "groq"}:
        live_client = LoopSafeOpenAIChatClient(
            provider=resolved_provider,
            model=resolved_model,
            api_key=resolved_key,
            base_url=resolved_base_url,
            explicit_mode=(chosen_mode in {"gemini", "groq"}),
        )
        return live_client, {
            "mode": "llm",
            "provider": resolved_provider,
            "model": resolved_model,
            "base_url": resolved_base_url,
            "is_llm": True,
            "has_key": True,
        }

    if require_llm:
        raise LLMUnavailableError(
            "No live LLM API key is configured. Set GEMINI_API_KEY or GROQ_API_KEY to use a live cloud LLM."
        )

    return DeterministicNutriTrackChatClient(), cfg


# ---------------------------------------------------------------------------
# Test and Mock Microsoft Agent Framework Clients
# ---------------------------------------------------------------------------


def tool_call_response(calls: Sequence[tuple[str, dict[str, Any]]], *, start_index: int = 1) -> ChatResponse:
    """Create a MAF `ChatResponse` containing one or more `function_call` contents."""
    contents = [
        Content.from_function_call(
            call_id=f"call_{start_index + idx}",
            name=tool_name,
            arguments=args,
        )
        for idx, (tool_name, args) in enumerate(calls)
    ]
    return ChatResponse(messages=[Message("assistant", contents)])


def text_response(text: str) -> ChatResponse:
    """Create a MAF `ChatResponse` with an assistant text message."""
    return ChatResponse(messages=[Message("assistant", [text])])


@dataclass
class FailingChatClient(FunctionInvocationLayer, BaseChatClient):
    """MAF chat client that simulates an unreachable or failing LLM endpoint."""

    error: Optional[Exception] = None
    fail_for_agents: Optional[set[str]] = None
    fallback_client: Optional[BaseChatClient] = None

    def __post_init__(self) -> None:
        FunctionInvocationLayer.__init__(
            self,
            function_invocation_configuration=dict(FUNCTION_INVOCATION_CONFIG),
        )
        BaseChatClient.__init__(self)

    async def _inner_get_response(
        self,
        *,
        messages: Sequence[Message],
        stream: bool,
        options: dict[str, Any],
        **kwargs: Any,
    ) -> ChatResponse:
        agent_key = _extract_agent_key(options)
        if self.fail_for_agents is None or agent_key in self.fail_for_agents:
            raise self.error or LLMUnavailableError("Simulated LLM endpoint unavailable.")
        if self.fallback_client is not None:
            return await self.fallback_client._inner_get_response(
                messages=messages,
                stream=stream,
                options=options,
                **kwargs,
            )
        return text_response("OK")


@dataclass
class ScriptedChatClient(FunctionInvocationLayer, BaseChatClient):
    """Deterministic scripted MAF chat client for reproducible unit and orchestration tests."""

    scripts: dict[str, list[ChatResponse | Callable[[Sequence[Message], dict[str, Any]], ChatResponse]]] = field(
        default_factory=dict
    )
    default_handler: Optional[Callable[[Sequence[Message], dict[str, Any]], ChatResponse]] = None
    _indices: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        FunctionInvocationLayer.__init__(
            self,
            function_invocation_configuration=dict(FUNCTION_INVOCATION_CONFIG),
        )
        BaseChatClient.__init__(self)

    async def _inner_get_response(
        self,
        *,
        messages: Sequence[Message],
        stream: bool,
        options: dict[str, Any],
        **kwargs: Any,
    ) -> ChatResponse:
        agent_key = _extract_agent_key(options) or "__default__"
        turns = self.scripts.get(agent_key)
        if turns is None:
            turns = self.scripts.get("__default__", [])
        idx = self._indices.get(agent_key, 0)
        if idx < len(turns):
            self._indices[agent_key] = idx + 1
            step = turns[idx]
            return step(messages, options) if callable(step) else step
        if self.default_handler is not None:
            return self.default_handler(messages, options)
        return text_response("Completed scripted turn.")


# ---------------------------------------------------------------------------
# Offline Deterministic MAF Chat Client (drives genuine MAF tool calls locally)
# ---------------------------------------------------------------------------


def _extract_agent_key(options: Optional[dict[str, Any]]) -> Optional[str]:
    instructions = (options or {}).get("instructions") or ""
    match = _AGENT_TAG_RE.search(instructions)
    return match.group(1) if match else None


def _latest_user_text(messages: Sequence[Message]) -> str:
    for msg in reversed(messages):
        if str(getattr(msg, "role", "")) == "user":
            parts = [getattr(c, "text", None) for c in getattr(msg, "contents", []) if getattr(c, "type", None) == "text"]
            text = " ".join(p for p in parts if p)
            if text.strip():
                return text.strip()
    return ""


def _parse_json_maybe(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                return json.loads(stripped)
            except Exception:
                return value
    return value


def _tool_history(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Collect executed (tool_name, arguments, parsed_result) tuples from MAF chat history."""
    calls: dict[str, dict[str, Any]] = {}
    ordered: list[dict[str, Any]] = []
    for msg in messages:
        for content in getattr(msg, "contents", []) or []:
            ctype = getattr(content, "type", None)
            if ctype == "function_call":
                cid = getattr(content, "call_id", None) or f"c-{len(ordered) + 1}"
                args = _parse_json_maybe(getattr(content, "arguments", None))
                entry = {
                    "call_id": cid,
                    "tool": getattr(content, "name", None),
                    "arguments": args if isinstance(args, dict) else {},
                    "result": None,
                }
                calls[cid] = entry
                ordered.append(entry)
            elif ctype == "function_result":
                cid = getattr(content, "call_id", None)
                entry = calls.get(cid) if cid else None
                parsed = _parse_json_maybe(getattr(content, "result", None))
                if entry is not None:
                    entry["result"] = parsed if isinstance(parsed, dict) else {"raw": parsed}
    return ordered


def _extract_date_hint(text: str) -> Optional[str]:
    date_hint, _, _ = extract_date_from_text(text or "")
    return date_hint


def _extract_custom_food_spec(text: str) -> Optional[dict[str, Any]]:
    """Detect explicit nutrition specs like 'Add custom food quinoa bowl: 380 kcal, 14g protein, 52g carbs, 12g fat'."""
    lower = text.lower()
    cal_m = re.search(r"(\d+(?:\.\d+)?)\s*(?:kcal|calories|cal)\b", lower)
    prot_m = re.search(r"(\d+(?:\.\d+)?)\s*(?:g|grams?)?\s*protein\b|protein\s*[:=]?\s*(\d+(?:\.\d+)?)\s*g?", lower)
    carb_m = re.search(r"(\d+(?:\.\d+)?)\s*(?:g|grams?)?\s*(?:carbs|carbohydrates)\b|(?:carbs|carbohydrates)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*g?", lower)
    fat_m = re.search(r"(\d+(?:\.\d+)?)\s*(?:g|grams?)?\s*fat\b|fat\s*[:=]?\s*(\d+(?:\.\d+)?)\s*g?", lower)
    if not (cal_m and prot_m and any(kw in lower for kw in ("custom food", "register food", "define food", "has ", "nutrition:"))):
        return None
    name_m = re.search(
        r"(?:custom food|register food|define food|my homemade|my|add)\s+([a-z][a-z0-9 _-]{1,50}?)(?:\s*(?:has|:|-|is)\s*\d|\s+with\s+\d)",
        lower,
    )
    name = (name_m.group(1).strip() if name_m else "").strip(" :-")
    if not name:
        return None
    protein = float(prot_m.group(1) or prot_m.group(2) or 0.0)
    carbs = float((carb_m.group(1) or carb_m.group(2)) if carb_m else 0.0)
    fat = float((fat_m.group(1) or fat_m.group(2)) if fat_m else 0.0)
    return {
        "name": name,
        "calories": float(cal_m.group(1)),
        "protein": protein,
        "carbs": carbs,
        "fat": fat,
    }


def _extract_goal_numbers(text: str) -> tuple[Optional[float], Optional[float]]:
    lower = text.lower()
    calories: Optional[float] = None
    protein: Optional[float] = None
    cal_m = re.search(r"(\d+(?:\.\d+)?)\s*(?:kcal|calories|cal)\b", lower)
    if not cal_m and any(k in lower for k in ("calorie goal", "calorie target", "daily calorie", "budget to")):
        cal_m = re.search(r"(?:to|target|goal|budget)\s*(?:of|is|=|:)?\s*(\d{3,4}(?:\.\d+)?)", lower)
    if cal_m:
        calories = float(cal_m.group(1))

    prot_m = re.search(r"(\d+(?:\.\d+)?)\s*(?:g|grams?)?\s*(?:of\s+)?protein\b", lower)
    if not prot_m:
        prot_m = re.search(r"protein\s+(?:target|goal)\s*(?:to|of|is|=|:)?\s*(\d+(?:\.\d+)?)\s*g?\b", lower)
    if prot_m:
        protein = float(prot_m.group(1))
    return calories, protein


def _extract_proposed_items(text: str) -> list[dict[str, Any]]:
    cleaned = re.sub(
        r"(?i)\b(does|would|will|can i eat|if i eat|what if i eat|fit my budget|fit in my budget|within my budget|against my budget|fit|today|my remaining)\b",
        " ",
        text,
    )
    _, _, pieces = split_meal_text(cleaned)
    items: list[dict[str, Any]] = []
    for piece in pieces:
        parsed = parse_item_text(piece)
        if parsed and getattr(parsed, "food_text", ""):
            items.append(
                {
                    "food": parsed.food_text,
                    "quantity": parsed.value if parsed.value is not None else 1,
                    "unit": parsed.unit,
                    "household_unit": parsed.household_unit,
                }
            )
    return items


def _extract_inline_preferences(text: str) -> dict[str, Any]:
    lower = text.lower()
    diet = None
    if "vegetarian" in lower or "veg meal" in lower or "veg only" in lower:
        diet = "vegetarian"
    elif "eggetarian" in lower:
        diet = "eggetarian"
    avoid: list[str] = []
    avoid_m = re.search(r"\b(?:avoid|without|no)\s+([a-z ]{2,30}?)(?:[,.;]|$|\band\b|\bwith\b)", lower)
    if avoid_m:
        phrase = avoid_m.group(1).strip()
        if phrase not in {"added sugar", "missing", "doubt", "problem"}:
            avoid.append(phrase)
    return {"diet": diet, "avoid": avoid or None}


def plan_specialists_for_request(user_request: str) -> list[str]:
    """Deterministically plan which specialist agent(s) a user request genuinely needs."""
    text = (user_request or "").strip()
    lower = text.lower()
    if not lower:
        return []

    is_recommendation = any(
        kw in lower
        for kw in (
            "recommend",
            "suggest",
            "what should i eat",
            "next meal",
            "meal idea",
            "meal plan",
            "hit my remaining",
            "what can i eat",
        )
    )
    is_budget_check = any(
        kw in lower
        for kw in (
            "fit my budget",
            "fit in my budget",
            "against my budget",
            "within my budget",
            "can i afford",
            "if i eat",
            "would ",
        )
    ) and ("fit" in lower or "budget" in lower or "calorie" in lower)

    is_goal_update = any(
        kw in lower
        for kw in (
            "set calorie",
            "set my calorie",
            "set daily calorie",
            "set protein",
            "set my protein",
            "calorie goal",
            "calorie target",
            "protein goal",
            "protein target",
            "change goal",
            "update goal",
            "update my target",
            "target to",
            "goal to",
        )
    )

    is_removal = any(kw in lower for kw in ("remove ", "delete ", "undo "))
    is_logging = is_removal or (
        not is_budget_check
        and any(
            kw in lower
            for kw in (
                "i ate",
                "i had",
                "log ",
                "add meal",
                "add my ",
                "for breakfast",
                "for lunch",
                "for dinner",
                "for snack",
                "for brunch",
                "register food",
                "custom food",
                "define food",
            )
        )
    )
    if not is_logging and not is_recommendation and not is_goal_update and not is_budget_check:
        # Check if message looks like plain food logging e.g. "3 boiled eggs and 2 bananas"
        analysis_only_kws = (
            "how many calories",
            "how much protein",
            "consumed",
            "did i eat",
            "today's total",
            "daily total",
            "macro",
            "carbs",
            "summary",
            "weekly",
            "insight",
            "trend",
            "7-day",
            "7 day",
            "average",
            "remaining",
            "left today",
            "budget",
            "target",
            "compare",
        )
        if not any(kw in lower for kw in analysis_only_kws):
            _, _, raw_pieces = split_meal_text(text)
            if raw_pieces:
                parsed = [parse_item_text(p) for p in raw_pieces]
                if any(getattr(p, "explicit_value", False) for p in parsed):
                    is_logging = True

    is_analysis = any(
        kw in lower
        for kw in (
            "how many calories",
            "how much protein",
            "consumed",
            "did i eat",
            "daily total",
            "today's total",
            "nutrition summary",
            "summary",
            "macro",
            "carbs",
            "fat",
            "weekly",
            "trend",
            "insight",
            "7-day",
            "7 day",
            "average",
            "progress",
            "compare",
            "unusual",
            "meal history",
            "history",
        )
    )
    is_goal_query = is_goal_update or is_budget_check or any(
        kw in lower for kw in ("remaining", "allowance", "calories left", "protein left", "budget")
    )

    plan: list[str] = []
    if is_logging:
        plan.append("meal_intake")
        if "how much" in lower or "how many" in lower or "protein" in lower or "calories" in lower or "total" in lower or "summary" in lower or not is_removal:
            plan.append("nutrition_analysis")
        if not is_removal or "remaining" in lower or "budget" in lower:
            plan.append("goal_budget")
    if is_analysis and "nutrition_analysis" not in plan:
        plan.append("nutrition_analysis")
    if is_goal_query and "goal_budget" not in plan:
        plan.append("goal_budget")
    if is_recommendation:
        if "goal_budget" not in plan:
            plan.append("goal_budget")
        plan.append("meal_recommendation")

    # Deduplicate preserving canonical execution order
    ordered = []
    for k in ("meal_intake", "nutrition_analysis", "goal_budget", "meal_recommendation"):
        if k in plan and k not in ordered:
            ordered.append(k)
    if not ordered and is_non_food_request(text):
        return []
    return ordered


class DeterministicNutriTrackChatClient(FunctionInvocationLayer, BaseChatClient):
    """Genuine MAF `BaseChatClient` + `FunctionInvocationLayer` orchestrating deterministic tool calls.

    Enables out-of-the-box offline execution of all 5 MAF agents, full tool invocation loops,
    real `NutritionState` updates, and honest error/missing-data reporting with zero fabricated values.
    """

    def __init__(self) -> None:
        FunctionInvocationLayer.__init__(
            self,
            function_invocation_configuration=dict(FUNCTION_INVOCATION_CONFIG),
        )
        BaseChatClient.__init__(self)

    async def _inner_get_response(
        self,
        *,
        messages: Sequence[Message],
        stream: bool,
        options: dict[str, Any],
        **kwargs: Any,
    ) -> ChatResponse:
        agent_key = _extract_agent_key(options)
        task = _latest_user_text(messages)
        history = _tool_history(messages)
        next_idx = len(history) + 1

        if agent_key == "meal_intake":
            return self._step_meal_intake(task, history, next_idx)
        if agent_key == "nutrition_analysis":
            return self._step_nutrition_analysis(task, history, next_idx)
        if agent_key == "goal_budget":
            return self._step_goal_budget(task, history, next_idx)
        if agent_key == "meal_recommendation":
            return self._step_meal_recommendation(task, history, next_idx)
        if agent_key == "nutrition_manager":
            return self._step_nutrition_manager(task, history, next_idx)

        return text_response(f"Unsupported agent key '{agent_key}'.")

    # ---- Specialist 1: Meal Intake Agent ----
    def _step_meal_intake(self, task: str, history: list[dict[str, Any]], idx: int) -> ChatResponse:
        date_hint = _extract_date_hint(task)
        lower = task.lower()
        called_names = [h["tool"] for h in history]

        # Custom food explicit nutrition definition
        custom = _extract_custom_food_spec(task)
        if custom is not None and "register_custom_food" not in called_names:
            return tool_call_response([("register_custom_food", custom)], start_index=idx)
        if custom is not None and "register_custom_food" in called_names:
            reg_res = history[-1]["result"] or {}
            if "lookup_food_nutrition" not in called_names:
                return tool_call_response(
                    [("lookup_food_nutrition", {"food": custom["name"], "quantity": 1})],
                    start_index=idx,
                )
            return text_response(reg_res.get("message") or f"Registered custom food '{custom['name']}'.")

        # Step 1: Parse entry
        if "parse_meal_entry" not in called_names:
            return tool_call_response([("parse_meal_entry", {"text": task})], start_index=idx)

        parse_res = next(h["result"] for h in history if h["tool"] == "parse_meal_entry")
        items = parse_res.get("items") or []
        effective_date = parse_res.get("date_hint") or date_hint

        # Removal branch
        if any(kw in lower for kw in ("remove ", "delete ", "undo ")):
            if "remove_meal_from_state" not in called_names:
                meal_id_m = re.search(
                    r"\b(m-[a-z0-9-]+|meal-\d{8}-[a-z0-9-]+|b-[a-z0-9-]+|batch-[a-z0-9-]+)\b",
                    lower,
                )
                kwargs_rm: dict[str, Any] = {}
                if effective_date:
                    kwargs_rm["date"] = effective_date
                if meal_id_m:
                    ident = meal_id_m.group(1)
                    if ident.startswith("b-") or ident.startswith("batch-"):
                        kwargs_rm["batch_id"] = ident
                    else:
                        kwargs_rm["meal_id"] = ident
                elif re.search(r"\b(?:my\s+|the\s+)?last(?:\s+(?:logged\s+)?(?:meal|food|item|entry))?\b", lower):
                    kwargs_rm["remove_last"] = True
                elif items:
                    cleaned_item = re.sub(
                        r"^(?:please\s+)?(?:remove|delete|undo)\s+(?:my\s+|the\s+|last\s+)?",
                        "",
                        items[0]["food"],
                        flags=re.IGNORECASE,
                    ).strip()
                    kwargs_rm["food_name"] = cleaned_item or items[0]["food"]
                else:
                    cleaned_name = re.sub(r"(?i)\b(please|remove|delete|undo|last|my|meal|from|today)\b", "", task).strip()
                    if cleaned_name:
                        kwargs_rm["food_name"] = cleaned_name
                    else:
                        kwargs_rm["remove_last"] = True
                return tool_call_response([("remove_meal_from_state", kwargs_rm)], start_index=idx)

            rm_res = next(h["result"] for h in history if h["tool"] == "remove_meal_from_state")
            return text_response(rm_res.get("message", "Meal removal checked."))

        if not items:
            return text_response(
                parse_res.get("message") or "No food items were found to log. Please name the food and quantity."
            )

        # Step 2: Lookup each parsed item sequentially using lookup_food_nutrition
        lookup_calls = [h for h in history if h["tool"] == "lookup_food_nutrition"]
        if len(lookup_calls) < len(items):
            item = items[len(lookup_calls)]
            lk_args: dict[str, Any] = {"food": item["food"]}
            if item.get("quantity") is not None:
                lk_args["quantity"] = item["quantity"]
            if item.get("unit"):
                lk_args["unit"] = item["unit"]
            if item.get("household_unit"):
                lk_args["household_unit"] = item["household_unit"]
            return tool_call_response([("lookup_food_nutrition", lk_args)], start_index=idx)

        # Step 3: Add parsed items to shared state via add_meal_to_state
        if "add_meal_to_state" not in called_names:
            add_args: dict[str, Any] = {
                "items": [
                    {
                        "food": it["food"],
                        "quantity": it.get("quantity"),
                        "unit": it.get("unit"),
                        "household_unit": it.get("household_unit"),
                    }
                    for it in items
                ],
            }
            if parse_res.get("meal_type"):
                add_args["meal_type"] = parse_res["meal_type"]
            if effective_date:
                add_args["date"] = effective_date
            return tool_call_response([("add_meal_to_state", add_args)], start_index=idx)

        add_res = next(h["result"] for h in history if h["tool"] == "add_meal_to_state")
        parts = [add_res.get("message", "Processed meal entry.")]
        unresolved = add_res.get("unresolved") or []
        for u in unresolved:
            q = u.get("question")
            if q and q not in parts[0]:
                parts.append(q)
        return text_response(" ".join(parts))

    # ---- Specialist 2: Nutrition Analysis Agent ----
    def _step_nutrition_analysis(self, task: str, history: list[dict[str, Any]], idx: int) -> ChatResponse:
        date_hint = _extract_date_hint(task)
        called_names = [h["tool"] for h in history]
        lower = task.lower()

        if "calculate_daily_totals" not in called_names:
            args = {"date": date_hint} if date_hint else {}
            return tool_call_response([("calculate_daily_totals", args)], start_index=idx)

        if "calculate_macro_split" not in called_names:
            args = {"date": date_hint} if date_hint else {}
            return tool_call_response([("calculate_macro_split", args)], start_index=idx)

        if "get_weekly_trends" not in called_names:
            wk_args: dict[str, Any] = {}
            dates = _DATE_RE.findall(task)
            if len(dates) >= 2:
                wk_args["start_date"] = dates[0]
                wk_args["end_date"] = dates[1]
            elif len(dates) == 1:
                wk_args["end_date"] = dates[0]
            days_m = re.search(r"\b(\d{1,2})\s*[- ]?days?\b", lower)
            if days_m:
                wk_args["days"] = int(days_m.group(1))
            return tool_call_response([("get_weekly_trends", wk_args)], start_index=idx)

        wk_calls = [h for h in history if h["tool"] == "get_weekly_trends"]
        if "compare" in lower and "week" in lower and len(wk_calls) < 2:
            cur_end = (wk_calls[0].get("result") or {}).get("period", {}).get("end")
            if cur_end:
                prev_end = (date.fromisoformat(cur_end) - timedelta(days=7)).isoformat()
                return tool_call_response([("get_weekly_trends", {"end_date": prev_end, "days": 7})], start_index=idx)

        if "compare_intake_with_targets" not in called_names:
            args = {"date": date_hint} if date_hint else {}
            return tool_call_response([("compare_intake_with_targets", args)], start_index=idx)

        totals_res = next(h["result"] for h in history if h["tool"] == "calculate_daily_totals")
        macro_res = next(h["result"] for h in history if h["tool"] == "calculate_macro_split")
        weekly_res = wk_calls[0]["result"]
        compare_res = next(h["result"] for h in history if h["tool"] == "compare_intake_with_targets")

        if "compare" in lower and "week" in lower and len(wk_calls) >= 2:
            prev_res = wk_calls[1]["result"]
            cur_days = int(weekly_res.get("days_with_data") or 0)
            prev_days = int(prev_res.get("days_with_data") or 0)
            cur_cal = (weekly_res.get("averages") or {}).get("calories", 0.0)
            cur_pro = (weekly_res.get("averages") or {}).get("protein", 0.0)
            prev_cal = (prev_res.get("averages") or {}).get("calories", 0.0)
            prev_pro = (prev_res.get("averages") or {}).get("protein", 0.0)
            cur_p = weekly_res.get("period") or {}
            prev_p = prev_res.get("period") or {}
            if cur_days == 0 and prev_days == 0:
                return text_response(
                    f"No meals recorded for this week ({cur_p.get('start')} to {cur_p.get('end')}) "
                    f"or last week ({prev_p.get('start')} to {prev_p.get('end')})."
                )
            if prev_days == 0:
                return text_response(
                    f"Week-over-week comparison: This week ({cur_p.get('start')} to {cur_p.get('end')}): "
                    f"{cur_days} day(s) logged, average {cur_cal} kcal and {cur_pro} g protein/day. "
                    f"Last week ({prev_p.get('start')} to {prev_p.get('end')}): 0 recorded days, so no prior-week baseline is available."
                )
            cal_delta = round(float(cur_cal) - float(prev_cal), 1)
            pro_delta = round(float(cur_pro) - float(prev_pro), 1)
            cal_sign = "+" if cal_delta > 0 else ""
            pro_sign = "+" if pro_delta > 0 else ""
            return text_response(
                f"Week-over-week comparison: This week ({cur_p.get('start')} to {cur_p.get('end')}, {cur_days} day(s) with data) "
                f"averages {cur_cal} kcal and {cur_pro} g protein/day vs last week ({prev_p.get('start')} to {prev_p.get('end')}, "
                f"{prev_days} day(s) with data) averaging {prev_cal} kcal and {prev_pro} g protein/day "
                f"(change: {cal_sign}{cal_delta} kcal/day, {pro_sign}{pro_delta} g protein/day)."
            )

        wants_weekly = any(kw in lower for kw in ("weekly", "trend", "insight", "7-day", "7 day", "average", "progress"))
        summary_lines = [totals_res.get("message", "")]
        if macro_res.get("has_data"):
            summary_lines.append(macro_res.get("message", ""))
        if compare_res.get("message"):
            summary_lines.append(compare_res["message"])
        if wants_weekly or weekly_res.get("days_with_data", 0) > 1:
            if weekly_res.get("message"):
                summary_lines.append(weekly_res["message"])
            if weekly_res.get("data_period_note"):
                summary_lines.append(weekly_res["data_period_note"])
        return text_response(" ".join(line for line in summary_lines if line))

    # ---- Specialist 3: Goal and Budget Agent ----
    def _step_goal_budget(self, task: str, history: list[dict[str, Any]], idx: int) -> ChatResponse:
        date_hint = _extract_date_hint(task)
        called_names = [h["tool"] for h in history]
        lower = task.lower()
        cal_goal, prot_goal = _extract_goal_numbers(task)
        is_setting_goal = any(kw in lower for kw in ("set", "goal", "target", "change", "update"))

        if is_setting_goal and cal_goal is not None and "set_calorie_target" not in called_names:
            return tool_call_response([("set_calorie_target", {"calories": cal_goal})], start_index=idx)

        if is_setting_goal and prot_goal is not None and "set_protein_target" not in called_names:
            return tool_call_response([("set_protein_target", {"protein": prot_goal})], start_index=idx)

        if "calculate_remaining_targets" not in called_names:
            args = {"date": date_hint} if date_hint else {}
            return tool_call_response([("calculate_remaining_targets", args)], start_index=idx)

        wants_fit_check = any(
            kw in lower for kw in ("fit", "would ", "can i eat", "if i eat", "afford")
        )
        if wants_fit_check and "check_meal_against_budget" not in called_names:
            proposed = _extract_proposed_items(task)
            if proposed:
                chk_args: dict[str, Any] = {"items": proposed}
                if date_hint:
                    chk_args["date"] = date_hint
                return tool_call_response([("check_meal_against_budget", chk_args)], start_index=idx)

        rem_res = next(h["result"] for h in history if h["tool"] == "calculate_remaining_targets")
        msgs = []
        for h in history:
            if h["tool"] in {"set_calorie_target", "set_protein_target"} and h["result"]:
                msgs.append(h["result"].get("message", ""))
        if rem_res.get("message"):
            msgs.append(rem_res["message"])
        for h in history:
            if h["tool"] == "check_meal_against_budget" and h["result"]:
                msgs.append(h["result"].get("message", ""))
        return text_response(" ".join(m for m in msgs if m))

    # ---- Specialist 4: Meal Recommendation Agent ----
    def _step_meal_recommendation(self, task: str, history: list[dict[str, Any]], idx: int) -> ChatResponse:
        called_names = [h["tool"] for h in history]
        prefs = _extract_inline_preferences(task)

        if "get_recommendation_candidates" not in called_names:
            gc_args: dict[str, Any] = {}
            if prefs["diet"]:
                gc_args["diet"] = prefs["diet"]
            if prefs["avoid"]:
                gc_args["avoid"] = prefs["avoid"]
            return tool_call_response([("get_recommendation_candidates", gc_args)], start_index=idx)

        gc_res = next(h["result"] for h in history if h["tool"] == "get_recommendation_candidates")
        candidate_ids = gc_res.get("candidate_ids") or []
        if not candidate_ids or gc_res.get("status") == "failed":
            return text_response(gc_res.get("message") or "No matching recommendation candidates were found.")

        if "estimate_meal_nutrition" not in called_names:
            return tool_call_response(
                [("estimate_meal_nutrition", {"candidate_ids": candidate_ids})],
                start_index=idx,
            )

        if "rank_meal_candidates" not in called_names:
            return tool_call_response([("rank_meal_candidates", {"top_n": 5})], start_index=idx)

        rk_res = next(h["result"] for h in history if h["tool"] == "rank_meal_candidates")
        best_id = rk_res.get("best_id")
        if not best_id or rk_res.get("status") == "failed":
            return text_response(rk_res.get("message") or "Could not rank meal candidates for remaining targets.")

        if "validate_recommendation_fit" not in called_names:
            return tool_call_response([("validate_recommendation_fit", {"candidate_id": best_id})], start_index=idx)

        vf_res = next(h["result"] for h in history if h["tool"] == "validate_recommendation_fit")
        ranked = rk_res.get("ranked") or []
        top = ranked[0] if ranked else {}
        alts = [r.get("name") for r in ranked[1:3] if r.get("name")]
        lines = [
            f"Recommended meal: {top.get('name', best_id)} "
            f"({top.get('calories', '?')} kcal, {top.get('protein', '?')} g protein, "
            f"{top.get('carbs', '?')} g carbs, {top.get('fat', '?')} g fat).",
            f"Why it fits: {top.get('reason', vf_res.get('message', ''))}",
        ]
        if alts:
            lines.append(f"Alternatives: {'; '.join(alts)}.")
        lines.append(rk_res.get("disclaimer") or "Estimated nutrition values; not medical or dietary advice.")
        return text_response(" ".join(lines))

    # ---- Manager: Nutrition Manager Agent ----
    def _step_nutrition_manager(self, task: str, history: list[dict[str, Any]], idx: int) -> ChatResponse:
        called_names = [h["tool"] for h in history]

        if "update_workflow_status" not in called_names:
            return tool_call_response(
                [("update_workflow_status", {"status": "planning", "detail": summarize(task, 120)})],
                start_index=idx,
            )

        if "read_shared_state" not in called_names:
            return tool_call_response(
                [
                    (
                        "read_shared_state",
                        {"sections": ["targets", "consumed", "remaining", "meals", "preferences"]},
                    )
                ],
                start_index=idx,
            )

        planned = plan_specialists_for_request(task)
        delegation_calls = [h for h in history if h["tool"] == "delegate_to_specialist"]
        delegated_keys = [h["arguments"].get("specialist") for h in delegation_calls]
        failed_delegation = next(
            (h for h in delegation_calls if (h.get("result") or {}).get("status") == "failed"),
            None,
        )

        # Non-food / unsupported request: polite capability response without specialist noise
        if not planned:
            if not any(
                h["tool"] == "update_workflow_status" and h["arguments"].get("status") == "completed"
                for h in history
            ):
                if "append_run_log" not in called_names:
                    return tool_call_response(
                        [
                            (
                                "append_run_log",
                                {
                                    "agent": "Nutrition Manager Agent",
                                    "task": summarize(task, 120),
                                    "tool": "route_request",
                                    "status": "success",
                                    "output_summary": "Non-food request handled politely without specialist delegation.",
                                },
                            )
                        ],
                        start_index=idx,
                    )
                return tool_call_response(
                    [("update_workflow_status", {"status": "completed", "detail": "Handled non-food query"})],
                    start_index=idx,
                )
            return text_response(
                "I'm NutriTrack, your multi-agent nutrition assistant. Ask me to log meals "
                "(e.g., 'I ate 3 boiled eggs and 2 bananas'), check today's calories and protein, "
                "update calorie/protein goals, recommend your next meal, or show weekly nutrition insights."
            )

        # Abort remaining delegation if a specialist already failed critically
        if failed_delegation is not None:
            res = failed_delegation["result"] or {}
            err_code = res.get("code") or "specialist_failed"
            err_msg = res.get("message") or "Specialist agent failed."
            if "validate_agent_results" not in called_names:
                return tool_call_response(
                    [("validate_agent_results", {"expected_specialists": delegated_keys})],
                    start_index=idx,
                )
            if "append_run_log" not in called_names:
                return tool_call_response(
                    [
                        (
                            "append_run_log",
                            {
                                "agent": "Nutrition Manager Agent",
                                "task": summarize(task, 120),
                                "tool": "handle_specialist_failure",
                                "status": "failed",
                                "error_code": str(err_code),
                                "error_message": summarize(err_msg, 200),
                            },
                        )
                    ],
                    start_index=idx,
                )
            if not any(
                h["tool"] == "update_workflow_status" and h["arguments"].get("status") == "failed"
                for h in history
            ):
                return tool_call_response(
                    [("update_workflow_status", {"status": "failed", "detail": summarize(err_msg, 120)})],
                    start_index=idx,
                )
            return text_response(
                f"Unable to complete your request safely because a specialist failed ({err_code}): {err_msg}"
            )

        # Delegate one specialist at a time in causal order
        if len(delegation_calls) < len(planned):
            next_spec = planned[len(delegation_calls)]
            return tool_call_response(
                [("delegate_to_specialist", {"specialist": next_spec, "task": task})],
                start_index=idx,
            )

        # Validate results
        if "validate_agent_results" not in called_names:
            return tool_call_response(
                [("validate_agent_results", {"expected_specialists": planned})],
                start_index=idx,
            )

        # Re-read fresh shared state after all specialists have updated it
        read_calls = [h for h in history if h["tool"] == "read_shared_state"]
        if len(read_calls) < 2:
            return tool_call_response(
                [
                    (
                        "read_shared_state",
                        {"sections": ["targets", "consumed", "remaining", "meals", "recommendations", "preferences"]},
                    )
                ],
                start_index=idx,
            )

        if "append_run_log" not in called_names:
            return tool_call_response(
                [
                    (
                        "append_run_log",
                        {
                            "agent": "Nutrition Manager Agent",
                            "task": summarize(task, 120),
                            "tool": "compile_final_response",
                            "status": "success",
                            "output_summary": f"Coordinated specialists: {', '.join(planned)}",
                        },
                    )
                ],
                start_index=idx,
            )

        if not any(
            h["tool"] == "update_workflow_status" and h["arguments"].get("status") == "completed"
            for h in history
        ):
            return tool_call_response(
                [("update_workflow_status", {"status": "completed", "detail": "Compiled final response"})],
                start_index=idx,
            )

        # Build concise grounded final response from specialist outputs + fresh shared state
        val_res = next((h["result"] for h in history if h["tool"] == "validate_agent_results"), {}) or {}
        fresh_state = (read_calls[-1].get("result") or {}).get("state", {})
        specialist_summaries = [
            (h["result"] or {}).get("specialist_response", "").strip()
            for h in delegation_calls
            if (h["result"] or {}).get("specialist_response")
        ]
        response_parts = list(specialist_summaries)
        open_qs = [q.get("question") for q in val_res.get("open_questions", []) if q.get("question")]
        for q in open_qs:
            if not any(q in part for part in response_parts):
                response_parts.append(q)

        consumed = fresh_state.get("consumed", {})
        remaining = fresh_state.get("remaining", {})
        if "meal_intake" in planned and consumed:
            response_parts.append(
                f"Updated totals: {consumed.get('calories', 0)} kcal, "
                f"{consumed.get('protein', 0)} g protein consumed; "
                f"remaining: {remaining.get('calories', 0)} kcal, {remaining.get('protein', 0)} g protein."
            )
        return text_response("\n".join(p for p in response_parts if p))
