"""Fast deterministic Python pipeline and intelligent router for straightforward chat & diary operations."""

from __future__ import annotations

import math
import re
import time
import uuid
from typing import Any, Optional

from ..core.nutrition import coerce_diary_portion, resolve_item
from ..core.quantity import (
    extract_date_from_text,
    is_non_food_request,
    parse_item_text,
    split_meal_text_with_meta,
)
from ..state import StateError, summarize
from .analysis_tools import compute_daily_totals_data, compute_insights_bundle, compute_macro_split_data
from .common import fingerprint, fmt, normalise_items, resolve_items, to_state_entry
from .performance import get_perf_monitor

_COMPLEX_CHAT_RE = re.compile(
    r"\b("
    r"recommend|suggest|what should i eat|what can i eat|next meal|meal idea|meal plan|"
    r"fit|afford|budget|if i eat|would |can i eat|"
    r"set |change |update |calorie goal|calorie target|protein goal|protein target|target to|goal to|"
    r"custom food|register food|define food|"
    r"why|explain|advice|coach|recipe|plan|compare|unusual"
    r")\b",
    re.I,
)
_LOG_VERB_RE = re.compile(
    r"\b(?:log|add|ate|eaten|eat|had|have|consumed|drank|record|track|enter)\b|"
    r"\bfor\s+(?:breakfast|brunch|lunch|dinner|supper|snacks?)\b",
    re.I,
)
_REMOVE_LEAD_RE = re.compile(r"^(?:please\s+)?(?:remove|delete|undo)\b", re.I)
_REMOVE_LAST_RE = re.compile(r"\b(?:my\s+)?(?:the\s+)?last(?:\s+(?:logged\s+)?(?:meal|food|item|entry))?\b", re.I)
_MEAL_OR_BATCH_ID_RE = re.compile(r"\b(m-[a-z0-9-]+|meal-\d{8}-[a-z0-9-]+|b-[a-z0-9-]+|batch-[a-z0-9-]+)\b", re.I)
_SIMPLE_WEEKLY_RE = re.compile(
    r"^(?:please\s+)?(?:show|give|get|analy[sz]e|check|view|display)?\s*(?:me\s+)?(?:my\s+)?"
    r"(?:7[- ]?day|weekly)\s+(?:nutrition|calories?|proteins?|macros?|progress|summary|insights?|trends?|analytics)"
    r"(?:\s+(?:insights?|trends?|summary|report))?[?.!]*$",
    re.I,
)
_SIMPLE_HISTORY_RE = re.compile(
    r"^(?:please\s+)?(?:(?:show|view|display|get|list|check)\s+(?:me\s+)?(?:my\s+)?(?:recent\s+)?)?"
    r"meal\s+history"
    r"(?:\s+(?:for\s+)?(?:today|yesterday|this\s+week|on\s+\d{4}-\d{2}-\d{2}|\d{4}-\d{2}-\d{2}))?[?.!]*$",
    re.I,
)
_SIMPLE_REMAINING_RE = re.compile(
    r"^(?:please\s+)?(?:show|tell|give|check|what(?:'s|\s+is|\s+are))\s+(?:me\s+)?(?:my\s+)?"
    r"remaining\s+(?:calories?|protein|macros?|budget|targets?)"
    r"(?:\s+(?:for\s+)?(?:today|yesterday|on\s+\d{4}-\d{2}-\d{2}|\d{4}-\d{2}-\d{2}))?[?.!]*$",
    re.I,
)
_SIMPLE_DAILY_RE = re.compile(
    r"^(?:please\s+)?(?:show|tell|give|check|what(?:'s|\s+is|\s+are)|how\s+(?:many|much))\s+"
    r"(?:me\s+)?(?:my\s+)?(?:(?:today'?s|yesterday'?s)\s+)?"
    r"(?:calories?|protein|macros?|macro\s+split|nutrition|totals?|daily\s+totals?|summary)"
    r"(?:\s+(?:and\s+(?:protein|calories?|macros?|carbs|fat|fiber|fibre)))?"
    r"(?:\s+(?:did\s+i\s+(?:eat|have|log|consume)|have\s+i\s+(?:eaten|had|logged|consumed)|consumed|logged))?"
    r"(?:\s+(?:today|yesterday|on\s+\d{4}-\d{2}-\d{2}|\d{4}-\d{2}-\d{2}))?[?.!]*$",
    re.I,
)
_SIMPLE_WHAT_ATE_RE = re.compile(
    r"^(?:what|show\s+what)\s+(?:did\s+i\s+eat|meals?\s+(?:did\s+i\s+(?:eat|log)|i\s+logged))\s*"
    r"(?:today|yesterday|on\s+\d{4}-\d{2}-\d{2})?[?.!]*$",
    re.I,
)


def fast_log_meal(
    state_manager: Any,
    run_log: Any,
    *,
    message: Optional[str] = None,
    items: Optional[list[dict[str, Any]]] = None,
    user_id: str = "demo_user",
    date: Optional[str] = None,
    meal_type: Optional[str] = None,
    request_id: Optional[str] = None,
    ai_status: Optional[dict[str, Any]] = None,
    mode_info: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Deterministically parse, validate, resolve, and persist straightforward meal additions without LLM calls."""
    t_total0 = time.perf_counter()
    perf = get_perf_monitor()
    run_id = f"fast-log-{uuid.uuid4().hex[:10]}"
    st = state_manager.get(user_id)
    effective_date = date
    effective_meal_type = meal_type
    data_only_active = bool((ai_status or {}).get("selected_mode") == "offline" or (ai_status or {}).get("data_only_active"))

    t_parse0 = time.perf_counter()
    parsed_items: list[dict[str, Any]] = []
    problems: list[dict[str, Any]] = []

    if items is not None:
        try:
            parsed_items = normalise_items(items)
        except StateError as exc:
            return {
                "status": "failed",
                "error_code": "invalid_input",
                "error_message": str(exc),
                "run_id": run_id,
                "user_id": user_id,
                "state": st.snapshot(day=effective_date),
                "ai_status": ai_status,
                "mode_info": mode_info,
            }
    else:
        raw_text = (message or "").strip()
        if not raw_text:
            return {
                "status": "failed",
                "error_code": "empty_input",
                "error_message": "Please describe what food you ate.",
                "run_id": run_id,
                "user_id": user_id,
                "state": st.snapshot(day=effective_date),
                "ai_status": ai_status,
                "mode_info": mode_info,
            }
        if len(raw_text) > 1000:
            return {
                "status": "failed",
                "error_code": "input_too_long",
                "error_message": "That message is too long to read safely (limit 1000 characters).",
                "run_id": run_id,
                "user_id": user_id,
                "state": st.snapshot(day=effective_date),
                "ai_status": ai_status,
                "mode_info": mode_info,
            }

        m_type, date_hint, invalid_date_err, phrases = split_meal_text_with_meta(raw_text)
        if effective_meal_type is None:
            effective_meal_type = m_type
        if date_hint:
            effective_date = date_hint

        if invalid_date_err:
            snap = st.snapshot(day=None)
            return {
                "status": "needs_input",
                "error_code": "invalid_date",
                "error_message": invalid_date_err,
                "response": f"{invalid_date_err} Please use YYYY-MM-DD, 'today', or 'yesterday'.",
                "open_questions": [{"code": "invalid_date", "message": invalid_date_err}],
                "run_id": run_id,
                "user_id": user_id,
                "date": snap["date"],
                "is_llm": False,
                "data_only_active": data_only_active,
                "execution_mode": "fast_deterministic",
                "activated_specialists": ["meal_intake"],
                "workflow_status": "completed",
                "state": snap,
                "ai_status": ai_status,
                "mode_info": mode_info,
                "timing_ms": {"total_ms": round((time.perf_counter() - t_total0) * 1000.0, 3)},
            }

        foods = [p for p in phrases if not is_non_food_request(p)]
        for phrase in foods:
            p = parse_item_text(phrase)
            if p.problems:
                problems.extend({"item": phrase, **prob} for prob in p.problems)
            if p.food_text:
                qty = p.value
                if qty is None and not (p.unit or p.household_unit):
                    qty = None
                parsed_items.append(
                    {
                        "food": p.food_text,
                        "quantity": qty,
                        "unit": p.unit,
                        "household_unit": p.household_unit,
                    }
                )
    parse_ms = round((time.perf_counter() - t_parse0) * 1000.0, 3)

    if not parsed_items or problems:
        msg = problems[0]["message"] if problems else "I couldn't find any food item in that input."
        try:
            snap = st.snapshot(day=effective_date)
        except StateError as exc:
            snap = st.snapshot(day=None)
            return {
                "status": "failed",
                "error_code": "invalid_input",
                "error_message": str(exc),
                "response": str(exc),
                "run_id": run_id,
                "user_id": user_id,
                "date": snap["date"],
                "is_llm": False,
                "data_only_active": data_only_active,
                "execution_mode": "fast_deterministic",
                "state": snap,
                "ai_status": ai_status,
                "mode_info": mode_info,
            }
        return {
            "status": "needs_input",
            "error_code": problems[0]["code"] if problems else "no_food_found",
            "error_message": msg,
            "response": msg,
            "open_questions": problems or [{"code": "no_food_found", "message": msg}],
            "run_id": run_id,
            "user_id": user_id,
            "date": snap["date"],
            "is_llm": False,
            "data_only_active": data_only_active,
            "execution_mode": "fast_deterministic",
            "activated_specialists": ["meal_intake"],
            "workflow_status": "completed",
            "state": snap,
            "ai_status": ai_status,
            "mode_info": mode_info,
            "timing_ms": {"parse_ms": parse_ms, "total_ms": round((time.perf_counter() - t_total0) * 1000.0, 3)},
        }

    t_lookup0 = time.perf_counter()
    resolved, failures = resolve_items(parsed_items, user_id=user_id)
    lookup_ms = round((time.perf_counter() - t_lookup0) * 1000.0, 3)

    if not resolved:
        first = failures[0]
        q_msg = first.get("question") or first["message"]
        try:
            snap = st.snapshot(day=effective_date)
        except StateError as exc:
            snap = st.snapshot(day=None)
            return {
                "status": "failed",
                "error_code": "invalid_input",
                "error_message": str(exc),
                "response": str(exc),
                "run_id": run_id,
                "user_id": user_id,
                "date": snap["date"],
                "is_llm": False,
                "data_only_active": data_only_active,
                "execution_mode": "fast_deterministic",
                "state": snap,
                "ai_status": ai_status,
                "mode_info": mode_info,
            }
        return {
            "status": "needs_input",
            "error_code": first["code"],
            "error_message": first["message"],
            "response": f"{first['message']} {q_msg}".strip(),
            "open_questions": failures,
            "unresolved": failures,
            "run_id": run_id,
            "user_id": user_id,
            "date": snap["date"],
            "is_llm": False,
            "data_only_active": data_only_active,
            "execution_mode": "fast_deterministic",
            "activated_specialists": ["meal_intake"],
            "workflow_status": "completed",
            "state": snap,
            "ai_status": ai_status,
            "mode_info": mode_info,
            "timing_ms": {
                "parse_ms": parse_ms,
                "lookup_ms": lookup_ms,
                "total_ms": round((time.perf_counter() - t_total0) * 1000.0, 3),
            },
        }

    entries = [to_state_entry(r["nutrition"]) for r in resolved]
    rid = request_id or f"{run_id}:{fingerprint(entries)}"

    t_state0 = time.perf_counter()
    try:
        saved = st.add_meals(
            entries,
            request_id=rid,
            meal_type=effective_meal_type,
            day=effective_date,
            source="meal_intake",
        )
    except StateError as exc:
        snap = st.snapshot(day=None)
        return {
            "status": "failed",
            "error_code": "invalid_input",
            "error_message": str(exc),
            "response": str(exc),
            "run_id": run_id,
            "user_id": user_id,
            "date": snap["date"],
            "is_llm": False,
            "data_only_active": data_only_active,
            "execution_mode": "fast_deterministic",
            "state": snap,
            "ai_status": ai_status,
            "mode_info": mode_info,
        }
    state_ms = round((time.perf_counter() - t_state0) * 1000.0, 3)

    t_log0 = time.perf_counter()
    added_kcal = round(math.fsum(e["calories"] for e in saved["added"]), 1)
    added_prot = round(math.fsum(e["protein"] for e in saved["added"]), 1)
    added_fiber = round(math.fsum(e.get("fiber", 0.0) for e in saved["added"]), 1)
    snap = st.snapshot(day=saved["date"])

    if saved["duplicate"]:
        reply = "Duplicate request ignored; your saved meal log was kept unchanged."
    else:
        item_desc = ", ".join(f"{e['quantity_str']} {e['food']}" for e in saved["added"])
        reply = (
            f"Logged {item_desc} ({fmt(added_kcal)} kcal, {fmt(added_prot)} g protein, "
            f"{fmt(added_fiber)} g fiber) for {saved['date']}. "
            f"Day total: {fmt(snap['consumed']['calories'])} kcal."
        )
        if failures:
            unresolved_names = ", ".join(f"'{f['food']}'" for f in failures)
            reply += f" Could not log unknown/unclear item(s) without your values: {unresolved_names}."

    if run_log is not None:
        run_log.append(
            run_id=run_id,
            user_id=user_id,
            agent="Meal Intake Agent",
            task=summarize(message or f"fast_log:{len(entries)}_items", 160),
            tool="fast_log_meal",
            status="success",
            input_summary=summarize({"date": saved["date"], "meal_type": effective_meal_type, "count": len(entries)}),
            output_summary=summarize(reply, 200),
            execution_mode="fast_deterministic",
        )
    log_ms = round((time.perf_counter() - t_log0) * 1000.0, 3)
    total_ms = round((time.perf_counter() - t_total0) * 1000.0, 3)
    perf.record("parse_meal_entry", parse_ms)
    perf.record("food_lookup", lookup_ms)
    perf.record("state_add_meals", state_ms)
    perf.record("fast_log_meal", total_ms)

    status = "needs_input" if failures else "success"
    return {
        "status": status,
        "duplicate": bool(saved["duplicate"]),
        "run_id": run_id,
        "user_id": user_id,
        "date": snap["date"],
        "response": reply,
        "is_llm": False,
        "data_only_active": data_only_active,
        "execution_mode": "fast_deterministic",
        "activated_specialists": ["meal_intake"],
        "manager_tools": ["fast_log_meal"],
        "added": saved["added"],
        "meal_ids": saved["meal_ids"],
        "unresolved": failures,
        "open_questions": failures,
        "workflow_status": "completed",
        "state": snap,
        "ai_status": ai_status,
        "mode_info": mode_info,
        "timing_ms": {
            "parse_ms": parse_ms,
            "lookup_ms": lookup_ms,
            "state_ms": state_ms,
            "log_ms": log_ms,
            "total_ms": total_ms,
        },
    }


def fast_remove_meal(
    state_manager: Any,
    run_log: Any,
    *,
    meal_id: Optional[str] = None,
    food_name: Optional[str] = None,
    batch_id: Optional[str] = None,
    remove_last: bool = False,
    user_id: str = "demo_user",
    date: Optional[str] = None,
    ai_status: Optional[dict[str, Any]] = None,
    mode_info: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Deterministically remove a saved meal without LLM or MAF round-trip overhead."""
    t0 = time.perf_counter()
    perf = get_perf_monitor()
    run_id = f"fast-del-{uuid.uuid4().hex[:10]}"
    st = state_manager.get(user_id)
    data_only_active = bool((ai_status or {}).get("selected_mode") == "offline" or (ai_status or {}).get("data_only_active"))
    try:
        removed = st.remove_meal(
            meal_id=meal_id,
            batch_id=batch_id,
            food_name=food_name,
            day=date,
            remove_last=remove_last,
        )
    except StateError as exc:
        return {
            "status": "failed",
            "error_code": "invalid_input",
            "error_message": str(exc),
            "run_id": run_id,
            "user_id": user_id,
            "is_llm": False,
            "data_only_active": data_only_active,
            "execution_mode": "fast_deterministic",
            "state": st.snapshot(day=None),
            "ai_status": ai_status,
            "mode_info": mode_info,
        }

    if not removed.get("removed"):
        return {
            "status": "failed",
            "error_code": "not_found",
            "error_message": "Meal not found.",
            "response": "I couldn't find a matching logged meal to remove.",
            "run_id": run_id,
            "user_id": user_id,
            "is_llm": False,
            "data_only_active": data_only_active,
            "execution_mode": "fast_deterministic",
            "activated_specialists": ["meal_intake"],
            "workflow_status": "completed",
            "state": st.snapshot(day=date),
            "ai_status": ai_status,
            "mode_info": mode_info,
        }

    day = (removed.get("dates") or [st.resolve_date(date)])[0]
    snap = st.snapshot(day=day)
    gone = ", ".join(f"{m['quantity_str']} {m['food']}" for m in removed["removed"])
    reply = f"Removed {gone}. Day total: {fmt(snap['consumed']['calories'])} kcal."
    if run_log is not None:
        run_log.append(
            run_id=run_id,
            user_id=user_id,
            agent="Meal Intake Agent",
            task=f"remove meal {meal_id or food_name or batch_id or ('last' if remove_last else '')}",
            tool="fast_remove_meal",
            status="success",
            output_summary=summarize(reply, 200),
            execution_mode="fast_deterministic",
        )
    total_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    perf.record("fast_remove_meal", total_ms)
    return {
        "status": "success",
        "run_id": run_id,
        "user_id": user_id,
        "date": snap["date"],
        "response": reply,
        "removed": removed["removed"],
        "is_llm": False,
        "data_only_active": data_only_active,
        "execution_mode": "fast_deterministic",
        "activated_specialists": ["meal_intake"],
        "manager_tools": ["fast_remove_meal"],
        "workflow_status": "completed",
        "state": snap,
        "ai_status": ai_status,
        "mode_info": mode_info,
        "timing_ms": {"total_ms": total_ms},
    }


def _fast_date_error_response(
    snap: dict[str, Any],
    *,
    run_id: str,
    user_id: str,
    message: str,
    ai_status: Optional[dict[str, Any]],
    mode_info: Optional[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "status": "needs_input",
        "error_code": "invalid_date",
        "error_message": message,
        "response": f"{message} Please use YYYY-MM-DD, 'today', or 'yesterday'.",
        "run_id": run_id,
        "user_id": user_id,
        "date": snap["date"],
        "is_llm": False,
        "data_only_active": bool((ai_status or {}).get("selected_mode") == "offline"),
        "execution_mode": "fast_deterministic",
        "state": snap,
        "ai_status": ai_status,
        "mode_info": mode_info,
    }


def _fast_daily_summary(
    state_manager: Any,
    run_log: Any,
    *,
    message: str,
    user_id: str,
    date: Optional[str],
    ai_status: Optional[dict[str, Any]],
    mode_info: Optional[dict[str, Any]],
) -> dict[str, Any]:
    t0 = time.perf_counter()
    run_id = f"fast-sum-{uuid.uuid4().hex[:10]}"
    st = state_manager.get(user_id)
    date_hint, date_err, _ = extract_date_from_text(message)
    if date_err:
        return _fast_date_error_response(
            st.snapshot(day=None), run_id=run_id, user_id=user_id, message=date_err, ai_status=ai_status, mode_info=mode_info
        )
    try:
        target_day = st.resolve_date(date_hint or date)
    except StateError as exc:
        return _fast_date_error_response(
            st.snapshot(day=None), run_id=run_id, user_id=user_id, message=str(exc), ai_status=ai_status, mode_info=mode_info
        )
    snap = st.snapshot(day=target_day)
    totals = st.day_totals(target_day)
    daily = compute_daily_totals_data(st, target_day, totals=totals, targets=snap["targets"])
    macro = compute_macro_split_data(target_day, totals)
    parts = [daily["message"]]
    if macro.get("has_data"):
        parts.append(macro["message"])
    reply = " ".join(parts)
    if run_log is not None:
        run_log.append(
            run_id=run_id,
            user_id=user_id,
            agent="Nutrition Analysis Agent",
            task=summarize(message, 160),
            tool="calculate_daily_totals",
            status="success",
            output_summary=summarize(reply, 200),
            execution_mode="fast_deterministic",
        )
    total_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    get_perf_monitor().record("fast_daily_summary", total_ms)
    return {
        "status": "success",
        "run_id": run_id,
        "user_id": user_id,
        "date": snap["date"],
        "response": reply,
        "is_llm": False,
        "data_only_active": bool((ai_status or {}).get("selected_mode") == "offline"),
        "execution_mode": "fast_deterministic",
        "activated_specialists": ["nutrition_analysis"],
        "manager_tools": ["calculate_daily_totals"],
        "workflow_status": "completed",
        "state": snap,
        "ai_status": ai_status,
        "mode_info": mode_info,
        "timing_ms": {"total_ms": total_ms},
    }


def _fast_remaining_summary(
    state_manager: Any,
    run_log: Any,
    *,
    message: str,
    user_id: str,
    date: Optional[str],
    ai_status: Optional[dict[str, Any]],
    mode_info: Optional[dict[str, Any]],
) -> dict[str, Any]:
    t0 = time.perf_counter()
    run_id = f"fast-rem-{uuid.uuid4().hex[:10]}"
    st = state_manager.get(user_id)
    date_hint, date_err, _ = extract_date_from_text(message)
    if date_err:
        return _fast_date_error_response(
            st.snapshot(day=None), run_id=run_id, user_id=user_id, message=date_err, ai_status=ai_status, mode_info=mode_info
        )
    try:
        target_day = st.resolve_date(date_hint or date)
    except StateError as exc:
        return _fast_date_error_response(
            st.snapshot(day=None), run_id=run_id, user_id=user_id, message=str(exc), ai_status=ai_status, mode_info=mode_info
        )
    snap = st.snapshot(day=target_day)
    rem = snap.get("remaining") or {}
    targets = snap.get("targets") or {}
    cal_left = fmt(float(rem.get("calories", 0.0)))
    prot_left = fmt(float(rem.get("protein", 0.0)))
    reply = (
        f"{snap['date']}: {cal_left} kcal remaining (target {fmt(float(targets.get('calories', 0.0)))} kcal); "
        f"{prot_left} g protein remaining."
    )
    if run_log is not None:
        run_log.append(
            run_id=run_id,
            user_id=user_id,
            agent="Goal and Budget Agent",
            task=summarize(message, 160),
            tool="calculate_remaining_targets",
            status="success",
            output_summary=summarize(reply, 200),
            execution_mode="fast_deterministic",
        )
    total_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    get_perf_monitor().record("fast_remaining_summary", total_ms)
    return {
        "status": "success",
        "run_id": run_id,
        "user_id": user_id,
        "date": snap["date"],
        "response": reply,
        "is_llm": False,
        "data_only_active": bool((ai_status or {}).get("selected_mode") == "offline"),
        "execution_mode": "fast_deterministic",
        "activated_specialists": ["goal_budget"],
        "manager_tools": ["calculate_remaining_targets"],
        "workflow_status": "completed",
        "state": snap,
        "ai_status": ai_status,
        "mode_info": mode_info,
        "timing_ms": {"total_ms": total_ms},
    }


def _fast_meal_history(
    state_manager: Any,
    run_log: Any,
    *,
    message: str,
    user_id: str,
    date: Optional[str],
    ai_status: Optional[dict[str, Any]],
    mode_info: Optional[dict[str, Any]],
) -> dict[str, Any]:
    t0 = time.perf_counter()
    run_id = f"fast-hist-{uuid.uuid4().hex[:10]}"
    st = state_manager.get(user_id)
    date_hint, date_err, _ = extract_date_from_text(message)
    if date_err:
        return _fast_date_error_response(
            st.snapshot(day=None), run_id=run_id, user_id=user_id, message=date_err, ai_status=ai_status, mode_info=mode_info
        )
    try:
        anchor_day = st.resolve_date(date_hint or date)
    except StateError as exc:
        return _fast_date_error_response(
            st.snapshot(day=None), run_id=run_id, user_id=user_id, message=str(exc), ai_status=ai_status, mode_info=mode_info
        )
    snap = st.snapshot(day=anchor_day)
    if date_hint:
        meals = snap.get("meals") or []
        if not meals:
            reply = f"No meals recorded for {snap['date']}."
        else:
            desc = ", ".join(f"{m['quantity_str']} {m['food']} ({fmt(float(m['calories']))} kcal)" for m in meals)
            reply = f"Meal history for {snap['date']} ({len(meals)} item(s), {fmt(float(snap['consumed']['calories']))} kcal): {desc}."
    else:
        hist_rows = st.history(days=14)
        if not hist_rows:
            reply = "No meals have been logged in your recent history yet."
        else:
            parts = []
            for row in reversed(hist_rows[-7:]):
                d_str = row.get("date")
                d_snap = st.snapshot(day=d_str)
                d_meals = d_snap.get("meals") or []
                items_txt = ", ".join(f"{m['quantity_str']} {m['food']}" for m in d_meals[:5])
                if len(d_meals) > 5:
                    items_txt += f", +{len(d_meals) - 5} more"
                parts.append(f"{d_str}: {fmt(float(d_snap['consumed']['calories']))} kcal ({items_txt})")
            reply = f"Recent meal history ({len(hist_rows)} day(s) with data): " + "; ".join(parts) + "."
    if run_log is not None:
        run_log.append(
            run_id=run_id,
            user_id=user_id,
            agent="Nutrition Analysis Agent",
            task=summarize(message, 160),
            tool="read_shared_state",
            status="success",
            output_summary=summarize(reply, 200),
            execution_mode="fast_deterministic",
        )
    total_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    get_perf_monitor().record("fast_meal_history", total_ms)
    return {
        "status": "success",
        "run_id": run_id,
        "user_id": user_id,
        "date": snap["date"],
        "response": reply,
        "is_llm": False,
        "data_only_active": bool((ai_status or {}).get("selected_mode") == "offline"),
        "execution_mode": "fast_deterministic",
        "activated_specialists": ["nutrition_analysis"],
        "manager_tools": ["read_shared_state"],
        "workflow_status": "completed",
        "state": snap,
        "ai_status": ai_status,
        "mode_info": mode_info,
        "timing_ms": {"total_ms": total_ms},
    }


def _fast_weekly_summary(
    state_manager: Any,
    run_log: Any,
    *,
    message: str,
    user_id: str,
    date: Optional[str],
    ai_status: Optional[dict[str, Any]],
    mode_info: Optional[dict[str, Any]],
) -> dict[str, Any]:
    t0 = time.perf_counter()
    run_id = f"fast-wk-{uuid.uuid4().hex[:10]}"
    st = state_manager.get(user_id)
    date_hint, date_err, _ = extract_date_from_text(message)
    if date_err:
        return _fast_date_error_response(
            st.snapshot(day=None), run_id=run_id, user_id=user_id, message=date_err, ai_status=ai_status, mode_info=mode_info
        )
    try:
        anchor_day = st.resolve_date(date_hint or date)
    except StateError as exc:
        return _fast_date_error_response(
            st.snapshot(day=None), run_id=run_id, user_id=user_id, message=str(exc), ai_status=ai_status, mode_info=mode_info
        )
    bundle = compute_insights_bundle(st, end_date=anchor_day, days=7)
    trends = bundle["trends"]
    reply = trends["message"]
    if trends.get("data_period_note"):
        reply = f"{reply} {trends['data_period_note']}"
    snap = st.snapshot(day=anchor_day)
    if run_log is not None:
        run_log.append(
            run_id=run_id,
            user_id=user_id,
            agent="Nutrition Analysis Agent",
            task=summarize(message, 160),
            tool="get_weekly_trends",
            status="success",
            output_summary=summarize(reply, 200),
            execution_mode="fast_deterministic",
        )
    total_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    get_perf_monitor().record("fast_weekly_summary", total_ms)
    return {
        "status": "success",
        "run_id": run_id,
        "user_id": user_id,
        "date": snap["date"],
        "response": reply,
        "trends": trends,
        "is_llm": False,
        "data_only_active": bool((ai_status or {}).get("selected_mode") == "offline"),
        "execution_mode": "fast_deterministic",
        "activated_specialists": ["nutrition_analysis"],
        "manager_tools": ["get_weekly_trends"],
        "workflow_status": "completed",
        "state": snap,
        "ai_status": ai_status,
        "mode_info": mode_info,
        "timing_ms": {"total_ms": total_ms},
    }


def route_chat_fast(
    state_manager: Any,
    run_log: Any,
    *,
    message: str,
    user_id: str = "demo_user",
    date: Optional[str] = None,
    request_id: Optional[str] = None,
    ai_status: Optional[dict[str, Any]] = None,
    mode_info: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Deterministically handle simple meal logs, removals, and direct state queries without LLM overhead.

    Returns `None` when the message requires multi-agent orchestration or selected LLM reasoning.
    """
    text = (message or "").strip()
    if not text or len(text) > 1000:
        return None
    lower = text.lower()

    # 1. Explicit simple daily, remaining, history, or weekly local analytics queries
    if _SIMPLE_WEEKLY_RE.match(text):
        return _fast_weekly_summary(
            state_manager, run_log, message=text, user_id=user_id, date=date, ai_status=ai_status, mode_info=mode_info
        )
    if _SIMPLE_HISTORY_RE.match(text):
        return _fast_meal_history(
            state_manager, run_log, message=text, user_id=user_id, date=date, ai_status=ai_status, mode_info=mode_info
        )
    if _SIMPLE_REMAINING_RE.match(text):
        return _fast_remaining_summary(
            state_manager, run_log, message=text, user_id=user_id, date=date, ai_status=ai_status, mode_info=mode_info
        )
    if _SIMPLE_DAILY_RE.match(text) or _SIMPLE_WHAT_ATE_RE.match(text):
        return _fast_daily_summary(
            state_manager, run_log, message=text, user_id=user_id, date=date, ai_status=ai_status, mode_info=mode_info
        )

    # Anything asking a complex question, recommendation, budget check, or goal update goes to MAF / LLM
    if "?" in text or _COMPLEX_CHAT_RE.search(lower):
        return None

    # 2. Straightforward meal removal ("Remove my last meal", "Delete banana", "Remove meal m-123")
    if _REMOVE_LEAD_RE.search(lower):
        date_hint, date_err, cleaned = extract_date_from_text(text)
        if date_err:
            st = state_manager.get(user_id)
            snap = st.snapshot(day=None)
            return {
                "status": "needs_input",
                "error_code": "invalid_date",
                "error_message": date_err,
                "response": f"{date_err} Please use YYYY-MM-DD, 'today', or 'yesterday'.",
                "user_id": user_id,
                "date": snap["date"],
                "is_llm": False,
                "execution_mode": "fast_deterministic",
                "state": snap,
                "ai_status": ai_status,
                "mode_info": mode_info,
            }
        effective_date = date_hint or date
        id_m = _MEAL_OR_BATCH_ID_RE.search(cleaned)
        if id_m:
            ident = id_m.group(1).lower()
            if ident.startswith("b-") or ident.startswith("batch-"):
                return fast_remove_meal(
                    state_manager, run_log, batch_id=ident, user_id=user_id, date=effective_date,
                    ai_status=ai_status, mode_info=mode_info,
                )
            return fast_remove_meal(
                state_manager, run_log, meal_id=ident, user_id=user_id, date=effective_date,
                ai_status=ai_status, mode_info=mode_info,
            )
        if _REMOVE_LAST_RE.search(cleaned):
            return fast_remove_meal(
                state_manager, run_log, remove_last=True, user_id=user_id, date=effective_date,
                ai_status=ai_status, mode_info=mode_info,
            )
        food_clean = re.sub(
            r"^(?:please\s+)?(?:remove|delete|undo)\s+(?:my\s+|the\s+|last\s+)?(?:meal\s+)?",
            "",
            cleaned,
            flags=re.I,
        ).strip(" .,:;-")
        if food_clean and not is_non_food_request(food_clean):
            return fast_remove_meal(
                state_manager, run_log, food_name=food_clean, user_id=user_id, date=effective_date,
                ai_status=ai_status, mode_info=mode_info,
            )
        return None

    # 3. Straightforward natural-language meal logging
    _, date_hint, invalid_date_err, phrases = split_meal_text_with_meta(text)
    if invalid_date_err and _LOG_VERB_RE.search(lower):
        return fast_log_meal(
            state_manager, run_log, message=text, user_id=user_id, date=date, request_id=request_id,
            ai_status=ai_status, mode_info=mode_info,
        )
    if not phrases or any(is_non_food_request(p) for p in phrases):
        return None
    has_log_verb = bool(_LOG_VERB_RE.search(lower))
    parsed = [parse_item_text(p) for p in phrases]
    has_qty = any(getattr(p, "explicit_value", False) for p in parsed)
    if has_log_verb or has_qty:
        return fast_log_meal(
            state_manager, run_log, message=text, user_id=user_id, date=date, request_id=request_id,
            ai_status=ai_status, mode_info=mode_info,
        )

    return None
