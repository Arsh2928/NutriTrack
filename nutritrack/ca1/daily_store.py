"""
daily_store.py
---------------
Adds day-to-day persistence on top of SessionMemory (memory.py).

Three jobs, all backed by a single JSON file (daily_log.json) next to
this module:

    1. SAME-DAY PERSISTENCE
       Every meal/budget change is written to disk immediately, so
       today's log survives closing and reopening the notebook, the
       terminal agent, or the web server — not just one Python process.

    2. AUTOMATIC DAY ROLLOVER
       The first agent turn of a new calendar day detects the date has
       changed, archives the previous day's final totals into history,
       and starts a fresh empty log for today — carrying the same daily
       budget forward unless the user sets a new one.

    3. HISTORY
       Past days are kept under "history" so nothing is lost. The agent
       (or the web frontend) can show a "last 7 days" view.

This is intentionally a flat JSON file, not a database — it matches the
scale and simplicity expected for a CA1 project, and needs no setup.
"""

from __future__ import annotations
import copy
import json
import logging
import os
import tempfile
from datetime import date, datetime
from typing import Optional

from .memory import SessionMemory, MealEntry
from ..paths import data_dir

log = logging.getLogger(__name__)

_STORE_FILE = str(data_dir() / "daily_log.json")

_EMPTY_STORE = {
    "current_date": None,
    "budget": None,
    "meals": [],
    "history": {},
}


def _today_str() -> str:
    return date.today().isoformat()


def _load_store() -> dict:
    # deepcopy: the defaults contain a list and a dict, and a shallow copy would
    # share them with every caller (audit defect D5).
    if not os.path.exists(_STORE_FILE):
        return copy.deepcopy(_EMPTY_STORE)
    try:
        with open(_STORE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("daily log must be a JSON object")
        # Guard against a partially-written or older-shaped file
        for key, default in _EMPTY_STORE.items():
            data.setdefault(key, copy.deepcopy(default))
        return data
    except Exception as exc:
        # A corrupt file must never stop the agent from running -- but it must
        # not be silently overwritten either. Keep it under a new name first.
        backup = f"{_STORE_FILE}.corrupt-{datetime.now():%Y%m%d%H%M%S}"
        try:
            os.replace(_STORE_FILE, backup)
            log.error("Daily log was unreadable (%s); kept a copy at %s and started fresh.", exc, backup)
        except OSError:
            log.error("Daily log was unreadable (%s) and could not be backed up.", exc)
        return copy.deepcopy(_EMPTY_STORE)


def _save_store(store: dict) -> bool:
    """Atomic write (temp file + os.replace) so a crash can never leave a
    half-written file. Returns False -- and logs the reason -- if the save
    failed. The session still keeps working (data stays correct in memory),
    but the failure is no longer silent."""
    directory = os.path.dirname(_STORE_FILE) or "."
    tmp = None
    try:
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".daily_log_", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(store, f, indent=2)
        os.replace(tmp, _STORE_FILE)
        return True
    except Exception as exc:
        log.error("Could not save the daily log to %s: %s", _STORE_FILE, exc)
        if tmp and os.path.exists(tmp):
            os.remove(tmp)
        return False


def sync_after_change(memory: SessionMemory) -> bool:
    """
    Call after any mutation (set_budget / add_meal / delete_meal / reset)
    to persist today's current state to disk immediately.
    Returns False if the save failed (see the log); callers may ignore it.
    """
    store = _load_store()
    store["current_date"] = _today_str()
    store["budget"] = memory.daily_budget
    store["meals"] = [m.to_dict() for m in memory.meals]
    return _save_store(store)


def load_or_rollover(memory: SessionMemory) -> dict:
    """
    Call at the START of every agent turn (e.g. the first line of
    agent.chat()). Handles two situations automatically:

        A. Same calendar day, fresh process (e.g. notebook kernel was
           restarted, or the web server restarted) → restores today's
           already-logged meals and budget back into `memory`.

        B. A NEW calendar day has started since the last saved entry →
           archives the previous day's final totals into history, then
           resets `memory` for a clean day (the budget carries over
           unless the user sets a new one afterwards).

    Returns:
        {"action": "restored" | "rolled_over" | "none",
         "previous_date": str | None}
    """
    store = _load_store()
    today = _today_str()

    if store["current_date"] is None:
        # Very first run ever — nothing to restore or roll over.
        store["current_date"] = today
        _save_store(store)
        return {"action": "none", "previous_date": None}

    if store["current_date"] == today:
        # Same day. If this is a fresh process but disk already has today's
        # budget and/or meals, restore them. (The budget is restored even when
        # no meal has been logged yet -- audit defect D5.)
        restored = False
        if store.get("budget") and not memory.has_budget():
            memory.set_budget(store["budget"])
            restored = True
        if not memory.meals and store["meals"]:
            for m in store["meals"]:
                memory.add_meal(MealEntry(
                    food=m["food"],
                    quantity_str=m["quantity"],
                    calories=m["calories"],
                    protein=m["protein"],
                    carbs=m["carbs"],
                    fat=m["fat"],
                ))
            restored = True
        return {"action": "restored" if restored else "none", "previous_date": None}

    # --- Date has changed: archive yesterday, start a fresh today -------
    prev_date = store["current_date"]
    prev_meals = store["meals"]
    store["history"][prev_date] = {
        "budget": store["budget"],
        "meals": prev_meals,
        "total_calories": round(sum(m["calories"] for m in prev_meals), 1),
        "total_protein": round(sum(m["protein"] for m in prev_meals), 1),
        "total_carbs": round(sum(m["carbs"] for m in prev_meals), 1),
        "total_fat": round(sum(m["fat"] for m in prev_meals), 1),
    }

    carried_budget = store["budget"]  # same budget carries into the new day
    memory.reset_meals()
    if carried_budget:
        memory.set_budget(carried_budget)

    store["current_date"] = today
    store["meals"] = []
    _save_store(store)

    return {"action": "rolled_over", "previous_date": prev_date}


def get_history(limit: Optional[int] = 14) -> dict:
    """
    Return the saved history of past (fully-closed) days, most recent
    first. `limit` caps how many days are returned (None = all).
    """
    history = _load_store().get("history", {})
    dates_sorted = sorted(history.keys(), reverse=True)
    if limit is not None:
        dates_sorted = dates_sorted[:limit]
    return {d: history[d] for d in dates_sorted}
