"""
ca1_import.py
-------------
One-time import of a CA1 `daily_log.json` into a NEW NutritionState document, so
the history you built with CA1 survives the upgrade.

CA1 format:  {"current_date": "YYYY-MM-DD", "budget": float|null,
              "meals": [ {food, quantity, calories, protein, carbs, fat}, ... ],
              "history": { "YYYY-MM-DD": {"budget", "meals": [...], "total_*": ...} } }

Rules
  * Meals are copied with their saved numbers -- nothing is recalculated or estimated.
  * CA1 archived EVERY past day on rollover, including days with no meals, as
    0 kcal. Those empty days are skipped: "nothing was logged" is not "ate nothing".
  * A CA1 budget becomes the calorie target (source "ca1_import"); CA1 had no
    protein target, so older days are compared with the current protein target.
  * All-or-nothing: the document is only modified after the whole file was read.
"""
import math
import uuid
from datetime import date

NUTRIENT_KEYS = ("calories", "protein", "carbs", "fat")


def _entry(meal, day):
    """CA1 meal dict -> state entry, or None if it is unusable."""
    try:
        food = str(meal["food"]).strip()
        qty = str(meal["quantity"]).strip()
        values = {k: float(meal[k]) for k in NUTRIENT_KEYS}
    except (KeyError, TypeError, ValueError):
        return None
    if not food or not qty or any((not math.isfinite(v)) or v < 0 for v in values.values()):
        return None
    return {"meal_id": "m-" + uuid.uuid4().hex[:8], "batch_id": f"b-ca1-{day}", "meal_type": None,
            "logged_at": f"{day}T00:00:00", "date": day, "source": "ca1_import", "request_id": None,
            "food": food[:100], "quantity_str": qty[:60], **{k: round(v, 1) for k, v in values.items()},
            "serving_size": None, "provenance": "ca1_import", "estimated": True}


def _valid_day(text, today):
    try:
        d = date.fromisoformat(str(text))
    except ValueError:
        return None
    return d.isoformat() if d <= date.fromisoformat(today) else None


def _budget(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v > 0 else None


def import_ca1_daily_log(doc, store, today):
    if not isinstance(store, dict):
        raise ValueError("the CA1 daily log must be a JSON object")
    new_days, skipped_meals, skipped_empty = {}, 0, 0

    def add_day(day_text, meals, budget):
        nonlocal skipped_meals, skipped_empty
        day = _valid_day(day_text, today)
        if day is None or not isinstance(meals, list):
            skipped_empty += 1
            return
        entries = []
        for m in meals:
            e = _entry(m, day)
            if e is None:
                skipped_meals += 1
            else:
                entries.append(e)
        if not entries:
            skipped_empty += 1
            return
        new_days.setdefault(day, {"targets": {"calories": _budget(budget), "protein": None}, "meals": []})
        new_days[day]["meals"].extend(entries)

    history = store.get("history") or {}
    if not isinstance(history, dict):
        raise ValueError("'history' must be an object")
    for day_text, rec in history.items():
        if isinstance(rec, dict):
            add_day(day_text, rec.get("meals", []), rec.get("budget"))
    if store.get("current_date"):
        add_day(store["current_date"], store.get("meals", []), store.get("budget"))

    # -- all parsed; now apply ------------------------------------------------
    doc["days"].update(new_days)
    budget = _budget(store.get("budget"))
    if budget is not None:
        doc["targets"]["calories"], doc["targets_source"]["calories"] = budget, "ca1_import"
    return {"imported_days": len(new_days),
            "imported_meals": sum(len(d["meals"]) for d in new_days.values()),
            "skipped_meals": skipped_meals, "skipped_empty_days": skipped_empty,
            "calorie_target_from_ca1": budget}
