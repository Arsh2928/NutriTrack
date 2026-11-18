"""Helpers shared by several tool modules (no tools are defined here)."""
import hashlib

from ..core.nutrition import resolve_item
from ..state import StateError

MAX_ITEMS = 20


def normalise_items(items):
    """Validate an agent-supplied list of {"food", "quantity"?, "unit"?, "household_unit"?}.
    Raises StateError (-> invalid_input) on anything malformed."""
    if not isinstance(items, list) or not items:
        raise StateError("items must be a non-empty list of {food, quantity, unit} objects.")
    if len(items) > MAX_ITEMS:
        raise StateError(f"At most {MAX_ITEMS} items at a time.")
    out = []
    for i, item in enumerate(items, 1):
        if not isinstance(item, dict):
            raise StateError(f"Item {i} must be an object like {{\"food\": \"egg\", \"quantity\": 2}}.")
        food = item.get("food")
        if not isinstance(food, str) or not food.strip():
            raise StateError(f"Item {i} needs a 'food' name.")
        quantity, unit, household = item.get("quantity"), item.get("unit"), item.get("household_unit")
        if quantity is not None and (isinstance(quantity, bool) or not isinstance(quantity, (int, float))):
            raise StateError(f"Item {i}: 'quantity' must be a number (got {quantity!r}).")
        if unit is not None and not isinstance(unit, str):
            raise StateError(f"Item {i}: 'unit' must be text such as 'g', 'ml' or 'piece'.")
        out.append({"food": food.strip(), "quantity": quantity, "unit": unit,
                    "household_unit": household if isinstance(household, str) else None})
    return out


def resolve_items(items, *, user_id=None):
    """Resolve normalised items with the shared engine -> (resolved, failures).
    resolved:  [{"index", "request": item, "nutrition": {...}}]
    failures:  [{"index", "food", "code", "message", "question", "did_you_mean"}]"""
    resolved, failures = [], []
    for index, item in enumerate(items, 1):
        res = resolve_item(item["food"], item["quantity"], item["unit"], item["household_unit"], user_id=user_id)
        if res["ok"]:
            resolved.append({"index": index, "request": item, "nutrition": res})
        else:
            failures.append({"index": index, "food": item["food"], "code": res["reason"],
                             "message": res["message"], "question": res.get("question"),
                             "did_you_mean": res.get("did_you_mean", []),
                             "closest_match": res.get("closest_match")})
    return resolved, failures


def to_state_entry(nutrition):
    """Resolved nutrition -> the entry stored in the shared state."""
    entry = {"food": nutrition["food"], "quantity_str": nutrition["quantity_str"],
             "calories": nutrition["calories"], "protein": nutrition["protein"],
             "carbs": nutrition["carbs"], "fat": nutrition["fat"],
             "fiber": float(nutrition.get("fiber", 0.0) or 0.0),
             "serving_size": nutrition["serving_size"], "provenance": nutrition["provenance"],
             "estimated": True, "external": bool(nutrition.get("external", False))}
    if nutrition.get("external_source"):
        entry["external_source"] = nutrition["external_source"]
    return entry


def totals_of(nutritions):
    keys = ("calories", "protein", "carbs", "fat")
    return {k: round(sum(n[k] for n in nutritions), 1) for k in keys}


def fingerprint(entries):
    """Stable short hash of what is being logged: used to stop an LLM that calls
    add_meal_to_state twice with the same items in one run from logging them twice."""
    text = "|".join(f"{e['food']}:{e['quantity_str']}" for e in entries)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def fmt(n):
    """Number for messages: 234.0 -> '234', 18.35 -> '18.4'."""
    return f"{n:.1f}".rstrip("0").rstrip(".")
