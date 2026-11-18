"""
nutrition.py
------------
Resolve "what did the user eat, and how much?" into scaled nutrition values
from the LOCAL food database (built-in foods + the user's custom foods).

This is the single place where a quantity is converted and a food's nutrition
is scaled. It is used by ca1.tools.lookup_calories (CA1 rule-based path) and by
every CA2 agent tool, so both always give the same answer.

Design rule: if the amount or the food is not clear, return a failure with a
question for the user. Never guess a unit, never convert with an invented
factor, never accept a low-confidence match. (CA1 audit defects D1-D4.)

resolve_item / resolve_query return a dict:
    success: {"ok": True,  "food", "serving_size", "quantity_str", "calories",
              "protein", "carbs", "fat", "scale_factor", "provenance", "notes"}
    failure: {"ok": False, "reason", "message", "question"?, ...}
    reasons: unknown_food, low_confidence_match, invalid_quantity,
             implausible_quantity, ambiguous_quantity, missing_quantity,
             unit_mismatch, unsupported_unit, no_food_name
"""
import difflib

from ..ca1.food_database import FOOD_DB, add_custom_food, get_food_db, match_details
from .external_nutrition import external_enabled, lookup_external_food
from .quantity import (COUNT_UNITS, TO_BASE, UNIT_ALIASES, parse_item_text,
                       plausibility_problem)

# For foods stored per piece/slice/scoop: which user units mean "one of these".
_ACCEPTS = {
    "piece": {"piece", "serving"},
    "slice": {"slice", "piece", "serving"},
    "scoop": {"scoop", "serving", "piece"},
}

# Approximate gram weights per piece/slice/scoop for Quick Add measurement conversions
_DEFAULT_UNIT_GRAMS = {
    "egg": 50.0,
    "bread": 28.0,
    "roti": 40.0,
    "apple": 182.0,
    "banana": 118.0,
    "orange": 131.0,
    "mango": 200.0,
    "potato": 173.0,
    "sweet potato": 114.0,
    "carrot": 61.0,
    "tomato": 123.0,
    "onion": 110.0,
    "whey protein": 30.0,
}

_DIARY_MEASURE_TO_GRAMS = {
    "g": 1.0,
    "kg": 1000.0,
    "ml": 1.0,
    "l": 1000.0,
    "cup": 200.0,
    "bowl": 200.0,
    "tbsp": 15.0,
    "tsp": 5.0,
    "piece": 50.0,
    "slice": 30.0,
    "scoop": 30.0,
}


def _fail(reason, message, **extra):
    out = {"ok": False, "reason": reason, "message": message}
    out.update(extra)
    return out


def did_you_mean(text, limit=3, *, user_id=None):
    """Closest known food names for an unknown food. A SUGGESTION to show the
    user -- it is never applied automatically."""
    db = get_food_db(user_id)
    names = {}
    for key, entry in db.items():
        for alias in [key, *entry.get("aliases", [])]:
            names.setdefault(alias.lower(), key)
    close = difflib.get_close_matches(text.lower().strip(), list(names), n=limit * 2, cutoff=0.6)
    seen, out = set(), []
    for alias in close:
        key = names[alias]
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out[:limit]


def format_quantity(value, unit):
    """CA1's display format ("3 pieces", "50.0 g"), with fractional pieces fixed."""
    if unit == "piece":
        if value == 1:
            return "1 piece"
        return f"{int(value)} pieces" if float(value).is_integer() else f"{value:g} pieces"
    return f"{float(value)} {unit}"


def _convert(entry, name, value, unit):
    """-> (effective amount in the food's own unit, display unit, note) or a failure dict."""
    native = entry["unit"]

    portions = entry.get("portions") or {}
    if native in ("g", "ml"):
        if unit in portions:                                 # household unit with a known weight (e.g. 1 cup)
            return (1.0 if value is None else value) * portions[unit], unit, None
        if unit is None and ("piece" in portions or "serving" in portions):
            eff_u = "piece" if "piece" in portions else "serving"
            return (1.0 if value is None else value) * portions[eff_u], eff_u, None

    if native in _ACCEPTS:                                   # piece / slice / scoop foods
        if unit is None:
            return (1.0 if value is None else value), native, None
        if unit in _ACCEPTS[native]:
            return value, unit, None
        if unit in TO_BASE:
            return _fail("unit_mismatch",
                         f"{name} is counted in {native}s, not by weight or volume, so I can't turn "
                         f"{value:g} {unit} into a number of {native}s without guessing.",
                         question=f"How many {native}s of {name} did you have?")
        return _fail("unit_mismatch",
                     f"'{unit}' doesn't fit {name}, which is counted in {native}s.",
                     question=f"How many {native}s of {name} did you have?")

    # weight- or volume-based foods
    measure, example = ("grams", "150 g") if native == "g" else ("ml", "250 ml")
    if unit is None:
        if value is None:
            return _fail("missing_quantity", f"I need an amount for {name}.",
                         question=f"How much {name}? Please give {measure}, e.g. '{example} {name}'.")
        return _fail("ambiguous_quantity",
                     f"'{value:g}' has no unit, so I can't tell whether it means grams, ml or pieces of {name}.",
                     question=f"Please include the unit, e.g. '{value:g} {native} {name}'.")
    if native == "g":
        if unit == "serving":
            return (1.0 if value is None else value) * entry["base_amount"], "serving", None
        if unit in ("g", "kg"):
            return value * TO_BASE[unit], unit, None
        return _fail("unit_mismatch",
                     f"{name} is measured by weight, and I don't know how many grams a {unit} of it is.",
                     question=f"How many grams of {name} was that? e.g. '150 g {name}'.")
    # native == "ml"
    if unit == "serving":
        return (1.0 if value is None else value) * entry["base_amount"], "serving", None
    if unit in ("ml", "l", "cup", "tbsp", "tsp"):
        return value * TO_BASE[unit], unit, None
    if unit in ("g", "kg"):
        return value * TO_BASE[unit], unit, "Treated 1 g as 1 ml (an approximation for liquids)."
    return _fail("unit_mismatch",
                 f"{name} is a liquid measured in ml or cups, not in {unit}s.",
                 question=f"How many ml (or cups) of {name} was that?")


def resolve_item(food_text, quantity=None, unit=None, household_unit=None, allow_external=None,
                 register_external=False, user_id=None):
    """
    Resolve a food name plus an optional quantity/unit.

    `unit` may be any alias ("grams", "pcs", ...) or None. Order of checks:
    quantity sanity -> food match (local DB scoped to `user_id`; then opt-in public nutrition API when
    NUTRITRACK_ONLINE_LOOKUP=1 and allow_external is True and mode is not offline) -> unit conversion.
    External lookups are NEVER silently saved to the local database.
    """
    food_text = (food_text or "").strip()
    if not food_text:
        return _fail("no_food_name", "I couldn't find a food name.",
                     question="Which food did you have?")

    if unit is not None:
        canonical_unit = UNIT_ALIASES.get(str(unit).strip().lower())
        if canonical_unit is None:
            return _fail("unsupported_unit", f"I don't recognise the unit '{unit}'.",
                         question="Please give the amount in grams, ml, pieces, slices, cups or scoops.")
        unit = canonical_unit
    if quantity is not None:
        try:
            quantity = float(quantity)
        except (TypeError, ValueError):
            return _fail("invalid_quantity", f"'{quantity}' is not a number.")
        if quantity != quantity or quantity in (float("inf"), float("-inf")):
            return _fail("invalid_quantity", "The quantity must be a real number.")
        problem = plausibility_problem(quantity, unit)
        if problem:
            return _fail(problem["code"], problem["message"], question="What was the amount?")

    db = get_food_db(user_id)
    details = match_details(food_text, user_id=user_id)
    key = details["key"]
    external_note = None
    external_source = None
    entry = None
    if allow_external is None:
        allow_external = external_enabled()
    needs_lookup = bool(allow_external) and external_enabled() and (key is None or details["leftover_words"])
    if needs_lookup:
        found = lookup_external_food(food_text)
        if found:
            external_source = found.get("source") or "external public database"
            external_note = (
                f"External source ({external_source}, provenance: {found['provenance']}): "
                f"values for '{found['name']}' per 100 g were fetched online and not saved locally."
            )
            serving_g = found["portions"].get("serving") or 100.0
            portions = dict(found["portions"])
            portions.setdefault("serving", serving_g)
            key = " ".join(food_text.lower().split())
            entry = {"canonical_name": key, "unit": "g", "base_amount": 100.0,
                     "serving_size": f"{serving_g:g} g serving", "calories_per_unit": found["calories"],
                     "protein": found["protein"], "carbs": found["carbs"], "fat": found["fat"],
                     "fiber": float(found.get("fiber", 0.0) or 0.0),
                     "provenance": found["provenance"], "portions": portions}
            details = {"key": key, "leftover_words": [], "blocked_by": None}
    if key is None:
        suggestions = did_you_mean(food_text, user_id=user_id)
        msg = f"'{food_text}' is not in the nutrition database, so I won't guess its values."
        if details["blocked_by"]:
            msg = (f"'{details['blocked_by']}' is a different food from the similarly named ones in the "
                   "database, so I won't guess its values.")
        return _fail("unknown_food", msg, food=food_text, did_you_mean=suggestions,
                     question=f"Please tell me the calories, protein, carbs and fat of '{food_text}' "
                              "(per serving or per 100 g), and I'll save it as your own food.")
    if details["leftover_words"]:
        extra = ", ".join(details["leftover_words"])
        return _fail("low_confidence_match",
                     f"'{food_text}' only partly matches '{key}': I don't recognise '{extra}'. "
                     "A branded or specially prepared food can differ a lot from the plain one.",
                     food=food_text, closest_match=key,
                     question=f"Did you mean plain {key}? If it is a different product, give me its label "
                              "values (calories, protein, carbs, fat) and I'll save it.")

    if entry is None:
        entry = db[key]
    name = entry.get("canonical_name", key)

    if household_unit:
        portions = entry.get("portions") or {}
        if household_unit in portions:
            unit = household_unit
            household_unit = None
        else:
            return _fail("unsupported_unit",
                         f"I don't know how big a '{household_unit}' of {name} is.",
                         question=f"About how many grams or ml of {name} was that? e.g. '250 ml {name}'.")

    converted = _convert(entry, name, quantity, unit)
    if isinstance(converted, dict):
        converted["food"] = key
        return converted
    effective, display_unit, note = converted

    over = plausibility_problem(effective, entry["unit"] if entry["unit"] in ("g", "ml") else "piece")
    if over:
        return _fail(over["code"], over["message"], question="What was the amount?")

    scale = effective / entry["base_amount"]
    shown_value = 1.0 if quantity is None else quantity
    notes = ["Estimated nutritional values; actual values vary by brand, preparation and serving size."]
    if external_note:
        notes.insert(0, external_note)
    if note:
        notes.append(note)
    return {
        "ok": True,
        "food": key,
        "serving_size": entry["serving_size"],
        "quantity_str": format_quantity(shown_value, display_unit),
        "calories": round(entry["calories_per_unit"] * scale, 1),
        "protein": round(entry["protein"] * scale, 1),
        "carbs": round(entry["carbs"] * scale, 1),
        "fat": round(entry["fat"] * scale, 1),
        "fiber": round(float(entry.get("fiber", 0.0) or 0.0) * scale, 1),
        "scale_factor": round(scale, 3),
        "provenance": entry.get("provenance", "unknown"),
        "external": bool(external_source),
        "external_source": external_source,
        "notes": notes,
    }


def resolve_query(text, *, user_id=None):
    """Parse a single food phrase ("200 g chicken breast") and resolve it."""
    parsed = parse_item_text(text)
    if parsed.problems:
        first = parsed.problems[0]
        return _fail(first["code"], first["message"], question="What was the amount?")
    return resolve_item(parsed.food_text, parsed.value, parsed.unit, parsed.household_unit, user_id=user_id)


def coerce_diary_portion(food_text, portion_text, *, user_id=None):
    """
    Convert any measurement unit selected in the Quick Add Food to Diary UI
    (serving, piece, g, kg, ml, l, cup, bowl, tbsp, tsp, slice, scoop) into
    a portion string compatible with `food_text`'s native unit when needed.
    Returns (effective_portion_str, conversion_note_or_None).
    """
    portion_clean = (portion_text or "").strip()
    if not portion_clean or not (food_text or "").strip():
        return portion_clean, None

    combined = f"{portion_clean} {food_text}".strip()
    parsed = parse_item_text(combined)
    if parsed.problems:
        return portion_clean, None

    qty = 1.0 if parsed.value is None else float(parsed.value)
    req_unit = parsed.unit or parsed.household_unit
    if req_unit is not None:
        req_unit = UNIT_ALIASES.get(req_unit.lower(), req_unit.lower())
    if req_unit in ("bowls", "katori", "katoris"):
        req_unit = "bowl"

    details = match_details(parsed.food_text or food_text, user_id=user_id)
    key = details.get("key")
    if not key or details.get("leftover_words"):
        # For unknown/external foods, only translate household 'bowl' to 200 g if needed
        if req_unit == "bowl":
            grams = round(qty * 200.0, 1)
            return f"{grams:g} g", f"1 bowl ≈ 200 g ({qty:g} bowl → {grams:g} g)."
        return portion_clean, None

    entry = get_food_db(user_id).get(key)
    if not entry:
        return portion_clean, None

    native = entry["unit"]
    portions = entry.get("portions") or {}
    if req_unit in portions:
        return portion_clean, None

    # 1. Native is piece / slice / scoop, user picked weight or volume in Quick Add
    if native in _ACCEPTS:
        if req_unit is None or req_unit in _ACCEPTS[native]:
            return portion_clean, None
        factor_g = _DIARY_MEASURE_TO_GRAMS.get(req_unit)
        if factor_g is not None:
            total_g = qty * factor_g
            unit_g = _DEFAULT_UNIT_GRAMS.get(key, 50.0)
            count = round(max(0.1, total_g / unit_g), 2)
            return (
                f"{count:g} {native}",
                f"Converted {qty:g} {req_unit} ≈ {count:g} {native}(s) (~{unit_g:g} g per {native}).",
            )

    # 2. Native is grams (g), user picked cup / bowl / tbsp / tsp / piece / slice / scoop / ml / l
    elif native == "g":
        if req_unit in (None, "g", "kg", "serving"):
            return portion_clean, None
        factor_g = _DIARY_MEASURE_TO_GRAMS.get(req_unit)
        if factor_g is not None:
            total_g = round(qty * factor_g, 1)
            return (
                f"{total_g:g} g",
                f"Converted {qty:g} {req_unit} ≈ {total_g:g} g.",
            )

    # 3. Native is millilitres (ml), user picked bowl / piece / slice / scoop
    elif native == "ml":
        if req_unit in (None, "ml", "l", "cup", "tbsp", "tsp", "g", "kg", "serving"):
            return portion_clean, None
        factor_ml = _DIARY_MEASURE_TO_GRAMS.get(req_unit)
        if factor_ml is not None:
            total_ml = round(qty * factor_ml, 1)
            return (
                f"{total_ml:g} ml",
                f"Converted {qty:g} {req_unit} ≈ {total_ml:g} ml.",
            )

    return portion_clean, None
