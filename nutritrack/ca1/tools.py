"""
tools.py
--------
Tool functions for the Meal Calorie Helper agent.

These are the real Python functions that the agent calls during its
plan-act loop.  Each tool returns structured data (a dict) so the
agent can read and reason about the result before deciding the next step.

Tools:
    lookup_calories(query)  — look up estimated nutrition for a food
    add_meal(food, qty, ...)— log a meal and update session memory
    get_daily_summary()     — retrieve current consumed totals + macro split
    check_snack_fit(food)   — decide whether a snack fits the remaining budget

Safety note:
    All nutritional values are *estimated*.
    Actual values vary by brand, preparation, and serving size.
"""

from __future__ import annotations
import re
from typing import Optional

from .food_database import FOOD_DB, find_food, add_custom_food
from ..core.nutrition import resolve_query
from ..state.run_log import sanitize_error_text
from .memory import MealEntry, SessionMemory
from . import config


# ---------------------------------------------------------------------------
# Module-level session memory (shared across all tool calls in one session)
# ---------------------------------------------------------------------------
_memory = SessionMemory()

# Caches the most recent successful ONLINE (API) lookup, so the user can
# say "save this food" without having to retype its nutrition values.
_last_api_lookup: Optional[dict] = None


def reset_session() -> None:
    """Reset memory for a brand-new session (e.g., new notebook run)."""
    global _memory
    _memory = SessionMemory()


def get_memory() -> SessionMemory:
    """Return the current session memory (for use by the agent core)."""
    return _memory


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_quantity(query: str) -> tuple[str, float, str]:
    """
    Parse a natural-language food query into:
        (food_name_hint, quantity_value, quantity_type)

    Examples:
        "3 boiled eggs"         → ("boiled eggs", 3.0, "piece")
        "200 g chicken breast"  → ("chicken breast", 200.0, "g")
        "2 slices of bread"     → ("bread", 2.0, "slice")
        "1 cup of milk"         → ("milk", 1.0, "cup")
        "banana"                → ("banana", 1.0, "piece")   # default

    Returns (food_hint, qty, unit_type)
    """
    query = query.strip()

    # Pattern: <number> <unit> <food> or <number> <food>
    patterns = [
        # "200 g chicken breast", "200ml milk", "50gm rice" (with space or gm abbrev)
        r"^(\d+(?:\.\d+)?)\s*(g|gm|gram|grams|ml|millilitre|milliliter|kg)\s+(?:of\s+)?(.+)$",
        # No-space weight/volume: "250gm curd", "50g rice"
        r"^(\d+(?:\.\d+)?)(g|gm|gram|grams|ml|kg)(?:\s+(?:of\s+)?(.+))?$",
        # "2 slices of bread", "3 pieces of chicken"
        r"^(\d+(?:\.\d+)?)\s*(slice|slices|piece|pieces|cup|cups|scoop|scoops|tbsp|tsp|serving|servings)\s+(?:of\s+)?(.+)$",
        # "3 boiled eggs", "2 roti"
        r"^(\d+(?:\.\d+)?)\s+(.+)$",
    ]

    def _norm_weight_unit(u: str) -> str:
        """Normalise weight/volume unit strings."""
        u = u.lower()
        if u in ("gm", "gram", "grams"):
            return "g"
        if u in ("millilitre", "milliliter"):
            return "ml"
        return u

    for i, pattern in enumerate(patterns):
        m = re.match(pattern, query, re.IGNORECASE)
        if m:
            groups = m.groups()
            if i == 0:
                # Pattern 0: weight/volume with space
                qty = float(groups[0])
                unit = _norm_weight_unit(groups[1])
                food_hint = groups[2].strip()
                return food_hint, qty, unit
            elif i == 1:
                # Pattern 1: no-space "250gm curd" or bare "250g"
                qty = float(groups[0])
                unit = _norm_weight_unit(groups[1])
                food_hint = (groups[2] or "").strip()
                if not food_hint:
                    # e.g. bare "250g" with no food — return raw query
                    return query, 1.0, "piece"
                return food_hint, qty, unit
            elif i == 2:
                # Pattern 2: piece/slice/cup
                qty = float(groups[0])
                unit = groups[1].lower()
                if unit in ("slice", "slices"):
                    unit = "slice"
                elif unit in ("piece", "pieces"):
                    unit = "piece"
                elif unit in ("cup", "cups"):
                    unit = "cup"
                elif unit in ("scoop", "scoops"):
                    unit = "scoop"
                food_hint = groups[2].strip()
                return food_hint, qty, unit
            else:
                # Pattern 3: "3 boiled eggs"
                qty = float(groups[0])
                food_hint = groups[1].strip()
                return food_hint, qty, "piece"

    # No number found — default qty = 1, unit = piece
    return query, 1.0, "piece"


def _scale_entry(entry: dict, qty: float, unit: str, display_name: str) -> dict:
    """
    Core scaling math, operating on a raw nutrition entry dict (the same
    shape as a FOOD_DB value). Shared by:
        - _scale_nutrition() for local database lookups
        - lookup_calories()'s Open Food Facts branch, for API results that
          are always given per 100 g/ml and need scaling to match the
          quantity the user actually asked for.

    Handles conversions:
        - g/ml: scale linearly from base_amount
        - piece/slice: scale linearly from base_amount (which is 1 for these)
        - cup: convert to ml/g (1 cup ≈ 240 ml or 200 g depending on context)
        - scoop: treat as piece (base_amount = 1)
    """
    base_amount = entry["base_amount"]
    db_unit = entry["unit"]

    if unit in ("g", "ml", "piece", "slice", "scoop") and unit == db_unit:
        effective_qty = qty
    elif unit in ("g", "ml") and db_unit in ("g", "ml"):
        effective_qty = qty
    elif unit == "cup":
        if db_unit == "ml":
            effective_qty = qty * 240
        else:
            effective_qty = qty * 200
    elif unit == "scoop" and db_unit in ("piece", "g"):
        effective_qty = qty * base_amount
    elif unit == "piece" and db_unit == "g":
        if base_amount == 1:
            effective_qty = 100 * qty
        elif qty == 1:
            effective_qty = base_amount
        else:
            effective_qty = qty
    elif unit == "piece" and db_unit == "ml":
        effective_qty = 240 * qty
    elif unit == "slice" and db_unit in ("g", "ml"):
        effective_qty = 28 * qty
    else:
        effective_qty = qty * base_amount

    scale = effective_qty / base_amount

    return {
        "food": display_name,
        "serving_size": entry["serving_size"],
        "quantity_str": f"{qty} {unit}" if unit != "piece" else (
            f"{int(qty)} piece" if qty == 1 else f"{int(qty)} pieces"
        ),
        "calories": round(entry["calories_per_unit"] * scale, 1),
        "protein": round(entry["protein"] * scale, 1),
        "carbs": round(entry["carbs"] * scale, 1),
        "fat": round(entry["fat"] * scale, 1),
        "scale_factor": round(scale, 3),
    }


def _scale_nutrition(food_key: str, qty: float, unit: str) -> dict:
    """
    Scale nutrition values from the LOCAL database to the requested
    quantity. Thin wrapper around _scale_entry() for a known FOOD_DB key.
    """
    entry = FOOD_DB[food_key]
    return _scale_entry(entry, qty, unit, entry.get("canonical_name", food_key))


# ---------------------------------------------------------------------------
# TOOL 1: lookup_calories
# ---------------------------------------------------------------------------

def _fetch_from_openfoodfacts(query: str):
    """
    Query Open Food Facts — free, no API key required at all.

    Open Food Facts is a product/barcode database: it matches by product
    NAME (best for packaged/branded foods like "Pintola Muesli"), and
    always returns nutrition PER 100 g/ml — it does not parse quantities
    out of the query text the way CalorieNinjas does. The "needs_scaling"
    flag tells lookup_calories() to scale this result using the qty/unit
    already parsed from the user's original query.
    """
    import requests
    resp = requests.get(
        config.OFF_API_URL,
        params={
            "search_terms": query,
            "search_simple": 1,
            "action": "process",
            "json": 1,
            "page_size": 1,
        },
        timeout=config.API_TIMEOUT_SECONDS,
        headers={"User-Agent": "T17-MealCalorieHelper-StudentProject/1.0"},
    )
    resp.raise_for_status()
    data = resp.json()

    products = data.get("products", [])
    if not products:
        return None

    product = products[0]
    nutriments = product.get("nutriments", {})
    calories = nutriments.get("energy-kcal_100g")
    if calories is None:
        return None  # entry exists but has no usable calorie data

    name = (product.get("product_name") or query).strip().lower()
    return {
        "food": name,
        "serving_size": "100 g (Open Food Facts estimate)",
        "unit": "g",
        "base_amount": 100,
        "calories": round(float(calories), 1),
        "protein": round(float(nutriments.get("proteins_100g", 0)), 1),
        "carbs": round(float(nutriments.get("carbohydrates_100g", 0)), 1),
        "fat": round(float(nutriments.get("fat_100g", 0)), 1),
        "source": "api",
        "needs_scaling": True,
    }


def _fetch_from_calorieninjas(query: str):
    """
    Query CalorieNinjas / API Ninjas — needs a free API key (config.py).

    Unlike Open Food Facts, this API parses natural-language quantities
    itself ("3 boiled eggs" → totals for 3), so its result is already the
    final answer and needs no further scaling.
    """
    import requests
    resp = requests.get(
        config.CALORIE_API_URL,
        params={"query": query},
        headers={"X-Api-Key": config.CALORIE_API_KEY},
        timeout=config.API_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    data = resp.json()

    items = data.get("items", [])
    if not items:
        return None

    item = items[0]
    return {
        "food": (item.get("name") or query).strip().lower(),
        "serving_size": f"{item.get('serving_size_g', '?')} g (online estimate)",
        "calories": round(float(item.get("calories", 0)), 1),
        "protein": round(float(item.get("protein_g", 0)), 1),
        "carbs": round(float(item.get("carbohydrates_total_g", 0)), 1),
        "fat": round(float(item.get("fat_total_g", 0)), 1),
        "source": "api",
        "needs_scaling": False,
    }


def _fetch_from_api(query: str):
    """
    Online fallback used only when a food is not found in the local
    database AND the configured provider (config.API_PROVIDER) is usable.

    Returns:
        - dict with "source": "api" if a match was found. Includes
          "needs_scaling" to tell lookup_calories() whether this result
          is already final (CalorieNinjas) or needs local scaling to the
          user's requested quantity (Open Food Facts, always per 100g).
        - dict with "status": "api_error" if the request itself failed
          (no internet, bad/missing key, timeout, missing 'requests'
          package) — kept distinct from "food genuinely doesn't exist".
        - None if the provider responded fine but had no match.
    """
    if not config.has_api_key():
        return None
    from ..ai.config import is_offline_mode
    if is_offline_mode():
        return None

    try:
        if config.API_PROVIDER == "calorieninjas":
            return _fetch_from_calorieninjas(query)
        return _fetch_from_openfoodfacts(query)
    except Exception as exc:
        # Network error / timeout / bad key / missing 'requests' package —
        # never let this crash the agent or expose secrets in error strings.
        return {"status": "api_error", "error": sanitize_error_text(exc)}


def lookup_calories(query: str) -> dict:
    """
    Tool 1 — Look up estimated calories and macros for a food item.

    Lookup order:
        1. Local food_database.py (instant, offline, tried first).
        2. Online API fallback (config.py) — ONLY if not found locally
           AND an API key is configured.
        3. If neither has it, return "not_found" with can_add_manually=True
           so the agent can offer the "add custom food" path.

    Parameters:
        query (str): Natural language food description.
                     Examples: "3 boiled eggs", "200 g chicken breast",
                               "banana", "2 slices of bread"

    Returns:
        dict with keys:
            status       : "found" | "not_found"
            source       : "local_database" | "api" (only when found)
            food         : canonical food name (if found)
            quantity_str : parsed quantity (if found)
            serving_size : reference serving
            calories     : estimated kcal
            protein      : grams
            carbs        : grams
            fat          : grams
            note         : safety / informational note
            error        : error message (if not_found)
            can_add_manually : True (if not_found)
    """
    global _last_api_lookup

    # Phase 3: parsing, validation, matching and scaling now live in the shared
    # engine (core/nutrition.py), which asks instead of guessing (audit D1-D4).
    resolved = resolve_query(query)

    # ---- 1. Found locally --------------------------------------------
    if resolved["ok"]:
        result = {k: resolved[k] for k in (
            "food", "serving_size", "quantity_str", "calories", "protein", "carbs", "fat", "scale_factor")}
        result["status"] = "found"
        result["source"] = "local_database"
        result["provenance"] = resolved["provenance"]
        result["note"] = (
            "Estimated nutritional values; actual values can vary by brand, "
            "preparation method, and serving size."
        )
        return result

    # ---- 1b. Understood, but the amount or the match needs clarifying ---
    # Reported as "not_found" (the status CA1's callers already handle) plus a
    # machine-readable `reason`; the user is asked, nothing is guessed.
    if resolved["reason"] != "unknown_food":
        question = resolved.get("question")
        return {
            "status": "not_found",
            "query": query,
            "reason": resolved["reason"],
            "error": resolved["message"] + (f" {question}" if question else ""),
            "can_add_manually": resolved["reason"] == "low_confidence_match",
        }

    food_hint, qty, unit = _parse_quantity(query)   # needed by the online fallback below

    # ---- 2. Not local — try the online API fallback -------------------
    api_result = _fetch_from_api(query)

    if api_result is not None and api_result.get("status") == "api_error":
        return {
            "status": "not_found",
            "query": query,
            "reason": "unknown_food",
            "error": (
                f"'{query}' is not in the local database, and the online "
                f"lookup could not be reached ({api_result['error']}). "
                "You can add it yourself instead — see 'add custom food'."
            ),
            "can_add_manually": True,
        }

    if api_result is not None:
        note = (
            "Fetched from an ONLINE source (not your local database) — "
            "estimated values, may vary. Say 'save this food' to store it "
            "permanently for instant, offline lookups next time."
        )

        if api_result.get("needs_scaling"):
            # Open Food Facts (or similar): result is per-100g and must be
            # scaled to the qty/unit already parsed from the user's query.
            # Reuse the exact same scaling math as local database lookups
            # by wrapping the API values in the same entry shape.
            entry = {
                "serving_size": api_result["serving_size"],
                "unit": api_result["unit"],
                "base_amount": api_result["base_amount"],
                "calories_per_unit": api_result["calories"],
                "protein": api_result["protein"],
                "carbs": api_result["carbs"],
                "fat": api_result["fat"],
            }
            scaled = _scale_entry(entry, qty, unit, api_result["food"])
            scaled["status"] = "found"
            scaled["source"] = "api"
            scaled["note"] = note

            # Cache the per-100g values (not the scaled ones) so
            # 'save this food' stores a reusable, scalable entry.
            _last_api_lookup = {
                "food": api_result["food"],
                "calories": api_result["calories"],
                "protein": api_result["protein"],
                "carbs": api_result["carbs"],
                "fat": api_result["fat"],
                "serving_size": api_result["serving_size"],
                "unit": api_result["unit"],
                "base_amount": api_result["base_amount"],
            }
            return scaled

        # CalorieNinjas-style: API already parsed the quantity itself —
        # its result is the final answer, no further scaling needed.
        _last_api_lookup = dict(api_result)
        api_result["status"] = "found"
        api_result["quantity_str"] = query
        api_result["note"] = note
        return api_result

    # ---- 3. Neither local DB nor API had it ----------------------------
    return {
        "status": "not_found",
        "query": query,
        "reason": "unknown_food",
        "error": (
            f"'{query}' is not in the local database"
            + ("" if config.has_api_key()
               else " and the online lookup is switched off (see config.py)")
            + ". You can add it yourself — see 'add custom food'."
        ),
        "can_add_manually": True,
    }


# ---------------------------------------------------------------------------
# TOOL 2: add_meal
# ---------------------------------------------------------------------------

def add_meal(query: str) -> dict:
    """
    Tool 2 — Add a food item to today's meal log and update daily totals.

    Internally calls lookup_calories() to get nutrition data, then stores
    the result in session memory.

    Parameters:
        query (str): Same natural language food description as lookup_calories.

    Returns:
        dict with keys:
            status         : "added" | "not_found"
            meal_added     : dict with food details (if added)
            daily_totals   : dict with updated daily sums
            remaining_kcal : kcal remaining (None if no budget set)
            macro_split    : approximate macro % breakdown
            note           : safety / informational note
            error          : error message (if not_found)
    """
    lookup = lookup_calories(query)

    if lookup["status"] == "not_found":
        return {
            "status": "not_found",
            "query": query,
            "error": lookup["error"],
        }

    entry = MealEntry(
        food=lookup["food"],
        quantity_str=lookup["quantity_str"],
        calories=lookup["calories"],
        protein=lookup["protein"],
        carbs=lookup["carbs"],
        fat=lookup["fat"],
    )
    _memory.add_meal(entry)

    return {
        "status": "added",
        "meal_added": entry.to_dict(),
        "daily_totals": {
            "calories": _memory.total_calories,
            "protein": _memory.total_protein,
            "carbs": _memory.total_carbs,
            "fat": _memory.total_fat,
        },
        "remaining_kcal": _memory.remaining_calories(),
        "budget": _memory.daily_budget,
        "macro_split": _memory.macro_split(),
        "note": lookup["note"],
    }


# ---------------------------------------------------------------------------
# TOOL 3: get_daily_summary
# ---------------------------------------------------------------------------

def get_daily_summary() -> dict:
    """
    Tool 3 — Return a complete summary of today's nutrition and meals.

    Returns:
        dict with full session state including macro split.
    """
    summary = _memory.summary()
    summary["macro_split"] = _memory.macro_split()
    return summary


# ---------------------------------------------------------------------------
# TOOL 4: check_snack_fit
# ---------------------------------------------------------------------------

def check_snack_fit(query: str) -> dict:
    """
    Tool 4 — Check whether a planned snack fits the remaining calorie budget.

    Steps performed internally (visible to agent through return value):
        1. Call lookup_calories() to get snack's kcal.
        2. Read remaining_kcal from memory.
        3. Compare and decide.

    Parameters:
        query (str): Snack description e.g. "banana", "2 slices of bread"

    Returns:
        dict with keys:
            status          : "fits" | "does_not_fit" | "no_budget" | "not_found"
            snack_food      : canonical food name
            snack_calories  : kcal for the snack
            consumed_kcal   : kcal consumed so far today
            budget          : daily budget
            remaining_kcal  : kcal remaining before snack
            surplus_deficit : positive → snack fits with this much to spare
                              negative → snack exceeds budget by abs(value)
            decision        : human-readable yes/no/warning string
            macro_preview   : macros for the snack
            note            : safety note
    """
    if not _memory.has_budget():
        return {
            "status": "no_budget",
            "decision": (
                "No daily calorie budget has been set. "
                "Please set your budget first (e.g. 'My calorie budget is 2000 kcal')."
            ),
        }

    lookup = lookup_calories(query)
    if lookup["status"] == "not_found":
        return {
            "status": "not_found",
            "query": query,
            "error": lookup["error"],
        }

    snack_cal = lookup["calories"]
    remaining = _memory.remaining_calories()
    surplus_deficit = round(remaining - snack_cal, 1)

    if surplus_deficit >= 0:
        fits = True
        decision = (
            f"✅ YES — {lookup['food'].title()} ({lookup['quantity_str']}) "
            f"is approximately {snack_cal} kcal. "
            f"You have {remaining} kcal remaining, so it fits your budget "
            f"with {surplus_deficit} kcal to spare."
        )
        status = "fits"
    else:
        fits = False
        decision = (
            f"❌ NO — {lookup['food'].title()} ({lookup['quantity_str']}) "
            f"is approximately {snack_cal} kcal. "
            f"You only have {remaining} kcal remaining, so you would exceed "
            f"your budget by {abs(surplus_deficit)} kcal."
        )
        status = "does_not_fit"

    return {
        "status": status,
        "fits": fits,
        "snack_food": lookup["food"],
        "snack_calories": snack_cal,
        "snack_protein": lookup["protein"],
        "snack_carbs": lookup["carbs"],
        "snack_fat": lookup["fat"],
        "quantity_str": lookup["quantity_str"],
        "consumed_kcal": _memory.total_calories,
        "budget": _memory.daily_budget,
        "remaining_kcal": remaining,
        "surplus_deficit": surplus_deficit,
        "decision": decision,
        "macro_preview": {
            "protein": lookup["protein"],
            "carbs": lookup["carbs"],
            "fat": lookup["fat"],
        },
        "note": lookup.get("note", ""),
    }


# ---------------------------------------------------------------------------
# TOOL 5: set_budget
# ---------------------------------------------------------------------------

def set_budget(budget_kcal: float) -> dict:
    """
    Tool 5 — Set the daily calorie budget in session memory.

    Parameters:
        budget_kcal (float): Target daily calories.

    Returns:
        dict confirming budget was set.
    """
    if budget_kcal <= 0:
        return {
            "status": "error",
            "error": "Budget must be a positive number of calories.",
        }

    _memory.set_budget(budget_kcal)
    return {
        "status": "set",
        "daily_budget": _memory.daily_budget,
        "message": (
            f"Daily calorie budget set to {budget_kcal:.0f} kcal. "
            "You can now start logging meals."
        ),
    }


# ---------------------------------------------------------------------------
# TOOL 6: add_custom_food_tool
# ---------------------------------------------------------------------------

def add_custom_food_tool(
    name: str,
    calories: float,
    protein: float,
    carbs: float,
    fat: float,
    serving_size: str = "1 serving",
    unit: str = "piece",
    base_amount: float = 1,
) -> dict:
    """
    Tool 6 — Let the user permanently add their OWN food to the database
    with manually supplied nutrition info (e.g. a homemade dish like
    "mom's rajma" that will never be in any generic database, or any food
    the online API also failed to find).

    Parameters:
        name         : food name to store it under (becomes the lookup key)
        calories     : kcal for one `base_amount` `unit` serving
        protein/carbs/fat : grams for the same serving
        serving_size : human-readable description, e.g. "1 bowl (250 g)"
        unit         : "piece" | "g" | "ml" — what the numbers above are FOR.
                       Use "g"/"ml" (with base_amount) for packaged foods
                       given per weight/volume, e.g. "per 100 g".
                       Use "piece" (base_amount=1) for one-off dishes like
                       "1 bowl of rajma" that aren't scaled by weight.
        base_amount  : how many `unit`s the numbers above are FOR
                       (e.g. unit="g", base_amount=100 -> values are per 100 g)

    Returns:
        dict with keys:
            status   : "added_to_database" | "error"
            food_key : the canonical key it was stored under
            message  : confirmation text
            error    : error message (if invalid input)
    """
    if calories < 0 or protein < 0 or carbs < 0 or fat < 0:
        return {
            "status": "error",
            "error": "Nutrition values cannot be negative.",
        }
    if not name or not name.strip():
        return {
            "status": "error",
            "error": "Please provide a name for the food.",
        }

    key = add_custom_food(
        name=name,
        calories=calories,
        protein=protein,
        carbs=carbs,
        fat=fat,
        serving_size=serving_size,
        unit=unit,
        base_amount=base_amount,
    )
    return {
        "status": "added_to_database",
        "food_key": key,
        "message": (
            f"'{name.strip()}' has been saved to your local food database "
            f"({calories:.0f} kcal, P:{protein:.0f}g C:{carbs:.0f}g F:{fat:.0f}g "
            f"per {serving_size}). You can log it any time from now on, "
            "instantly and offline."
        ),
    }


# ---------------------------------------------------------------------------
# TOOL 7: save_last_lookup
# ---------------------------------------------------------------------------

def save_last_lookup() -> dict:
    """
    Tool 7 — Permanently save the most recent ONLINE (API) lookup result
    into the local food database, so future lookups for that food don't
    need internet access at all.

    Works together with the "found via API" path of lookup_calories():
    every time an API lookup succeeds, its result is cached in
    _last_api_lookup. Calling this tool persists that cached result via
    add_custom_food() and clears the cache.

    Returns:
        dict with keys:
            status   : "saved" | "error"
            food_key : the canonical key it was stored under (if saved)
            error    : error message (if nothing to save)
    """
    global _last_api_lookup

    if _last_api_lookup is None:
        return {
            "status": "error",
            "error": (
                "No recent online lookup to save. Look up a food first "
                "(one that isn't in the local database) and then say "
                "'save this food'."
            ),
        }

    cached = _last_api_lookup
    key = add_custom_food(
        name=cached["food"],
        calories=cached["calories"],
        protein=cached["protein"],
        carbs=cached["carbs"],
        fat=cached["fat"],
        serving_size=cached["serving_size"],
        unit=cached.get("unit", "piece"),
        base_amount=cached.get("base_amount", 1),
    )
    _last_api_lookup = None
    return {
        "status": "saved",
        "food_key": key,
        "message": (
            f"'{key}' has been saved to your local food database. "
            "Future lookups for it will work instantly, offline."
        ),
    }


# ---------------------------------------------------------------------------
# TOOL 8: delete_meal
# ---------------------------------------------------------------------------

def delete_meal(index: int | None = None, food_name: str | None = None) -> dict:
    """
    Tool 8 — Remove a wrongly-logged meal from today's log and recompute
    daily totals + macro split. Useful for fixing mistakes such as a food
    matcher bug incorrectly logging the wrong item.

    Provide EXACTLY ONE of:
        index     : 1-based position in the meal list, as shown by
                    get_daily_summary() (e.g. 2 for the 2nd meal logged today)
        food_name : a name to match against today's logged meals — removes
                    the MOST RECENT matching entry if there are several

    Returns:
        dict with keys:
            status         : "deleted" | "not_found" | "error"
            removed        : dict of the removed meal (if deleted)
            daily_totals   : updated totals after removal
            remaining_kcal : kcal remaining (None if no budget set)
            macro_split    : updated macro breakdown
            error          : error message (if not_found/error)
    """
    if index is None and not food_name:
        return {
            "status": "error",
            "error": "Specify either a meal number or a food name to delete.",
        }

    if index is not None:
        removed = _memory.remove_meal(index)
        if removed is None:
            return {
                "status": "not_found",
                "error": (
                    f"No meal at position {index}. Say 'show today's summary' "
                    "to see the current numbered list."
                ),
            }
    else:
        removed = _memory.remove_meal_by_food(food_name)
        if removed is None:
            return {
                "status": "not_found",
                "error": f"No logged meal matching '{food_name}' found today.",
            }

    return {
        "status": "deleted",
        "removed": removed.to_dict(),
        "daily_totals": {
            "calories": _memory.total_calories,
            "protein": _memory.total_protein,
            "carbs": _memory.total_carbs,
            "fat": _memory.total_fat,
        },
        "remaining_kcal": _memory.remaining_calories(),
        "macro_split": _memory.macro_split(),
    }


# ---------------------------------------------------------------------------
# TOOL 9: get_daily_history
# ---------------------------------------------------------------------------

def get_daily_history(limit: int = 7) -> dict:
    """
    Tool 9 — Return the saved history of past (fully-closed) days, most
    recent first. Today's in-progress log is NOT included — use
    get_daily_summary() for that.

    Parameters:
        limit : how many past days to return (default 7)

    Returns:
        dict mapping "YYYY-MM-DD" -> {budget, meals, total_calories,
        total_protein, total_carbs, total_fat}
    """
    from . import daily_store
    return daily_store.get_history(limit=limit)
