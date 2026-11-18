"""
Meal Intake Agent tools.

    parse_meal_entry        text -> food items with quantities (no nutrition yet)
    lookup_food_nutrition   one food + amount -> nutrition from the local database
    add_meal_to_state       items -> recomputed nutrition -> saved in the shared state
    remove_meal_from_state  delete a logged meal
    register_custom_food    save nutrition values the USER supplied for a food we don't know

Rule that protects the numbers: add_meal_to_state accepts only {food, quantity,
unit}. It recomputes the nutrition itself, so a language model can never
alter calories or macros on their way into the state.
"""
import math
import re
from typing import Optional

from ..ca1 import food_database
from ..core.nutrition import resolve_item
from ..core.quantity import is_non_food_request, parse_item_text, split_meal_text_with_meta
from ..state import StateError
from .base import agent_tool, fail, ok
from .common import (fingerprint, fmt, normalise_items, resolve_items, to_state_entry, totals_of)

AGENT = "meal_intake"
CUSTOM_UNITS = ("piece", "slice", "scoop", "g", "ml")


@agent_tool(AGENT)
def parse_meal_entry(ctx, text: str) -> dict:
    """Split a sentence such as "I ate 3 boiled eggs and 2 bananas for breakfast" into food items.

    Understands digits, number words ("two", "a dozen", "half"), units (g, ml, cup, slice, piece, scoop),
    and trailing quantities ("paneer 100g"). It does NOT look up nutrition: use lookup_food_nutrition or
    add_meal_to_state for that. Parts of the message that are questions rather than foods are returned in
    ignored_phrases so they can be handled by another specialist.

    Args:
        text: the user's message about what they ate.

    Returns:
        items (each with food, quantity, unit, household_unit, quantity_explicit, problems), meal_type,
        date_hint ("yesterday" if the user said so), ignored_phrases, and any problems found.
    """
    if not isinstance(text, str) or not text.strip():
        return fail("empty_input", "There is no text to read.", needs_user_input=True,
                    question="What did you eat, and how much?")
    if len(text) > 1000:
        return fail("input_too_long", "That message is too long to read safely (limit 1000 characters).",
                    needs_user_input=True)
    meal_type, date_hint, invalid_date_err, phrases = split_meal_text_with_meta(text)
    if invalid_date_err:
        return fail("invalid_date", invalid_date_err, needs_user_input=True,
                    question="Please check that date (use YYYY-MM-DD, 'today', or 'yesterday').")
    foods = [p for p in phrases if not is_non_food_request(p)]
    ignored = [p for p in phrases if is_non_food_request(p)]
    items = [parse_item_text(p).as_dict() for p in foods]
    if not items:
        return fail("no_food_found", "I couldn't find any food in that message.", needs_user_input=True,
                    question="What did you eat, and how much? For example '2 boiled eggs and a banana'.",
                    meal_type=meal_type, ignored_phrases=ignored)
    problems = [{"item": i["raw"], **p} for i in items for p in i["problems"]]
    return ok(f"Found {len(items)} food item(s)" + (f" for {meal_type}" if meal_type else "") + ".",
              items=items, meal_type=meal_type, date_hint=date_hint, ignored_phrases=ignored,
              has_problems=bool(problems), problems=problems)


@agent_tool(AGENT)
def lookup_food_nutrition(ctx, food: str, quantity: Optional[float] = None, unit: Optional[str] = None,
                          household_unit: Optional[str] = None) -> dict:
    """Look up estimated nutrition for ONE food and amount in the local food database. Read-only.

    Never invents values: an unknown food, an unclear amount (e.g. "2 curd" with no unit), an impossible
    unit conversion (e.g. "200 g banana") or a partial match (e.g. a brand name) comes back as a failure
    with a question to ask the user.

    Args:
        food: food name only, e.g. "boiled egg" (no quantity in it).
        quantity: the amount, e.g. 3. Leave empty for "a banana" style requests.
        unit: g, kg, ml, l, cup, tbsp, tsp, piece, slice, scoop or serving. Leave empty for countable foods.
        household_unit: a measure we cannot size (glass, bowl, plate...) copied from parse_meal_entry.
    """
    res = resolve_item(food, quantity, unit, household_unit, user_id=ctx.user_id)
    if not res["ok"]:
        return fail(res["reason"], res["message"], needs_user_input=True, question=res.get("question"),
                    food=food, did_you_mean=res.get("did_you_mean", []), closest_match=res.get("closest_match"))
    msg = f"{res['quantity_str']} of {res['food']}: {fmt(res['calories'])} kcal, {fmt(res['protein'])} g protein."
    if res.get("external"):
        msg += f" [External source: {res.get('external_source')} ({res['provenance']}); not saved to local DB]"
    return ok(msg,
              food=res["food"], quantity_str=res["quantity_str"], serving_size=res["serving_size"],
              calories=res["calories"], protein=res["protein"], carbs=res["carbs"], fat=res["fat"],
              provenance=res["provenance"], external=bool(res.get("external", False)),
              external_source=res.get("external_source"), estimated=True, notes=res["notes"])


@agent_tool(AGENT)
def add_meal_to_state(ctx, items: list[dict], meal_type: Optional[str] = None,
                      request_id: Optional[str] = None, date: Optional[str] = None) -> dict:
    """Save foods the user ate into the shared nutrition state.

    Pass ONLY food names and amounts. Nutrition is recomputed here from the food database, never taken from
    the caller. Foods that cannot be resolved are NOT saved and are reported in `unresolved` with a question
    for the user; the resolved foods are still saved. Calling again with the same items in the same run does
    not create duplicates.

    Args:
        items: list of {"food": "egg", "quantity": 3, "unit": null}. Use the values from parse_meal_entry.
        meal_type: breakfast, brunch, lunch, dinner or snack (optional).
        request_id: optional unique id; replaying the same id changes nothing.
        date: "today" (default), "yesterday" or YYYY-MM-DD for an earlier day.
    """
    resolved, failures = resolve_items(normalise_items(items), user_id=ctx.user_id)
    if not resolved:
        first = failures[0]
        msg = first["message"] if len(failures) == 1 else f"None of the {len(failures)} foods could be saved."
        return fail(first["code"] if len(failures) == 1 else "no_items_resolved", msg, needs_user_input=True,
                    question=first["question"], unresolved=failures)
    entries = [to_state_entry(r["nutrition"]) for r in resolved]
    rid = request_id or f"{ctx.run_id}:{fingerprint(entries)}"
    saved = ctx.state.add_meals(entries, request_id=rid, meal_type=meal_type, day=date, source="meal_intake")
    totals = ctx.state.day_totals(saved["date"])
    remaining = ctx.state.remaining(saved["date"])
    base = dict(added=saved["added"], duplicate=saved["duplicate"], unresolved=failures, date=saved["date"],
                batch_id=saved.get("batch_id"), meal_ids=saved["meal_ids"],
                daily_totals={k: totals[k] for k in ("calories", "protein", "carbs", "fat", "fiber", "meal_count")},
                remaining=remaining, needs_user_input=bool(failures))
    if saved["duplicate"]:
        return ok("These foods were already saved in this run, so nothing was added again.",
                  code="duplicate_ignored", **base)
    added_kcal = round(sum(e["calories"] for e in saved["added"]), 1)
    ext_sources = sorted({e.get("external_source") or e.get("provenance") for e in saved["added"] if e.get("external")})
    base["has_external"] = bool(ext_sources)
    message = f"Saved {len(entries)} food(s), {fmt(added_kcal)} kcal. Today's total: {fmt(totals['calories'])} kcal."
    if ext_sources:
        message += f" [External lookup ({', '.join(ext_sources)}); not saved to local food DB]."
    if failures:
        names = ", ".join(f"'{f['food']}'" for f in failures)
        return ok(message + f" NOT saved (need your input): {names}.", code="partial", **base)
    return ok(message, **base)


@agent_tool(AGENT)
def remove_meal_from_state(ctx, meal_id: Optional[str] = None, food_name: Optional[str] = None,
                           batch_id: Optional[str] = None, date: Optional[str] = None,
                           remove_last: bool = False) -> dict:
    """Delete logged food. Give ONE of: meal_id (one entry), batch_id (everything logged together),
    food_name (the most recent entry on that day whose name contains it), or remove_last=True.

    Args:
        meal_id: id of a single entry, e.g. "m-1a2b3c4d".
        food_name: e.g. "banana". Removes the most recent match only.
        batch_id: removes all foods that were logged in the same request.
        date: day to search for food_name: "today" (default), "yesterday" or YYYY-MM-DD.
        remove_last: set True to remove the most recently logged meal.
    """
    result = ctx.state.remove_meal(
        meal_id=meal_id, batch_id=batch_id, food_name=food_name, day=date, remove_last=bool(remove_last)
    )
    if not result["removed"]:
        return fail("not_found", "I couldn't find a matching meal to remove.", needs_user_input=True,
                    question="Which meal should I remove? You can name the food or give its id.")
    day = result["dates"][0]
    totals = ctx.state.day_totals(day)
    gone = ", ".join(f"{m['quantity_str']} {m['food']}" for m in result["removed"])
    message = f"Removed {gone}. Total for {day}: {fmt(totals['calories'])} kcal."
    if result["remaining_matches"]:
        message += f" {result['remaining_matches']} other matching entr{'y' if result['remaining_matches'] == 1 else 'ies'} kept."
    return ok(message, removed=result["removed"], dates=result["dates"], remaining_matches=result["remaining_matches"],
              daily_totals={k: totals[k] for k in ("calories", "protein", "carbs", "fat", "fiber", "meal_count")})


@agent_tool(AGENT)
def register_custom_food(ctx, name: str, calories: float, protein: float, carbs: float, fat: float,
                         serving_size: str = "1 serving", unit: str = "piece", base_amount: float = 1,
                         replace: bool = False, fiber: float = 0.0) -> dict:
    """Save a food using nutrition values that the USER typed in (never values you estimated).

    Use this only after lookup_food_nutrition reported an unknown food and the user gave the numbers.
    The values describe `base_amount` of `unit`: e.g. an idli with 39 kcal each is unit "piece",
    base_amount 1; a sauce with 90 kcal per 100 g is unit "g", base_amount 100.

    Args:
        name: the food's name.
        calories, protein, carbs, fat: values for base_amount of unit (protein/carbs/fat in grams).
        serving_size: short description, e.g. "1 idli (~40 g)".
        unit: piece, slice, scoop, g or ml.
        base_amount: how many `unit`s the values are for.
        replace: must be true to overwrite a food that already exists.
    """
    clean = " ".join(str(name).split())
    if not 1 <= len(clean) <= 60 or not re.search(r"[A-Za-z]", clean):
        return fail("invalid_input", "The food needs a name of 1-60 characters that contains letters.",
                    needs_user_input=True)
    if unit not in CUSTOM_UNITS:
        return fail("invalid_input", f"unit must be one of {', '.join(CUSTOM_UNITS)}.", needs_user_input=True)
    values = {"calories": calories, "protein": protein, "carbs": carbs, "fat": fat, "fiber": fiber, "base_amount": base_amount}
    for label, v in values.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            return fail("invalid_input", f"{label} must be a number that is not negative.", needs_user_input=True)
    if base_amount <= 0 or base_amount > 5000:
        return fail("invalid_input", "base_amount must be greater than 0 and at most 5000.", needs_user_input=True)
    by_weight = unit in ("g", "ml")
    if (by_weight and calories > 10 * base_amount) or (not by_weight and calories > 5000) \
            or any(x > (base_amount if by_weight else 500) for x in (protein, carbs, fat)):
        return fail("implausible_values", "Those values look impossible for that amount (more than 10 kcal per gram, "
                    "or more grams of a nutrient than the food weighs). Please double-check the label.",
                    needs_user_input=True)
    key = food_database.normalise_name(clean)
    existing = food_database.get_food_db(ctx.user_id).get(key)
    if existing is not None and not replace:
        kind = "your own food" if existing.get("custom") else "a built-in food"
        return fail("food_exists", f"'{key}' already exists as {kind}. Choose a different name, or confirm that "
                    "you want to replace it.", needs_user_input=True, food=key)
    try:
        saved = food_database.add_custom_food(
            clean, calories, protein, carbs, fat, serving_size[:120], unit, base_amount,
            fiber=fiber, user_id=ctx.user_id,
        )
    except OSError as exc:
        return fail("save_failed", f"I couldn't save the food: {exc}")
    return ok(f"Saved '{saved}' ({fmt(calories)} kcal per {fmt(base_amount)} {unit}). You can log it now.",
              code="saved", food=saved, provenance="user_defined")
