"""
Goal and Calorie Budget Agent tools. Every number is computed in Python.

    set_calorie_target          save the user's daily calorie target
    set_protein_target          save the user's daily protein target
    calculate_remaining_targets target minus what was eaten (never clipped: over budget shows as negative)
    check_meal_against_budget   would this proposed meal fit in what is left today?

Targets are the USER's. Nothing here prescribes a target; the demo user starts with
clearly labelled demonstration defaults, and every other user is asked to choose.
"""
from typing import Optional

from .base import agent_tool, fail, ok
from .common import fmt, normalise_items, resolve_items, totals_of

AGENT = "goal_budget"


def _set_target(ctx, kind, value, unit):
    previous_source = ctx.state.targets_source()[kind]
    change = ctx.state.set_target(kind, value, source="user")          # validates limits, saves atomically
    remaining = ctx.state.remaining()
    old = "no target" if change["old"] is None else f"{fmt(change['old'])} {unit}"
    note = " (replacing a demonstration default)" if previous_source == "demo_default" else ""
    return ok(f"Daily {kind} target set to {fmt(change['new'])} {unit}, was {old}{note}. "
              f"Remaining today: {fmt(remaining[kind])} {unit}.",
              target=change["new"], previous=change["old"], previous_source=previous_source,
              remaining=remaining, source="user")


@agent_tool(AGENT)
def set_calorie_target(ctx, calories: float) -> dict:
    """Set the user's daily calorie target (kcal). Use the number the USER gave; never choose a target for them.
    Accepted range 500-10000 (this only catches typos; it is not medical advice).

    Args:
        calories: daily calorie target in kcal, e.g. 2300.
    """
    return _set_target(ctx, "calories", calories, "kcal")


@agent_tool(AGENT)
def set_protein_target(ctx, protein: float) -> dict:
    """Set the user's daily protein target (grams). Use the number the USER gave; never choose a target for them.
    Accepted range 10-500 g (this only catches typos; it is not medical advice).

    Args:
        protein: daily protein target in grams, e.g. 150.
    """
    return _set_target(ctx, "protein", protein, "g")


@agent_tool(AGENT)
def calculate_remaining_targets(ctx, date: Optional[str] = None) -> dict:
    """How many calories and grams of protein are left for the day (target minus what was logged).
    A negative number means the target was exceeded.

    Args:
        date: "today" (default), "yesterday" or YYYY-MM-DD.
    """
    st = ctx.state
    day = st.resolve_date(date)
    targets, totals = st.targets_for(day), st.day_totals(day)
    if targets["calories"] is None and targets["protein"] is None:
        return fail("no_target", "No calorie or protein target is set, so the remaining amount can't be calculated.",
                    needs_user_input=True, question="What daily calorie and protein targets would you like to use?")
    remaining = st.remaining(day)
    missing = [k for k in ("calories", "protein") if targets[k] is None]
    parts = []
    if remaining["calories"] is not None:
        r = remaining["calories"]
        parts.append(f"{fmt(r)} kcal left" if r >= 0 else f"{fmt(-r)} kcal OVER the calorie target")
    if remaining["protein"] is not None:
        r = remaining["protein"]
        parts.append(f"{fmt(r)} g protein still to go" if r >= 0 else f"protein target exceeded by {fmt(-r)} g")
    message = f"{day}: " + "; ".join(parts) + "."
    if missing:
        message += f" No {' or '.join(missing)} target is set."
    return ok(message, date=day, targets=targets,
              consumed={k: totals[k] for k in ("calories", "protein", "carbs", "fat")}, remaining=remaining,
              calories_over_by=(-remaining["calories"] if remaining["calories"] is not None and remaining["calories"] < 0 else 0.0),
              missing_targets=missing, needs_user_input=bool(missing),
              question=(f"Would you like to set a {' and a '.join(missing)} target?" if missing else None),
              targets_source=st.targets_source())


@agent_tool(AGENT)
def check_meal_against_budget(ctx, items: list[dict], date: Optional[str] = None) -> dict:
    """Would a proposed meal fit in the calories left today? Nutrition is looked up here from the food database
    (pass only food names and amounts; never pass calorie numbers).

    Args:
        items: list of {"food": "egg", "quantity": 3, "unit": null} describing the proposed meal.
        date: "today" (default), "yesterday" or YYYY-MM-DD.

    Returns:
        fits (True/False), the meal's nutrition, remaining_before / remaining_after, and surplus_deficit
        (calories left AFTER the meal; negative means the meal exceeds the budget).
    """
    resolved, failures = resolve_items(normalise_items(items), user_id=ctx.user_id)
    if failures:
        first = failures[0]
        names = ", ".join(f"'{f['food']}'" for f in failures)
        return fail(first["code"] if len(failures) == 1 else "unresolved_items",
                    f"I can't check the budget until I know about: {names}. " + (first["message"] if len(failures) == 1 else ""),
                    needs_user_input=True, question=first["question"], unresolved=failures)
    st = ctx.state
    day = st.resolve_date(date)
    meal = totals_of([r["nutrition"] for r in resolved])
    listing = [{"food": r["nutrition"]["food"], "quantity_str": r["nutrition"]["quantity_str"],
                "calories": r["nutrition"]["calories"], "protein": r["nutrition"]["protein"]} for r in resolved]
    targets, before = st.targets_for(day), st.remaining(day)
    if targets["calories"] is None:
        return fail("no_target", f"The meal is about {fmt(meal['calories'])} kcal, but no calorie target is set, "
                    "so I can't say whether it fits.", needs_user_input=True,
                    question="What is your daily calorie target?", meal=meal, items=listing)
    after_kcal = round(before["calories"] - meal["calories"], 1)
    after_protein = None if before["protein"] is None else round(before["protein"] - meal["protein"], 1)
    fits = after_kcal >= 0
    verdict = (f"Fits: {fmt(meal['calories'])} kcal leaves {fmt(after_kcal)} kcal of the {fmt(before['calories'])} kcal remaining."
               if fits else
               f"Does NOT fit: {fmt(meal['calories'])} kcal is {fmt(-after_kcal)} kcal more than the {fmt(before['calories'])} kcal remaining.")
    if before["calories"] < 0:
        verdict += " (The calorie target was already exceeded before this meal.)"
    return ok(verdict, code="fits_budget" if fits else "exceeds_budget", fits=fits, date=day, meal=meal, items=listing,
              remaining_before=before, remaining_after={"calories": after_kcal, "protein": after_protein},
              surplus_deficit=after_kcal, protein_remaining_after=after_protein, estimated=True)
