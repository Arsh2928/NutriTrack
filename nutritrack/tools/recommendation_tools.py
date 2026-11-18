"""
Meal Recommendation Agent tools -- a four-step pipeline, all deterministic:

    get_recommendation_candidates   build candidate meals from the food database (kept in a pool, referred to by id)
    estimate_meal_nutrition         recompute each candidate's nutrition from the database
    rank_meal_candidates            score, rank, explain, and offer alternatives
    validate_recommendation_fit     re-check a candidate from scratch before it is shown or saved

Candidates are referred to by short ids ("c07") so a language model never re-types nutrition numbers.
Scoring (documented, no randomness; ties broken by name):

    score = 0.50 * protein_score + 0.25 * calorie_fill + 0.15 * variety + 0.10 * preference
    meal size   = min(calories remaining, 600 kcal)            (a typical main-meal size)
    calorie_fill = 1 - |meal kcal - meal size| / meal size     (best when the meal uses the room available)
    protein_share = remaining protein * meal size / remaining kcal   (protein needed in proportion to the meal)
    protein_score = min(1, meal protein / protein_share)
    variety      = 1.0 for 2+ foods, 0.6 for a single food
    preference   = 1.0 if any food is one the user prefers

A meal that exceeds the remaining calories is never a candidate. Not medical advice.
"""
import itertools
import math
from typing import Optional

from ..ca1.food_database import get_food_db
from ..core.nutrition import resolve_item
from ..state import DIETS, StateError
from . import recommendation_data as data
from .base import agent_tool, fail, ok
from .common import fmt, normalise_items, resolve_items, totals_of

AGENT = "meal_recommendation"
MEAL_TARGET_KCAL = 600.0
PROTEIN_KCAL_SHARE = 0.25            # used only when the protein still needed is unknown
WEIGHTS = {"protein": 0.50, "calorie_fill": 0.25, "variety": 0.15, "preference": 0.10}
MIN_FILL = 0.35                      # a candidate must use at least 35% of the meal size
MIN_CANDIDATE_KCAL = 80.0
MAX_PER_MAIN_FOOD = 4                # diversity: at most 4 candidates built around the same main food
MAX_OWN_FOODS = 5                    # candidates reserved for the user's own custom foods (on top of the quota)
DISCLAIMER = ("Nutrition values are estimates from a food database and portions are suggestions. "
              "This is not medical or dietary advice.")
NUTRIENTS = ("calories", "protein", "carbs", "fat")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _pool(ctx):
    pool = ctx.state.pool
    pool.setdefault("candidates", {})
    pool.setdefault("meta", {})
    pool.setdefault("validated", [])
    return pool


def _names(values, label):
    if values is None:
        return None
    if isinstance(values, str) or not isinstance(values, (list, tuple)):
        raise StateError(f"'{label}' must be a list of food names.")
    return sorted({str(v).strip().lower() for v in values if str(v).strip()})


def _check_number(value, label):
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
        raise StateError(f"{label} must be a number.")


def _avoided(key, entry, avoid):
    names = {key, *[a.lower() for a in entry.get("aliases", [])]}
    return any(a in names or a.rstrip("s") in {n.rstrip("s") for n in names} for a in avoid)


def _remaining(ctx, remaining_calories, remaining_protein):
    """-> (kcal, protein or None, source). An explicit remaining_calories wins; otherwise the shared state."""
    _check_number(remaining_calories, "remaining_calories")
    _check_number(remaining_protein, "remaining_protein")
    if remaining_calories is not None:
        return float(remaining_calories), (None if remaining_protein is None else float(remaining_protein)), "user_supplied"
    rem = ctx.state.remaining()
    return rem["calories"], (rem["protein"] if remaining_protein is None else float(remaining_protein)), "shared_state"


def _name(items):
    return " + ".join(f"{i['quantity_str']} {i['food']}" for i in items)


def _meal(items):
    return {"items": items, **totals_of(items), "name": _name(items)}


def _score(meal, rc, rp, prefer):
    size = min(rc, MEAL_TARGET_KCAL)
    fill = max(0.0, 1 - abs(meal["calories"] - size) / size)
    if rp is None:
        share = PROTEIN_KCAL_SHARE * size / 4
    elif rp <= 0:
        share = None                                   # protein target already met
    else:
        share = rp * size / rc
    protein = 1.0 if share is None else min(1.0, meal["protein"] / share)
    variety = 1.0 if len(meal["items"]) >= 2 else 0.6
    liked = 1.0 if any(i["food"] in prefer for i in meal["items"]) else 0.0
    total = (WEIGHTS["protein"] * protein + WEIGHTS["calorie_fill"] * fill
             + WEIGHTS["variety"] * variety + WEIGHTS["preference"] * liked)
    return round(total, 4), {"protein": round(protein, 3), "calorie_fill": round(fill, 3), "variety": variety,
                             "preference": liked, "protein_share_g": None if share is None else round(share, 1)}


def _reason(meal, rc, rp):
    used = meal["calories"] / rc * 100
    text = (f"{fmt(meal['calories'])} kcal uses {used:.0f}% of the {fmt(rc)} kcal you have left, "
            f"leaving {fmt(rc - meal['calories'])} kcal, and provides {fmt(meal['protein'])} g protein")
    if rp is None:
        text += "."
    elif rp <= 0:
        text += " (your protein target is already met)."
    else:
        text += f" ({min(100, meal['protein'] / rp * 100):.0f}% of the {fmt(rp)} g protein still needed)."
    return text + " Estimated values; not medical advice."


def _recalculate(items, *, user_id=None):
    """Recompute a candidate from its foods and amounts -> meal dict, or (None, failures)."""
    resolved, failures = resolve_items(normalise_items(items), user_id=user_id)
    if failures:
        return None, failures
    return _meal([{"food": r["nutrition"]["food"], "quantity": r["request"]["quantity"], "unit": r["request"]["unit"],
                   "quantity_str": r["nutrition"]["quantity_str"],
                   **{k: r["nutrition"][k] for k in NUTRIENTS}} for r in resolved]), []


def _options_by_role(diet, avoid, rc, *, user_id=None):
    by_role = {}
    for key, entry in get_food_db(user_id).items():
        role = data.ROLES.get(key) or ("other" if entry.get("custom") else None)
        if role is None or not data.diet_allows(diet, key) or _avoided(key, entry, avoid):
            continue
        for quantity, unit in data.LADDERS.get(key) or data.fallback_ladder(entry):
            res = resolve_item(key, quantity, unit, allow_external=False, user_id=user_id)
            if res["ok"] and res["calories"] <= rc:
                by_role.setdefault(role, []).append(
                    {"food": key, "quantity": quantity, "unit": unit, "quantity_str": res["quantity_str"],
                     **{k: res[k] for k in NUTRIENTS}})
    return by_role


def _diverse(scored, limit):
    """Best-first, but at most MAX_PER_MAIN_FOOD per main food so one food can't fill the list."""
    chosen, per_main = [], {}
    for entry in scored:
        main = entry["meal"]["items"][0]["food"]
        if per_main.get(main, 0) >= MAX_PER_MAIN_FOOD:
            continue
        per_main[main] = per_main.get(main, 0) + 1
        chosen.append(entry)
        if len(chosen) == limit:
            break
    return chosen


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------
@agent_tool(AGENT)
def get_recommendation_candidates(ctx, remaining_calories: Optional[float] = None,
                                  remaining_protein: Optional[float] = None, diet: Optional[str] = None,
                                  avoid: Optional[list[str]] = None, prefer: Optional[list[str]] = None,
                                  max_candidates: int = 30) -> dict:
    """Build candidate meals from the food database that fit in the calories left, and keep them in a pool.

    Leave remaining_calories / remaining_protein EMPTY to use what is actually left today in the shared state
    (do not work these out yourself). Pass remaining_calories only when the user asks about a specific
    hypothetical amount ("a meal with 500 calories left"). Candidates are returned as short ids; pass those
    to estimate_meal_nutrition, rank_meal_candidates and validate_recommendation_fit.

    Args:
        remaining_calories: calories the meal must fit in (default: today's remaining from the state).
        remaining_protein: grams of protein still needed (default: from the state; unknown if you gave calories).
        diet: any, vegetarian (no meat or egg) or eggetarian (no meat). Default: the user's saved preference.
        avoid: foods to leave out (replaces the user's saved list for this search only).
        prefer: foods the user likes (a small ranking bonus).
        max_candidates: 1-60 (default 30). The user's own custom foods are added on top of this.
    """
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or not 1 <= max_candidates <= 60:
        raise StateError("max_candidates must be a whole number from 1 to 60.")
    prefs = ctx.state.preferences()
    diet = diet if diet is not None else prefs["diet"]
    if diet not in DIETS:
        raise StateError(f"diet must be one of {', '.join(DIETS)}.")
    avoid = _names(avoid, "avoid") if avoid is not None else prefs["avoid"]
    prefer = _names(prefer, "prefer") if prefer is not None else prefs["prefer"]
    rc, rp, source = _remaining(ctx, remaining_calories, remaining_protein)
    if rc is None:
        return fail("no_target", "No calorie target is set, so I don't know how many calories are left.",
                    needs_user_input=True, question="What is your daily calorie target?")
    pool = _pool(ctx)
    pool["candidates"], pool["validated"] = {}, []
    filters = {"remaining_calories": rc, "remaining_protein": rp, "source": source, "diet": diet,
               "avoid": avoid, "prefer": prefer}
    pool["meta"] = dict(filters)
    if rc <= 0:
        return ok(f"No calories are left today ({fmt(-rc)} kcal over), so any meal would exceed the target.",
                  code="no_budget_left", candidate_ids=[], count=0, candidates=[], filters=filters)

    min_kcal = max(MIN_CANDIDATE_KCAL, MIN_FILL * min(rc, MEAL_TARGET_KCAL))
    by_role = _options_by_role(diet, avoid, rc, user_id=ctx.user_id)
    scored = []
    for template in data.TEMPLATES:
        if any(role not in by_role for role in template):
            continue
        for combo in itertools.product(*(by_role[r] for r in template)):
            kcal = math.fsum(o["calories"] for o in combo)
            if min_kcal <= kcal <= rc:
                meal = _meal([dict(o) for o in combo])
                score, _ = _score(meal, rc, rp, prefer)
                scored.append({"score": score, "meal": meal, "roles": template})
    scored.sort(key=lambda e: (-e["score"], e["meal"]["name"]))
    # The user's own (custom) foods are single-food meals that would almost never beat thousands of
    # built-in combinations, so each gets its best candidate reserved on top of the normal quota.
    regular = [e for e in scored if e["roles"] != ("other",)]
    own_best = {}
    for e in scored:
        if e["roles"] == ("other",):
            own_best.setdefault(e["meal"]["items"][0]["food"], e)
    chosen = _diverse(regular, max_candidates) + list(own_best.values())[:MAX_OWN_FOODS]
    chosen.sort(key=lambda e: (-e["score"], e["meal"]["name"]))
    if not chosen:
        return ok(f"No meal in the food database fits between {fmt(min_kcal)} and {fmt(rc)} kcal with these filters.",
                  code="no_candidates", candidate_ids=[], count=0, candidates=[], filters=filters)
    brief = []
    for n, entry in enumerate(chosen, 1):
        cid = f"c{n:02d}"
        items = [{k: i[k] for k in ("food", "quantity", "unit", "quantity_str")} for i in entry["meal"]["items"]]
        pool["candidates"][cid] = {"id": cid, "items": items, "roles": list(entry["roles"]), "name": entry["meal"]["name"]}
        brief.append({"id": cid, "name": entry["meal"]["name"], "roles": list(entry["roles"])})
    return ok(f"Built {len(brief)} candidate meal(s) that fit within {fmt(rc)} kcal.", candidate_ids=[b["id"] for b in brief],
              count=len(brief), candidates=brief, filters=filters)


@agent_tool(AGENT)
def estimate_meal_nutrition(ctx, candidate_ids: Optional[list[str]] = None, items: Optional[list[dict]] = None) -> dict:
    """Estimate the nutrition of candidate meals (or of one ad-hoc meal) from the food database.
    Values are recomputed here; never pass calorie numbers.

    Args:
        candidate_ids: ids from get_recommendation_candidates (default: all candidates).
        items: alternatively, one meal as a list of {"food": "egg", "quantity": 2, "unit": null}.
    """
    if items is not None and candidate_ids is not None:
        raise StateError("Give either candidate_ids or items, not both.")
    if items is not None:
        meal, failures = _recalculate(items, user_id=ctx.user_id)
        if failures:
            first = failures[0]
            return fail(first["code"], first["message"], needs_user_input=True, question=first["question"],
                        unresolved=failures)
        return ok(f"Estimated: {fmt(meal['calories'])} kcal, {fmt(meal['protein'])} g protein.",
                  estimates=[{"id": "adhoc", **meal}], estimated=True, disclaimer=DISCLAIMER)
    pool = _pool(ctx)
    if not pool["candidates"]:
        return fail("no_candidates", "There are no candidate meals yet. Call get_recommendation_candidates first.")
    ids = candidate_ids if candidate_ids else list(pool["candidates"])
    unknown = [i for i in ids if i not in pool["candidates"]]
    if unknown:
        return fail("unknown_candidate", f"Unknown candidate id(s): {', '.join(map(str, unknown))}.")
    estimates = []
    for cid in ids:
        meal, failures = _recalculate(pool["candidates"][cid]["items"], user_id=ctx.user_id)
        if failures:
            return fail("unresolvable_candidate", f"Candidate {cid} contains a food I can no longer resolve.",
                        unresolved=failures)
        pool["candidates"][cid]["estimate"] = {k: meal[k] for k in NUTRIENTS}
        estimates.append({"id": cid, **meal})
    return ok(f"Estimated nutrition for {len(estimates)} candidate meal(s).", estimates=estimates,
              estimated=True, disclaimer=DISCLAIMER)


@agent_tool(AGENT)
def rank_meal_candidates(ctx, remaining_calories: Optional[float] = None, remaining_protein: Optional[float] = None,
                         top_n: int = 5) -> dict:
    """Rank the estimated candidates (best first), explain each, and return the others as alternatives.
    Call estimate_meal_nutrition first. Leave the remaining_* arguments empty to rank against the same
    amounts the candidates were built for.

    Args:
        remaining_calories: optional override of the calories the meal must fit in.
        remaining_protein: optional override of the protein still needed.
        top_n: how many ranked meals to return (1-10, default 5).
    """
    if isinstance(top_n, bool) or not isinstance(top_n, int) or not 1 <= top_n <= 10:
        raise StateError("top_n must be a whole number from 1 to 10.")
    pool = _pool(ctx)
    if not pool["candidates"]:
        return fail("no_candidates", "There are no candidate meals yet. Call get_recommendation_candidates first.")
    estimated = {cid: c for cid, c in pool["candidates"].items() if c.get("estimate")}
    if not estimated:
        return fail("not_estimated", "The candidates have not been estimated yet. Call estimate_meal_nutrition first.")
    meta = pool["meta"]
    if remaining_calories is None and remaining_protein is None:
        rc, rp = meta["remaining_calories"], meta["remaining_protein"]
    else:
        rc, rp, _ = _remaining(ctx, remaining_calories, remaining_protein)
    if rc is None or rc <= 0:
        return fail("no_budget_left", "There are no calories left to build a meal from.")
    prefer = meta.get("prefer", [])
    scored = []
    for cid, cand in estimated.items():
        meal = {"items": cand["items"], **cand["estimate"], "name": cand["name"]}
        if meal["calories"] > rc:
            continue
        score, breakdown = _score(meal, rc, rp, prefer)
        scored.append({"id": cid, "score": score, "breakdown": breakdown, "meal": meal})
    scored.sort(key=lambda e: (-e["score"], e["meal"]["name"]))
    if not scored:
        return ok("None of the estimated candidates fits within the remaining calories.", code="no_fit", ranked=[])
    top = _diverse(scored, top_n)
    ranked = []
    for rank, e in enumerate(top, 1):
        m = e["meal"]
        info = {"rank": rank, "score": e["score"], "breakdown": e["breakdown"], "reason": _reason(m, rc, rp)}
        pool["candidates"][e["id"]]["rank_info"] = info
        ranked.append({"id": e["id"], "name": m["name"], "items": m["items"],
                       **{k: m[k] for k in NUTRIENTS}, "fits_calories": True, **info})
    pool["ranked"] = [r["id"] for r in ranked]
    return ok(f"Ranked {len(ranked)} meal(s); best: {ranked[0]['name']}.", ranked=ranked, best_id=ranked[0]["id"],
              alternative_ids=[r["id"] for r in ranked[1:]],
              remaining_used={"calories": rc, "protein": rp}, weights=WEIGHTS, meal_target_kcal=MEAL_TARGET_KCAL,
              estimated=True, disclaimer=DISCLAIMER)


@agent_tool(AGENT)
def validate_recommendation_fit(ctx, candidate_id: str, remaining_calories: Optional[float] = None,
                                remaining_protein: Optional[float] = None) -> dict:
    """Re-check one candidate from scratch (nutrition is recomputed from the database, not trusted from earlier
    steps) against what is left. A valid candidate is saved in the shared state as a recommendation.
    Always run this on every meal you intend to recommend.

    Args:
        candidate_id: the id to check, e.g. "c03".
        remaining_calories / remaining_protein: optional overrides; by default today's real remaining is used
            (or the hypothetical amount the candidates were built for).
    """
    pool = _pool(ctx)
    cand = pool["candidates"].get(candidate_id)
    if cand is None:
        return fail("unknown_candidate", f"There is no candidate '{candidate_id}'. Use an id from the ranked list.")
    meal, failures = _recalculate(cand["items"], user_id=ctx.user_id)
    if failures:
        return fail("unresolvable_candidate", f"Candidate {candidate_id} contains a food I can no longer resolve.",
                    unresolved=failures)
    meta = pool["meta"]
    if remaining_calories is None and remaining_protein is None and meta.get("source") == "user_supplied":
        rc, rp = meta["remaining_calories"], meta["remaining_protein"]
    else:
        rc, rp, _ = _remaining(ctx, remaining_calories, remaining_protein)
    if rc is None:
        return fail("no_target", "No calorie target is set, so I can't check the fit.", needs_user_input=True,
                    question="What is your daily calorie target?")
    issues = []
    estimate = cand.get("estimate")
    if estimate and any(abs(estimate[k] - meal[k]) > 0.05 for k in NUTRIENTS):
        issues.append("The stored estimate no longer matches the recomputed nutrition.")
    if meal["calories"] > rc:
        issues.append(f"{fmt(meal['calories'])} kcal exceeds the {fmt(rc)} kcal remaining by {fmt(meal['calories'] - rc)} kcal.")
    checks = {"calories": meal["calories"], "remaining_calories": rc,
              "calories_left_after": round(rc - meal["calories"], 1), "fits_calories": meal["calories"] <= rc,
              "protein": meal["protein"], "remaining_protein": rp,
              "protein_percent_of_remaining": (None if not rp or rp <= 0 else round(meal["protein"] / rp * 100, 1))}
    if issues:
        return fail("invalid_recommendation", "This meal is not a valid recommendation: " + " ".join(issues),
                    valid=False, issues=issues, checks=checks, meal=meal)
    info = cand.get("rank_info") or {}
    rec = {"id": candidate_id, "name": meal["name"],
           "items": [{k: i[k] for k in ("food", "quantity_str", *NUTRIENTS)} for i in meal["items"]],
           **{k: meal[k] for k in NUTRIENTS}, "reason": info.get("reason") or _reason(meal, rc, rp),
           "score": info.get("score"), "estimated": True, "disclaimer": DISCLAIMER}
    pool["validated"] = [r for r in pool["validated"] if r["id"] != candidate_id] + [rec]
    ctx.state.set_recommendations(pool["validated"][-10:])
    return ok(f"Valid: {meal['name']} ({fmt(meal['calories'])} kcal) fits with {fmt(rc - meal['calories'])} kcal to spare.",
              code="valid", valid=True, recommendation=rec, checks=checks)
