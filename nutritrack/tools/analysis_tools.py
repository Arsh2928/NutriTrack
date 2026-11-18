"""
Nutrition Analysis Agent tools. All figures come from SAVED records in the
shared state, computed in Python; nothing is estimated or filled in.

    calculate_daily_totals       calories, protein, carbs, fat for a day
    calculate_macro_split        share of macro calories from protein / carbs / fat
    get_weekly_trends            averages, target hits and unusual days over a date range
    compare_intake_with_targets  consumed vs the calorie and protein targets
"""
import math
import statistics
from datetime import date, timedelta
from typing import Optional

from ..state import StateError
from .base import agent_tool, fail, ok
from .common import fmt

AGENT = "nutrition_analysis"
MAX_RANGE_DAYS = 92
LOW_FRACTION, HIGH_FRACTION = 0.50, 1.25            # vs the day's calorie target
LOW_MEDIAN_FRACTION, HIGH_MEDIAN_FRACTION = 0.50, 1.50   # vs the user's own median when no target exists
KCAL_PER_GRAM = {"protein": 4, "carbs": 4, "fat": 9}


def _mean(values):
    return round(math.fsum(values) / len(values), 1) if values else None


def compute_daily_totals_data(st, day: str, *, totals: Optional[dict] = None, targets: Optional[dict] = None) -> dict:
    """Pure single-pass daily totals envelope builder (no RunLog side effects)."""
    t = totals if totals is not None else st.day_totals(day)
    tg = targets if targets is not None else st.targets_for(day)
    consumed = {k: t[k] for k in ("calories", "protein", "carbs", "fat")}
    common = dict(date=day, consumed=consumed, fiber=t.get("fiber", 0.0), meal_count=t["meal_count"],
                  has_data=t["has_data"], targets=tg, source="saved_records")
    if not t["has_data"]:
        return ok(f"No meals are logged for {day}, so the totals are 0 because nothing was recorded.",
                  code="no_data", **common)
    return ok(f"{day}: {fmt(consumed['calories'])} kcal, {fmt(consumed['protein'])} g protein, "
              f"{fmt(consumed['carbs'])} g carbs, {fmt(consumed['fat'])} g fat, "
              f"{fmt(t.get('fiber', 0.0))} g fiber from {t['meal_count']} item(s).",
              **common)


def compute_macro_split_data(day: str, t: dict) -> dict:
    """Pure macro split calculation from already-computed day totals."""
    kcal = {k: t[k] * KCAL_PER_GRAM[k] for k in KCAL_PER_GRAM}
    macro_kcal = round(math.fsum(kcal.values()), 1)
    if not t["has_data"] or macro_kcal == 0:
        return ok(f"No macronutrients are logged for {day}.", code="no_data", date=day, has_data=False,
                  grams=None, percent=None, macro_kcal=0.0, logged_kcal=t["calories"])
    pct = {k: round(kcal[k] / macro_kcal * 100, 1) for k in kcal}
    return ok(f"{day}: protein {pct['protein']}%, carbs {pct['carbs']}%, fat {pct['fat']}% of macro calories.",
              date=day, has_data=True, grams={k: t[k] for k in KCAL_PER_GRAM}, percent=pct,
              macro_kcal=macro_kcal, logged_kcal=t["calories"],
              note="Percentages use 4/4/9 kcal per gram of protein/carbs/fat. The total can differ slightly "
                   "from the logged calories, which come from the food database.")


@agent_tool(AGENT)
def calculate_daily_totals(ctx, date: Optional[str] = None) -> dict:
    """Total calories, protein, carbs and fat the user logged on one day, with the targets for that day.

    Args:
        date: "today" (default), "yesterday" or YYYY-MM-DD.
    """
    st = ctx.state
    day = st.resolve_date(date)
    return compute_daily_totals_data(st, day)


@agent_tool(AGENT)
def calculate_macro_split(ctx, date: Optional[str] = None) -> dict:
    """Share of the day's macro calories that came from protein, carbohydrate and fat
    (protein 4 kcal/g, carbs 4 kcal/g, fat 9 kcal/g).

    Args:
        date: "today" (default), "yesterday" or YYYY-MM-DD.
    """
    st = ctx.state
    day = st.resolve_date(date)
    t = st.day_totals(day)
    return compute_macro_split_data(day, t)


def _unusual(rows, today):
    """Days with unusually low/high recorded calories. Today is never flagged (it is still in progress)."""
    flagged = []
    eligible = [r for r in rows if r["has_data"] and not r["is_today"]]
    medians = [r["calories"] for r in eligible]
    median = statistics.median(medians) if len(medians) >= 3 else None
    for r in eligible:
        target = r["calorie_target"]
        if target:
            low, high, basis = LOW_FRACTION * target, HIGH_FRACTION * target, f"{fmt(target)} kcal target"
        elif median:
            low, high, basis = LOW_MEDIAN_FRACTION * median, HIGH_MEDIAN_FRACTION * median, f"your median of {fmt(median)} kcal"
        else:
            continue
        if r["calories"] < low:
            flagged.append({"date": r["date"], "calories": r["calories"], "flag": "low",
                            "reason": f"{fmt(r['calories'])} kcal is under {int(LOW_FRACTION * 100) if target else int(LOW_MEDIAN_FRACTION * 100)}% of {basis}. "
                                      "Meals may be missing from the log."})
        elif r["calories"] > high:
            flagged.append({"date": r["date"], "calories": r["calories"], "flag": "high",
                            "reason": f"{fmt(r['calories'])} kcal is over {int(HIGH_FRACTION * 100) if target else int(HIGH_MEDIAN_FRACTION * 100)}% of {basis}."})
    return flagged


def compute_weekly_trends_data(
    st,
    *,
    end_date: Optional[str] = None,
    days: int = 7,
    start_date: Optional[str] = None,
    include_today: bool = True,
) -> tuple[dict, str, dict[str, dict], dict[str, dict]]:
    """Compute weekly trends once, returning `(trends_envelope, end_iso, totals_by_day, targets_by_day)`."""
    today = st.today()
    end = date.fromisoformat(st.resolve_date(end_date))
    end_iso = end.isoformat()
    if start_date is not None:
        start = date.fromisoformat(st.resolve_date(start_date))
    else:
        if isinstance(days, bool) or not isinstance(days, int) or not 1 <= days <= MAX_RANGE_DAYS:
            raise StateError(f"days must be a whole number from 1 to {MAX_RANGE_DAYS}.")
        start = end - timedelta(days=days - 1)
    if start > end:
        raise StateError("The start date must not be after the end date.")
    span = (end - start).days + 1
    if span > MAX_RANGE_DAYS:
        raise StateError(f"The date range is limited to {MAX_RANGE_DAYS} days (got {span}).")

    dates = [(start + timedelta(days=i)).isoformat() for i in range(span)]
    if not include_today:
        dates = [d for d in dates if d != today]

    st.reload_if_changed()
    rows = []
    totals_by_day: dict[str, dict] = {}
    targets_by_day: dict[str, dict] = {}
    for d in dates:
        t, tg = st.day_totals(d), st.targets_for(d)
        totals_by_day[d] = t
        targets_by_day[d] = tg
        has = t["has_data"]
        row = {"date": d, "has_data": has, "is_today": d == today, "meal_count": t["meal_count"],
               "calories": t["calories"] if has else None, "protein": t["protein"] if has else None,
               "carbs": t["carbs"] if has else None, "fat": t["fat"] if has else None,
               "fiber": t.get("fiber", 0.0) if has else None,
               "calorie_target": tg["calories"], "protein_target": tg["protein"],
               "within_calorie_target": None, "met_protein_target": None}
        if has and tg["calories"] is not None:
            row["within_calorie_target"] = t["calories"] <= tg["calories"]
        if has and tg["protein"] is not None:
            row["met_protein_target"] = t["protein"] >= tg["protein"]
        rows.append(row)

    data = [r for r in rows if r["has_data"]]
    missing = [r["date"] for r in rows if not r["has_data"]]
    period = {"start": dates[0] if dates else start.isoformat(), "end": end_iso, "days_in_range": len(dates)}
    series = [{"date": r["date"], "calories": r["calories"]} for r in rows]
    if not data:
        env = ok(f"No saved records between {period['start']} and {period['end']}. Nothing can be averaged yet.",
                 code="no_data", period=period, days_with_data=0, days_without_data=missing, averages=None,
                 daily=rows, calorie_series=series, unusual_days=[], calorie_target_summary=None,
                 protein_target_summary=None,
                 data_period_note=f"0 of {len(dates)} days have saved records.")
        return env, end_iso, totals_by_day, targets_by_day

    averages = {k: _mean([r[k] for r in data]) for k in ("calories", "protein", "carbs", "fat", "fiber")}
    judged = [r for r in data if r["within_calorie_target"] is not None]
    cal_summary = {
        "days_evaluated": len(judged),
        "days_within_target": sum(1 for r in judged if r["within_calorie_target"]),
        "days_over_target": sum(1 for r in judged if not r["within_calorie_target"]),
        "definition": "A day is 'within target' when the calories logged are at or below that day's calorie target.",
    }
    pj = [r for r in data if r["met_protein_target"] is not None]
    protein_summary = {
        "days_evaluated": len(pj),
        "days_met": sum(1 for r in pj if r["met_protein_target"]),
        "average_protein_g": averages["protein"],
        "average_percent_of_target": _mean([r["protein"] / r["protein_target"] * 100 for r in pj]) if pj else None,
        "definition": "A day 'meets' the protein target when the protein logged is at or above that day's target.",
    }
    first, last = data[0]["date"], data[-1]["date"]
    note = (f"Averages use {len(data)} of {len(dates)} days that have saved records ({first} to {last}); "
            + (f"{len(missing)} day(s) have no records and were NOT counted as zero." if missing
               else "every day in the range has records."))
    today_row = next((r for r in data if r["is_today"]), None)
    if today_row:
        note += f" Today ({today}) is included and may still be incomplete."
    env = ok(f"Weekly summary: {len(data)} day(s) of data, average {fmt(averages['calories'])} kcal and "
             f"{fmt(averages['protein'])} g protein per day.",
             period=period, days_with_data=len(data), days_without_data=missing, average_basis="days_with_data",
             averages=averages, calorie_target_summary=cal_summary, protein_target_summary=protein_summary,
             unusual_days=_unusual(rows, today), daily=rows, calorie_series=series, data_period_note=note,
             unusual_rule=(f"Low: under {int(LOW_FRACTION * 100)}% of the day's calorie target; high: over "
                           f"{int(HIGH_FRACTION * 100)}%. Without a target the user's own median is used "
                           f"({int(LOW_MEDIAN_FRACTION * 100)}% / {int(HIGH_MEDIAN_FRACTION * 100)}%, needs 3+ days). "
                           "Today is never flagged."))
    return env, end_iso, totals_by_day, targets_by_day


def compute_insights_bundle(
    st,
    *,
    end_date: Optional[str] = None,
    days: int = 7,
    start_date: Optional[str] = None,
    include_today: bool = True,
) -> dict:
    """Single-pass bundle returning `trends`, `macro_split`, and `daily_totals` without duplicate work."""
    trends, end_iso, totals_by_day, targets_by_day = compute_weekly_trends_data(
        st, end_date=end_date, days=days, start_date=start_date, include_today=include_today
    )
    end_totals = totals_by_day.get(end_iso) or st.day_totals(end_iso)
    end_targets = targets_by_day.get(end_iso) or st.targets_for(end_iso)
    return {
        "trends": trends,
        "macro_split": compute_macro_split_data(end_iso, end_totals),
        "daily_totals": compute_daily_totals_data(st, end_iso, totals=end_totals, targets=end_targets),
        "period_end": (trends.get("period") or {}).get("end") or end_iso,
    }


@agent_tool(AGENT)
def get_weekly_trends(ctx, end_date: Optional[str] = None, days: int = 7, start_date: Optional[str] = None,
                      include_today: bool = True) -> dict:
    """Averages, target hits and unusual days over a date range, from SAVED records only.

    Days with no saved meals are reported as missing and are NOT counted as zero, so averages are over the
    days that have data and `data_period_note` states exactly how many that is.

    Args:
        end_date: last day of the range: "today" (default), "yesterday" or YYYY-MM-DD.
        days: length of the range ending at end_date (1-92, default 7). Ignored if start_date is given.
        start_date: optional first day (YYYY-MM-DD) for a custom range.
        include_today: if true (default) today's still-incomplete totals are included and flagged.
    """
    trends, _, _, _ = compute_weekly_trends_data(
        ctx.state, end_date=end_date, days=days, start_date=start_date, include_today=include_today
    )
    return trends


@agent_tool(AGENT)
def compare_intake_with_targets(ctx, date: Optional[str] = None) -> dict:
    """Compare what was logged on a day with that day's calorie and protein targets.

    Args:
        date: "today" (default), "yesterday" or YYYY-MM-DD.
    """
    st = ctx.state
    day = st.resolve_date(date)
    t, tg = st.day_totals(day), st.targets_for(day)
    if tg["calories"] is None and tg["protein"] is None:
        return fail("no_target", "No calorie or protein target is set, so there is nothing to compare with.",
                    needs_user_input=True, question="What daily calorie and protein targets would you like to use?")
    missing = [k for k in ("calories", "protein") if tg[k] is None]

    def block(kind, consumed, target, ok_label, bad_label, passed):
        if target is None:
            return {"target": None, "consumed": consumed, "status": "no_target"}
        return {"target": target, "consumed": consumed, "difference": round(consumed - target, 1),
                "percent_of_target": round(consumed / target * 100, 1),
                "status": ok_label if passed(consumed, target) else bad_label}

    calories = block("calories", t["calories"], tg["calories"], "within_target", "over_target", lambda c, g: c <= g)
    protein = block("protein", t["protein"], tg["protein"], "met", "below_target", lambda c, g: c >= g)
    parts = []
    if calories["status"] != "no_target":
        parts.append(f"calories {fmt(t['calories'])}/{fmt(tg['calories'])} kcal ({calories['percent_of_target']}%, {calories['status'].replace('_', ' ')})")
    if protein["status"] != "no_target":
        parts.append(f"protein {fmt(t['protein'])}/{fmt(tg['protein'])} g ({protein['percent_of_target']}%, {protein['status'].replace('_', ' ')})")
    msg = f"{day}: " + "; ".join(parts) + "."
    if not t["has_data"]:
        msg += " Nothing is logged for this day yet."
    if missing:
        msg += f" No {' or '.join(missing)} target is set."
    return ok(msg, date=day, has_data=t["has_data"], calories=calories, protein=protein,
              missing_targets=missing, needs_user_input=bool(missing),
              question=(f"Would you like to set a {' and a '.join(missing)} target?" if missing else None))
