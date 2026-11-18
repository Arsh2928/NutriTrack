"""
external_nutrition.py
---------------------
Opt-in lookup for a food NOT in the local database from public nutrition APIs
and return its nutrition per 100 g.

Sources, tried in order (ONLY when NUTRITRACK_ONLINE_LOOKUP=1):
  1. USDA FoodData Central (Foundation + SR Legacy generic foods; per-100 g values and
     household portions such as "1 cup" or "1 medium"). Uses USDA_API_KEY via X-Api-Key
     header, or DEMO_KEY when unset.
  2. Open Food Facts (v2 Search API + CGI fallback; packaged / generic products worldwide).

Honesty & safety rules:
  * disabled by default: unknown foods ask the user unless NUTRITRACK_ONLINE_LOOKUP=1;
  * NEVER uses an LLM to generate or guess nutritional values;
  * non-food or gibberish queries are rejected (return None);
  * every result carries its external source in `source` and `provenance` ("usda_fdc",
    "open_food_facts") and is NEVER silently saved into local custom foods;
  * lookups are cached in memory per query.
"""
from collections import OrderedDict
import copy
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

USDA_BASE = "https://api.nal.usda.gov/fdc/v1"
OFF_V2_SEARCH = "https://world.openfoodfacts.org/api/v2/search"
OFF_SEARCH = "https://world.openfoodfacts.org/cgi/search.pl"
TIMEOUT_SECONDS = 2.5
DEFAULT_TOTAL_BUDGET_SECONDS = 4.0
USER_AGENT = "NutriTrack/2.0 (student project; nutrition lookup)"

_STOP_WORDS = {"a", "an", "the", "of", "some", "my", "fresh", "raw", "boiled", "cooked", "plain",
               "whole", "large", "medium", "small", "big", "ripe", "hot", "cold", "steamed",
               "piece", "pieces", "slice", "slices", "cup", "cups", "serving", "servings", "and", "with"}
_CACHE: OrderedDict[str, dict | None] = OrderedDict()          # normalised query -> result dict or None
_CACHE_LOCK = threading.Lock()
_IN_FLIGHT: dict[str, dict] = {}
_DEADLINE_LOCAL = threading.local()


def external_enabled():
    """Public nutrition API lookup is opt-in (NUTRITRACK_ONLINE_LOOKUP=1) and strictly blocked in Offline mode."""
    from ..ai.config import is_offline_mode

    if is_offline_mode():
        return False
    if os.environ.get("NUTRITRACK_DISABLE_EXTERNAL_FOOD", "").strip().lower() in ("1", "true", "yes"):
        return False
    return os.environ.get("NUTRITRACK_ONLINE_LOOKUP", "").strip().lower() in ("1", "true", "yes")


def _remaining_budget() -> float:
    deadline = getattr(_DEADLINE_LOCAL, "deadline", None)
    if deadline is None:
        return TIMEOUT_SECONDS
    return max(0.0, deadline - time.monotonic())


def _http_get_json(url, headers=None):
    from ..ai.config import assert_network_allowed

    assert_network_allowed("external_nutrition")
    rem = _remaining_budget()
    if rem <= 0.05:
        raise TimeoutError("External nutrition lookup time budget exhausted.")
    timeout = min(TIMEOUT_SECONDS, rem)
    req_headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, headers=req_headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:   # nosec - fixed public hosts
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _singular(word):
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("es") and word[-3] in "sxz":
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _words(text):
    return [_singular(w) for w in re.findall(r"[a-z]+", str(text).lower())]


def _query_words(query):
    return [w for w in _words(query) if w not in _STOP_WORDS and len(w) > 1]


def _name_matches(query, candidate_name):
    """Every meaningful query word must appear in the candidate's name."""
    needed = _query_words(query)
    if not needed:
        return False
    have = set(_words(candidate_name))
    return all(w in have for w in needed)


def _num(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and number >= 0 else None


def _with_default_portions(portions):
    """Ensure every external food has practical household portion weights in grams."""
    merged = dict(portions or {})
    serving_g = _num(merged.get("serving")) or _num(merged.get("piece")) or 100.0
    merged.setdefault("serving", round(serving_g, 1))
    merged.setdefault("piece", round(_num(merged.get("piece")) or serving_g, 1))
    merged.setdefault("cup", round(_num(merged.get("cup")) or 240.0, 1))
    merged.setdefault("bowl", round(_num(merged.get("bowl")) or 200.0, 1))
    merged.setdefault("tbsp", round(_num(merged.get("tbsp")) or 15.0, 1))
    merged.setdefault("tsp", round(_num(merged.get("tsp")) or 5.0, 1))
    merged.setdefault("slice", round(_num(merged.get("slice")) or 30.0, 1))
    merged.setdefault("scoop", round(_num(merged.get("scoop")) or 30.0, 1))
    return merged


def _extract_portion_from_label(label, grams, amount, candidates):
    if not grams or amount <= 0:
        return
    per_one = grams / amount
    if re.search(r"\bcup\b", label):
        unit, prio = "cup", 0
    elif re.search(r"\btbsp\b|tablespoon", label):
        unit, prio = "tbsp", 0
    elif re.search(r"\btsp\b|teaspoon", label):
        unit, prio = "tsp", 0
    elif re.search(r"\bslice\b", label):
        unit, prio = "slice", 0
    elif re.search(r"\bscoop\b", label):
        unit, prio = "scoop", 0
    elif re.search(r"\bbowl\b|\bkatori\b", label):
        unit, prio = "bowl", 0
    elif re.search(r"\bmedium\b", label):
        unit, prio = "piece", 0
    elif re.search(r"\bwhole\b|\blarge\b|\bsmall\b|\bundetermined\b|\bpiece\b|\bitem\b|\bfruit\b", label):
        unit, prio = "piece", 1
    elif re.search(r"\bserving\b", label):
        unit, prio = "serving", 1
    else:
        return
    candidates.setdefault(unit, []).append((prio, per_one))


# ---------------------------------------------------------------------------
# 1. USDA FoodData Central
# ---------------------------------------------------------------------------
def _usda_nutrients(food):
    """-> (kcal, protein, carbs, fat, fiber) per 100 g, or None if energy/protein are missing."""
    by_number = {}
    for n in food.get("foodNutrients") or []:
        number = str(n.get("nutrientNumber") or n.get("number") or "")
        value = _num(n.get("value", n.get("amount")))
        if number and value is not None:
            by_number.setdefault(number, value)
    kcal = next((by_number[k] for k in ("208", "957", "958") if k in by_number), None)
    protein, carbs, fat = by_number.get("203"), by_number.get("205"), by_number.get("204")
    fiber = by_number.get("291") if by_number.get("291") is not None else by_number.get("1079")
    if kcal is None or protein is None:
        return None
    return kcal, protein, carbs or 0.0, fat or 0.0, round(fiber or 0.0, 1)


def _usda_search_measures(food):
    """Extract household portions directly from /foods/search `foodMeasures` when present."""
    candidates = {}
    for m in food.get("foodMeasures") or []:
        grams = _num(m.get("gramWeight"))
        label = " ".join(str(x) for x in (m.get("disseminationText"), m.get("modifier"), m.get("measureUnitName")) if x).lower()
        _extract_portion_from_label(label, grams, 1.0, candidates)
    return {u: round(min(c, key=lambda t: t[0])[1], 1) for u, c in candidates.items()}


def _usda_portions(fdc_id, api_key):
    """Household portions -> {"piece": grams, "cup": grams, ...}. Failure is non-fatal."""
    try:
        detail = _http_get_json(f"{USDA_BASE}/food/{fdc_id}", headers={"X-Api-Key": api_key})
    except Exception:
        return {}
    candidates = {}      # unit -> list of (priority, grams)
    for p in detail.get("foodPortions") or []:
        amount, grams = _num(p.get("amount")) or 1.0, _num(p.get("gramWeight"))
        label = " ".join(str(x) for x in (p.get("modifier"), p.get("portionDescription"),
                                          (p.get("measureUnit") or {}).get("name")) if x).lower()
        _extract_portion_from_label(label, grams, amount, candidates)
    return {u: round(min(c, key=lambda t: t[0])[1], 1) for u, c in candidates.items()}


def _lookup_usda(query):
    api_key = os.environ.get("USDA_API_KEY", "").strip() or "DEMO_KEY"
    params = urllib.parse.urlencode({"query": query, "dataType": "Foundation,SR Legacy", "pageSize": 10})
    data = _http_get_json(f"{USDA_BASE}/foods/search?{params}", headers={"X-Api-Key": api_key})
    for food in data.get("foods") or []:
        name = food.get("description") or ""
        if not _name_matches(query, name):
            continue
        nutrients = _usda_nutrients(food)
        if nutrients is None:
            continue
        kcal, protein, carbs, fat, fiber = nutrients
        portions = _usda_search_measures(food)
        if not portions and food.get("fdcId") and _remaining_budget() >= 0.3:
            portions = _usda_portions(food.get("fdcId"), api_key)
        return {"name": name, "calories": kcal, "protein": protein, "carbs": carbs, "fat": fat,
                "fiber": fiber, "portions": _with_default_portions(portions),
                "source": "USDA FoodData Central", "provenance": "usda_fdc"}
    return None


# ---------------------------------------------------------------------------
# 2. Open Food Facts (v2 Search API + legacy CGI fallback)
# ---------------------------------------------------------------------------
_DERIVATIVE_MODIFIERS = {
    "oil", "spray", "seasoning", "extract", "flavour", "flavor",
    "syrup", "candy", "gum", "supplement", "capsule", "tablet",
}


def _parse_off_products(query, products):
    q_words = set(_words(query))
    best_match = None
    best_extra_count = 999
    for product in products or []:
        prod_name = (product.get("product_name") or "").strip()
        full_name = " ".join(x for x in (product.get("brands"), prod_name) if x).strip()
        check_name = full_name or prod_name
        if not check_name or not _name_matches(query, check_name):
            continue
        cand_words = set(_words(prod_name or full_name))
        extra_words = cand_words - q_words - _STOP_WORDS
        # Do not match "Avocado Oil Spray" when user asked for "avocado"
        if extra_words & (_DERIVATIVE_MODIFIERS - q_words):
            continue
        nut = product.get("nutriments") or {}
        kcal = _num(nut.get("energy-kcal_100g"))
        if kcal is None:
            kj = _num(nut.get("energy-kj_100g") or nut.get("energy_100g"))
            if kj is not None:
                kcal = round(kj / 4.184, 1)
        protein = _num(nut.get("proteins_100g"))
        if kcal is None or protein is None:
            continue
        portions = {}
        serving = _num(product.get("serving_quantity"))
        if serving and 0 < serving <= 1000:
            portions["serving"] = round(serving, 1)
        candidate = {
            "name": full_name or check_name,
            "calories": kcal,
            "protein": protein,
            "carbs": _num(nut.get("carbohydrates_100g")) or 0.0,
            "fat": _num(nut.get("fat_100g")) or 0.0,
            "fiber": round(_num(nut.get("fiber_100g")) or 0.0, 1),
            "portions": _with_default_portions(portions),
            "source": "Open Food Facts",
            "provenance": "open_food_facts",
        }
        if len(extra_words) < best_extra_count:
            best_match = candidate
            best_extra_count = len(extra_words)
            if best_extra_count == 0:
                break
    return best_match


def _lookup_off(query):
    v2_params = urllib.parse.urlencode({
        "search_terms": query,
        "page_size": 10,
        "fields": "product_name,brands,nutriments,serving_quantity",
    })
    try:
        v2_data = _http_get_json(f"{OFF_V2_SEARCH}?{v2_params}")
        found = _parse_off_products(query, v2_data.get("products"))
        if found:
            return found
    except Exception:
        pass

    if _remaining_budget() < 0.3:
        return None

    cgi_params = urllib.parse.urlencode({
        "search_terms": query, "search_simple": 1, "action": "process", "json": 1, "page_size": 10,
        "fields": "product_name,brands,nutriments,serving_quantity"})
    data = _http_get_json(f"{OFF_SEARCH}?{cgi_params}")
    return _parse_off_products(query, data.get("products"))


def _get_budget_seconds() -> float:
    raw = os.environ.get("NUTRITRACK_EXTERNAL_LOOKUP_BUDGET_SECONDS", "").strip()
    if raw:
        try:
            return max(0.3, min(15.0, float(raw)))
        except ValueError:
            pass
    return DEFAULT_TOTAL_BUDGET_SECONDS


def lookup_external_food(query):
    """Nutrition per 100 g for `query` from a public API when opted-in (never Offline), with budget & dedup."""
    key = " ".join(_words(query))
    if not key or not external_enabled():
        return None

    budget = _get_budget_seconds()
    waiter = None
    is_owner = False
    with _CACHE_LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return copy.deepcopy(_CACHE[key])
        existing = _IN_FLIGHT.get(key)
        if existing is not None:
            waiter = existing
        else:
            waiter = {"event": threading.Event(), "result": None}
            _IN_FLIGHT[key] = waiter
            is_owner = True

    if not is_owner:
        waiter["event"].wait(timeout=budget + 0.5)
        if not external_enabled():
            return None
        return copy.deepcopy(waiter["result"])

    result, had_error = None, False
    prev_deadline = getattr(_DEADLINE_LOCAL, "deadline", None)
    _DEADLINE_LOCAL.deadline = time.monotonic() + budget
    try:
        for finder in (_lookup_usda, _lookup_off):
            if _remaining_budget() < 0.2:
                had_error = True
                break
            try:
                result = finder(query.strip())
            except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
                result, had_error = None, True
            if result:
                break
        if result or not had_error:           # never remember a miss that was only a network/quota problem
            with _CACHE_LOCK:
                _CACHE[key] = copy.deepcopy(result)
                _CACHE.move_to_end(key)
                while len(_CACHE) > 500:
                    _CACHE.popitem(last=False)
        waiter["result"] = copy.deepcopy(result)
        return result
    finally:
        _DEADLINE_LOCAL.deadline = prev_deadline
        with _CACHE_LOCK:
            _IN_FLIGHT.pop(key, None)
        waiter["event"].set()


def clear_cache():
    with _CACHE_LOCK:
        _CACHE.clear()
