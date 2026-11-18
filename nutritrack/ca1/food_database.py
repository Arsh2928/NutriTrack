"""
food_database.py
----------------
Built-in nutrition data, food-name matching, and permanent storage for
user-defined (custom) foods.

*** RECONSTRUCTED FILE ***
The original CA1 food_database.py was missing from the submitted ZIP, although
tools.py imports it. This module was rebuilt (Phase 1 of the CA2 upgrade) from:

  * the public interface that tools.py relies on (FOOD_DB, find_food,
    add_custom_food, and the entry schema read by tools._scale_entry);
  * the matching rules documented in the CA1 README (whole-word matching,
    typo tolerance >= 0.8 similarity for words of 4+ letters, "most
    specific alias wins", custom foods persisted to custom_foods.json);
  * numbers recovered from CA1's own saved evidence (notebook cell outputs and
    web-UI screenshots).

Every entry carries a "provenance" field so nothing is presented as more
trustworthy than it is:

  "ca1_recovered"            Exact value back-calculated from CA1 evidence and
                             verified to reproduce every recorded CA1 result.
  "reference_reconstructed"  NOT recoverable from CA1. Typical published
                             reference values (USDA FoodData Central style,
                             Indian foods per common IFCT-style figures),
                             written from reference knowledge and NOT
                             machine-verified against the source. Treat as
                             estimates; review before relying on them.

If you still have the original food_database.py from CA1, prefer it.

Entry schema (consumed by tools._scale_entry):
    calories_per_unit : kcal for ONE base_amount of `unit`
    protein/carbs/fat : grams for the same base_amount
    unit              : "piece" | "slice" | "g" | "ml"
    base_amount       : how many `unit`s the numbers above are FOR
    serving_size      : human-readable description
    canonical_name    : display name (defaults to the dict key)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import threading
import warnings
from difflib import SequenceMatcher
from typing import Optional

from ..paths import data_dir
from ..state.io import atomic_write_json, user_file_lock

_USER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


# Custom foods live in the project data/ folder (see ../paths.py). The
# NUTRITRACK_CUSTOM_FOODS environment variable redirects just this file
# (used by tests and deployments) without touching code.
def _custom_path() -> str:
    override = os.environ.get("NUTRITRACK_CUSTOM_FOODS", "").strip()
    return override if override else str(data_dir() / "custom_foods.json")


def custom_path_for_user(user_id: Optional[str] = None) -> str:
    """Return the custom-foods JSON path scoped to `user_id` (isolating non-demo users per user)."""
    base_custom = _custom_path()
    if user_id is None or user_id == "demo_user":
        return base_custom
    if not isinstance(user_id, str) or not _USER_ID_RE.match(user_id):
        raise ValueError("user_id must be 1-64 characters: letters, digits, '_' or '-'.")
    override = os.environ.get("NUTRITRACK_CUSTOM_FOODS", "").strip()
    root = Path(override).parent if override else data_dir()
    return str(root / "custom_foods" / f"{user_id}.json")


# ---------------------------------------------------------------------------
# Built-in foods
# key: (aliases, unit, base_amount, serving_size, kcal, protein, carbs, fat, provenance)
# ---------------------------------------------------------------------------
_R = "ca1_recovered"
_X = "reference_reconstructed"

_BUILTIN: dict[str, tuple] = {
    # ---- recovered exactly from CA1 evidence ---------------------------
    "egg": (["egg", "eggs", "boiled egg", "boiled eggs", "anda"],
            "piece", 1, "1 large egg (~50 g)", 78, 6.0, 0.6, 5.3, _R),
    "milk": (["milk", "full cream milk", "whole milk", "doodh"],
             "ml", 240, "1 cup (240 ml), whole milk", 149, 8.0, 12.0, 8.0, _R),
    "oats": (["oats", "oat", "oatmeal", "rolled oats", "muesli"],
             "g", 100, "100 g dry oats", 370, 13.5, 66.5, 6.5, _R),
    "paneer": (["paneer"],
               "g", 100, "100 g paneer", 265, 18.3, 1.2, 20.8, _R),
    "chicken breast": (["chicken breast", "chicken breasts"],
                       "g", 100, "100 g cooked chicken breast", 165, 31.0, 0.0, 3.6, _R),
    "bread": (["bread", "bread slice", "bread slices", "toast"],
              "slice", 1, "1 slice of bread (~28 g)", 79, 2.7, 15.0, 1.0, _R),
    "roti": (["roti", "rotis", "chapati", "chapatis", "chapathi", "phulka"],
             "piece", 1, "1 medium roti/chapati", 104, 3.1, 18.0, 2.7, _R),
    "curd": (["curd", "dahi", "yogurt", "yoghurt"],
             "g", 100, "100 g curd", 98, 11.0, 3.4, 4.3, _R),
    "apple": (["apple", "apples"],
              "piece", 1, "1 medium apple", 95, 0.5, 25.0, 0.3, _R),
    "banana": (["banana", "bananas", "kela"],
               "piece", 1, "1 medium banana", 105, 1.3, 27.0, 0.4, _R),
    "soya chunks": (["soya chunks", "soya chunk", "soy chunks", "soy chunk",
                     "soybean chunks", "nutrela"],
                    "g", 100, "100 g dry soya chunks", 345, 52.0, 33.0, 0.5, _R),
    "spinach": (["spinach", "palak"],
                "g", 100, "100 g spinach", 23, 2.9, 3.6, 0.4, _X),  # kcal recovered, macros reference

    # ---- NOT recoverable from CA1: reference-reconstructed --------------
    "chicken leg": (["chicken leg", "chicken legs", "chicken drumstick", "drumstick chicken"],
                    "g", 100, "100 g roasted chicken leg meat (no skin)", 172, 28.3, 0.0, 5.7, _X),
    "skimmed milk": (["skimmed milk", "skim milk", "toned milk", "nonfat milk"],
                     "ml", 240, "1 cup (240 ml), skimmed milk", 83, 8.3, 12.2, 0.2, _X),
    "cheese": (["cheese", "cheddar", "cheddar cheese"],
               "g", 100, "100 g cheddar cheese", 403, 24.9, 1.3, 33.1, _X),
    "butter": (["butter"],
               "g", 100, "100 g butter", 717, 0.9, 0.1, 81.1, _X),
    "rice": (["rice", "white rice", "cooked rice", "chawal"],
             "g", 100, "100 g cooked white rice", 130, 2.7, 28.2, 0.3, _X),
    "brown rice": (["brown rice"],
                   "g", 100, "100 g cooked brown rice", 123, 2.7, 25.6, 1.0, _X),
    "poha": (["poha", "flattened rice", "beaten rice"],
             "g", 100, "100 g dry poha", 346, 6.6, 77.3, 1.2, _X),
    "dal": (["dal", "dhal", "daal", "lentils", "lentil", "toor dal", "moong dal"],
            "g", 100, "100 g cooked dal/lentils", 116, 9.0, 20.1, 0.4, _X),
    "rajma": (["rajma", "kidney beans", "kidney bean"],
              "g", 100, "100 g cooked kidney beans", 127, 8.7, 22.8, 0.5, _X),
    "chickpeas": (["chickpeas", "chickpea", "chana", "chole", "garbanzo", "garbanzo beans"],
                  "g", 100, "100 g cooked chickpeas", 164, 8.9, 27.4, 2.6, _X),
    "orange": (["orange", "oranges", "santra"],
               "piece", 1, "1 medium orange (~131 g)", 62, 1.2, 15.4, 0.2, _X),
    "mango": (["mango", "mangos", "mangoes", "aam"],
              "piece", 1, "1 medium mango (~200 g edible flesh)", 120, 1.6, 30.0, 0.8, _X),
    "grapes": (["grapes", "grape"],
               "g", 100, "100 g grapes", 69, 0.7, 18.1, 0.2, _X),
    "potato": (["potato", "potatoes", "aloo"],
               "piece", 1, "1 medium baked potato (~173 g)", 161, 4.3, 36.6, 0.2, _X),
    "sweet potato": (["sweet potato", "sweet potatoes", "shakarkandi"],
                     "piece", 1, "1 medium baked sweet potato (~114 g)", 103, 2.3, 23.6, 0.2, _X),
    "broccoli": (["broccoli"],
                 "g", 100, "100 g broccoli", 34, 2.8, 6.6, 0.4, _X),
    "carrot": (["carrot", "carrots", "gajar"],
               "piece", 1, "1 medium carrot (~61 g)", 25, 0.6, 5.8, 0.1, _X),
    "tomato": (["tomato", "tomatoes", "tamatar"],
               "piece", 1, "1 medium tomato (~123 g)", 22, 1.1, 4.8, 0.2, _X),
    "onion": (["onion", "onions", "pyaz"],
              "piece", 1, "1 medium onion (~110 g)", 44, 1.2, 10.3, 0.1, _X),
    "almonds": (["almonds", "almond", "badam"],
                "g", 100, "100 g almonds", 579, 21.2, 21.6, 49.9, _X),
    "peanuts": (["peanuts", "peanut", "moongphali"],
                "g", 100, "100 g peanuts", 567, 25.8, 16.1, 49.2, _X),
    "walnuts": (["walnuts", "walnut", "akhrot"],
                "g", 100, "100 g walnuts", 654, 15.2, 13.7, 65.2, _X),
    "whey protein": (["whey protein", "whey", "protein shake"],
                     "scoop", 1, "1 scoop (~30 g), typical; varies by brand", 120, 24.0, 3.0, 1.5, _X),
    "orange juice": (["orange juice"],
                     "ml", 240, "1 cup (240 ml) orange juice", 112, 1.7, 25.8, 0.5, _X),
    "green tea": (["green tea"],
                  "ml", 240, "1 cup (240 ml) brewed, unsweetened", 2, 0.5, 0.0, 0.0, _X),
}

# Compound phrases that CONTAIN a database word but are nutritionally a
# different food (e.g. "peanut butter" must not be logged as butter or peanuts).
# find_food() returns None for these so the caller asks the user instead of
# silently logging the wrong food. Added during reconstruction.
_NOT_SAME_AS: tuple[str, ...] = (
    "peanut butter", "almond butter", "butter chicken", "butter paneer",
    "almond milk", "soy milk", "soya milk", "coconut milk", "oat milk",
    "rice milk", "milk shake", "milkshake", "fried rice", "rice pudding",
    "egg roll", "chicken curry", "paneer butter masala", "fried egg",
    "fried chicken", "potato chips", "potato chip", "french fries",
    "apple juice", "apple pie", "banana bread", "banana chips",
    "chicken roll", "bread pakora", "bread pudding", "cheese cake",
    "cheesecake", "cheese burger", "cheeseburger",
)


_BUILTIN_FIBER_PER_BASE: dict[str, float] = {
    "egg": 0.0,
    "milk": 0.0,
    "oats": 10.1,
    "paneer": 0.0,
    "chicken breast": 0.0,
    "bread": 1.9,
    "roti": 2.7,
    "curd": 0.0,
    "apple": 4.4,
    "banana": 3.1,
    "soya chunks": 13.0,
    "spinach": 2.2,
    "chicken leg": 0.0,
    "skimmed milk": 0.0,
    "cheese": 0.0,
    "butter": 0.0,
    "rice": 0.4,
    "brown rice": 1.8,
    "poha": 0.7,
    "dal": 7.9,
    "rajma": 6.4,
    "chickpeas": 7.6,
    "orange": 3.1,
    "mango": 3.2,
    "grapes": 0.9,
    "potato": 3.8,
    "sweet potato": 3.8,
    "broccoli": 2.6,
    "carrot": 1.7,
    "tomato": 1.5,
    "onion": 1.9,
    "almonds": 12.5,
    "peanuts": 8.5,
    "walnuts": 6.7,
    "whey protein": 0.0,
    "orange juice": 0.5,
    "green tea": 0.0,
}


def _build_food_db() -> dict[str, dict]:
    db: dict[str, dict] = {}
    for key, (aliases, unit, base, serving, kcal, p, c, f, prov) in _BUILTIN.items():
        db[key] = {
            "canonical_name": key,
            "aliases": list(aliases),
            "unit": unit,
            "base_amount": base,
            "serving_size": serving,
            "calories_per_unit": kcal,
            "protein": p,
            "carbs": c,
            "fat": f,
            "fiber": float(_BUILTIN_FIBER_PER_BASE.get(key, 0.0)),
            "provenance": prov,
            "custom": False,
        }
    return db


_BUILTIN_DB: dict[str, dict] = _build_food_db()
FOOD_DB: dict[str, dict] = _build_food_db()
_USER_CUSTOM_CACHE_LOCK = threading.Lock()
_USER_CUSTOM_CACHE: dict[str, tuple[tuple[int, int] | None, dict[str, dict]]] = {}


def _stat_sig(path_str: str) -> Optional[tuple[int, int]]:
    try:
        st = os.stat(path_str)
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def _parse_custom_entries(raw_dict: dict) -> dict[str, dict]:
    parsed: dict[str, dict] = {}
    for key, entry in raw_dict.items():
        try:
            parsed[key] = {
                "canonical_name": entry.get("canonical_name", key),
                "aliases": entry.get("aliases", [key]),
                "unit": entry["unit"],
                "base_amount": float(entry["base_amount"]),
                "serving_size": entry.get("serving_size", "1 serving"),
                "calories_per_unit": float(entry["calories_per_unit"]),
                "protein": float(entry["protein"]),
                "carbs": float(entry["carbs"]),
                "fat": float(entry["fat"]),
                "fiber": float(entry.get("fiber", 0.0) or 0.0),
                "provenance": entry.get("provenance", "user_defined"),
                "custom": True,
                "portions": {str(u): float(g) for u, g in (entry.get("portions") or {}).items()},
            }
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            warnings.warn(f"Skipping malformed custom food '{key}': {exc}")
    return parsed


def _load_user_custom_entries(user_id: str) -> dict[str, dict]:
    path = custom_path_for_user(user_id)
    sig = _stat_sig(path)
    with _USER_CUSTOM_CACHE_LOCK:
        cached = _USER_CUSTOM_CACHE.get(path)
        if cached is not None and cached[0] == sig:
            return cached[1]
    if not os.path.exists(path):
        with _USER_CUSTOM_CACHE_LOCK:
            _USER_CUSTOM_CACHE[path] = (None, {})
        return {}
    lock_path = f"{path}.lock"
    with user_file_lock(lock_path):
        sig = _stat_sig(path)
        raw = _read_custom_file_raw(path)
        parsed = _parse_custom_entries(raw)
        with _USER_CUSTOM_CACHE_LOCK:
            if len(_USER_CUSTOM_CACHE) > 256:
                _USER_CUSTOM_CACHE.clear()
            _USER_CUSTOM_CACHE[path] = (sig, parsed)
        return parsed


def get_food_db(user_id: Optional[str] = None) -> dict[str, dict]:
    """Return built-in foods plus ONLY `user_id`'s own custom foods (or global FOOD_DB when user_id is None)."""
    if user_id is None:
        return FOOD_DB
    merged = dict(_BUILTIN_DB)
    merged.update(_load_user_custom_entries(user_id))
    return merged


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[a-z0-9']+")
_FUZZY_MIN_LEN = 4        # only words of 4+ letters get typo tolerance
_FUZZY_THRESHOLD = 0.8    # SequenceMatcher ratio (as documented in the CA1 README)


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower().replace("\u2019", "'"))


def _matched_query_word(
    alias_word: str,
    query_words: list[str],
    *,
    exact_vocab: Optional[set[str]] = None,
) -> Optional[str]:
    """Return the query word that satisfies `alias_word` (exact, or a typo-tolerant
    match for words of 4+ letters), else None."""
    if alias_word in query_words:
        return alias_word
    if len(alias_word) < _FUZZY_MIN_LEN:
        return None
    for q in query_words:
        if len(q) < _FUZZY_MIN_LEN or q[0] != alias_word[0]:
            # Same first letter required: keeps real typos ("protien",
            # "chiken") but blocks look-alikes such as "maple" -> "apple".
            continue
        # Never fuzzy-remap an exact known food word (e.g. "chicken") onto an unrelated food ("chickpea")
        if exact_vocab and q in exact_vocab:
            continue
        if SequenceMatcher(None, alias_word, q).ratio() >= _FUZZY_THRESHOLD:
            return q
    return None


# Words that may accompany a food name without changing WHICH food it is.
# Anything else left over after matching (a brand, "fried", "spicy", ...) means
# the match is only a guess -- see match_details().
FILLER_WORDS = frozenset({
    "a", "an", "the", "of", "some", "my", "few", "fresh", "raw", "boiled", "cooked", "plain",
    "whole", "large", "medium", "small", "big", "ripe", "hot", "cold", "steamed",
    "piece", "pieces", "slice", "slices", "cup", "cups", "serving", "servings",
})


def match_details(query: str, *, user_id: Optional[str] = None) -> dict:
    """
    Match free text to a food DB key for `user_id` and say how well it matched.

    Returns {"key": str | None, "matched_words": [...], "leftover_words": [...],
             "blocked_by": str | None}
    """
    result = {"key": None, "matched_words": [], "leftover_words": [], "blocked_by": None}
    if not query or not query.strip():
        return result

    lowered = " ".join(_tokens(query))
    for phrase in _NOT_SAME_AS:
        if re.search(rf"\b{re.escape(phrase)}\b", lowered):
            result["blocked_by"] = phrase
            return result

    db = get_food_db(user_id)
    qwords = _tokens(query)
    exact_vocab = {
        w
        for key, entry in db.items()
        for alias in (entry.get("aliases") or [key])
        for w in _tokens(alias)
    }
    best_key: Optional[str] = None
    best_score = (0, 0)
    best_matched: list[str] = []
    for key, entry in db.items():
        for alias in entry.get("aliases", [key]):
            awords = _tokens(alias)
            if not awords:
                continue
            matched = [_matched_query_word(w, qwords, exact_vocab=exact_vocab) for w in awords]
            if all(m is not None for m in matched):
                score = (len(awords), len(alias))
                if score > best_score:
                    best_key, best_score, best_matched = key, score, matched
    if best_key is None:
        return result
    result["key"] = best_key
    result["matched_words"] = sorted(set(best_matched))
    result["leftover_words"] = [w for w in qwords
                                if w not in best_matched and w not in FILLER_WORDS and not w.isdigit()]
    return result


def find_food(query: str, *, user_id: Optional[str] = None) -> Optional[str]:
    """
    Match free text such as "3 boiled eggs" to a food DB key for `user_id`.
    """
    return match_details(query, user_id=user_id)["key"]


# ---------------------------------------------------------------------------
# Custom foods (persisted, isolated per user)
# ---------------------------------------------------------------------------
def _write_custom_file(data: dict, path: Optional[str] = None) -> None:
    """Atomic write: temp file in the same directory, then os.replace()."""
    target = path or _custom_path()
    atomic_write_json(target, data)


def _load_custom_foods() -> None:
    """Merge default custom_foods.json into FOOD_DB for legacy single-user CA1 compatibility."""
    path = _custom_path()
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError("custom foods file must contain a JSON object")
    except Exception as exc:
        warnings.warn(f"Could not read custom foods from {path}: {exc}. "
                      "Built-in foods are unaffected.")
        return
    FOOD_DB.update(_parse_custom_entries(data))


def _read_custom_file_raw(path: Optional[str] = None) -> dict:
    target = path or _custom_path()
    if not os.path.exists(target):
        return {}
    try:
        with open(target, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def normalise_name(name: str) -> str:
    """The key a custom food is stored under (lower-cased, punctuation collapsed)."""
    return " ".join(_tokens(name)) or name.strip().lower()


def get_custom_food(name: str, *, user_id: Optional[str] = None) -> Optional[dict]:
    """Fetch a custom food owned by `user_id` (returns None for built-in or other users' foods)."""
    key = normalise_name(name)
    if user_id is None:
        entry = FOOD_DB.get(key)
        return dict(entry) if entry and entry.get("custom") else None
    custom_entries = _load_user_custom_entries(user_id)
    entry = custom_entries.get(key)
    return dict(entry) if entry else None


def add_custom_food(
    name: str,
    calories: float,
    protein: float,
    carbs: float,
    fat: float,
    serving_size: str = "1 serving",
    unit: str = "piece",
    base_amount: float = 1,
    portions: Optional[dict] = None,
    provenance: str = "user_defined",
    fiber: float = 0.0,
    user_id: Optional[str] = None,
) -> str:
    """
    Add (or replace) a user-defined food on disk (per-user when `user_id` is supplied)
    and return its key. Values are exactly what the user typed.
    """
    key = normalise_name(name)
    if not key:
        raise ValueError("Food name cannot be empty.")

    entry = {
        "canonical_name": key,
        "aliases": [key],
        "unit": unit,
        "base_amount": float(base_amount),
        "serving_size": serving_size,
        "calories_per_unit": float(calories),
        "protein": float(protein),
        "carbs": float(carbs),
        "fat": float(fat),
        "fiber": round(max(0.0, float(fiber or 0.0)), 1),
    }
    clean_portions = {str(u): float(g) for u, g in (portions or {}).items() if float(g) > 0}
    if clean_portions:
        entry["portions"] = clean_portions         # grams (or ml) in ONE of that household unit
    if provenance != "user_defined":
        entry["provenance"] = provenance

    target_path = custom_path_for_user(user_id)
    lock_path = f"{target_path}.lock"
    with user_file_lock(lock_path):
        stored = _read_custom_file_raw(target_path)
        stored[key] = entry
        _write_custom_file(stored, path=target_path)          # raises on failure -> caller learns the save failed
        with _USER_CUSTOM_CACHE_LOCK:
            _USER_CUSTOM_CACHE.pop(target_path, None)

    if user_id is None:
        FOOD_DB[key] = {**entry, "provenance": provenance, "custom": True, "portions": clean_portions}
    return key


def delete_custom_food(name: str, *, user_id: str) -> bool:
    """Delete a custom food belonging strictly to `user_id`. Returns True if deleted, False if not found."""
    key = normalise_name(name)
    if not key or key in _BUILTIN_DB:
        return False
    target_path = custom_path_for_user(user_id)
    lock_path = f"{target_path}.lock"
    with user_file_lock(lock_path):
        if not os.path.exists(target_path):
            return False
        stored = _read_custom_file_raw(target_path)
        if key not in stored:
            return False
        del stored[key]
        _write_custom_file(stored, path=target_path)
        with _USER_CUSTOM_CACHE_LOCK:
            _USER_CUSTOM_CACHE.pop(target_path, None)
    return True


_load_custom_foods()
