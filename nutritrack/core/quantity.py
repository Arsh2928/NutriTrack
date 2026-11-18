"""
quantity.py
-----------
Turn free text such as "3 boiled eggs", "250gm curd" or "two roti" into a
validated (food_text, quantity, unit). Pure Python, no I/O.

Fixes CA1 audit defects D2/D3: CA1 silently ignored a minus sign, accepted zero
and absurd amounts, read "2 curd" as 2 grams, and treated "two eggs" as ONE egg.
Anything it cannot read with confidence is reported as a *problem* so the caller
can ask the user, never guessed.
"""
import re
from dataclasses import dataclass, field
from datetime import date

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "half": 0.5, "dozen": 12,
}

UNIT_ALIASES = {
    "g": "g", "gm": "g", "gms": "g", "gram": "g", "grams": "g",
    "kg": "kg", "kgs": "kg", "kilo": "kg", "kilos": "kg", "kilogram": "kg", "kilograms": "kg",
    "ml": "ml", "millilitre": "ml", "millilitres": "ml", "milliliter": "ml", "milliliters": "ml",
    "l": "l", "litre": "l", "litres": "l", "liter": "l", "liters": "l",
    "cup": "cup", "cups": "cup",
    "tbsp": "tbsp", "tablespoon": "tbsp", "tablespoons": "tbsp",
    "tsp": "tsp", "teaspoon": "tsp", "teaspoons": "tsp",
    "piece": "piece", "pieces": "piece", "pc": "piece", "pcs": "piece",
    "slice": "slice", "slices": "slice",
    "scoop": "scoop", "scoops": "scoop",
    "serving": "serving", "servings": "serving",
}
# Household measures whose size we cannot know without asking.
HOUSEHOLD_UNITS = {
    "glass", "glasses", "bowl", "bowls", "plate", "plates", "katori", "katoris", "handful",
    "handfuls", "packet", "packets", "pack", "packs", "can", "cans", "bottle", "bottles",
    "spoon", "spoons", "bar", "bars", "sachet", "sachets", "cupful", "cupfuls",
}
COUNT_UNITS = {"piece", "slice", "scoop", "serving"}
TO_BASE = {"g": 1, "kg": 1000, "ml": 1, "l": 1000, "cup": 240, "tbsp": 15, "tsp": 5}  # grams or ml

MAX_COUNT = 50          # more than 50 pieces/slices/scoops of one food in one entry is almost surely a typo
MAX_AMOUNT = 5000       # more than 5 kg / 5 L of one food in one entry likewise

MEAL_TYPES = {"breakfast": "breakfast", "brunch": "brunch", "lunch": "lunch",
              "dinner": "dinner", "supper": "dinner", "snack": "snack", "snacks": "snack"}


@dataclass
class ParsedItem:
    raw: str
    food_text: str = ""
    value: "float | None" = None
    unit: "str | None" = None
    household_unit: "str | None" = None
    explicit_value: bool = False
    problems: list = field(default_factory=list)       # [{"code":..., "message":...}]

    def as_dict(self):
        return {"raw": self.raw, "food": self.food_text, "quantity": self.value, "unit": self.unit,
                "household_unit": self.household_unit, "quantity_explicit": self.explicit_value,
                "problems": list(self.problems)}


def _to_float(token):
    token = " ".join(token.split())
    if " " in token and "/" in token:
        whole_part, frac_part = token.split(" ", 1)
        num, den = frac_part.replace(" ", "").split("/")
        return float(whole_part) + (float(num) / float(den) if float(den) else 0.0)
    token = token.replace(" ", "")
    if "/" in token:
        num, den = token.split("/")
        return float(num) / float(den) if float(den) else 0.0
    return float(token)


def plausibility_problem(value, unit):
    """Return a problem dict if (value, unit) is zero/negative/absurd, else None."""
    if value is None:
        return None
    if value <= 0:
        return {"code": "invalid_quantity",
                "message": f"The quantity must be greater than zero (got {value:g})."}
    if unit in COUNT_UNITS and value > MAX_COUNT:
        return {"code": "implausible_quantity",
                "message": f"{value:g} {unit}s of one food in a single entry looks like a typo "
                           f"(the limit is {MAX_COUNT}). Please re-enter the amount."}
    if unit in TO_BASE and value * TO_BASE[unit] > MAX_AMOUNT:
        return {"code": "implausible_quantity",
                "message": f"{value:g} {unit} of one food in a single entry looks like a typo "
                           f"(the limit is about {MAX_AMOUNT / 1000:g} kg / L). Please re-enter the amount."}
    return None


def parse_item_text(text):
    """Parse ONE food phrase. Never raises; problems are returned in .problems."""
    item = ParsedItem(raw=str(text))
    t = " ".join(str(text).lower().replace("\u00bd", "0.5").replace("\u2019", "'").split())
    t = t.strip(" .,!?;:")

    neg = re.match(r"^[-\u2212\u2013]\s*(?=\d)", t)
    if neg:
        item.problems.append({"code": "invalid_quantity",
                              "message": "A quantity cannot be negative. Please give a positive amount."})
        t = t[neg.end():]

    rest = t
    m = re.match(r"^(\d+\s+\d+\s*/\s*\d+|\d+\s*/\s*\d+|\d+(?:\.\d+)?|\.\d+)", t)
    split_bad_token = False
    if m:
        item.value, item.explicit_value = _to_float(m.group(1)), True
        tail_raw = t[m.end():]
        if tail_raw and tail_raw[0].isalpha():
            alnum_m = re.match(r"^([a-z0-9]+)", tail_raw)
            attached = alnum_m.group(1) if alnum_m else ""
            if attached not in UNIT_ALIASES and attached not in HOUSEHOLD_UNITS and attached != "x":
                split_bad_token = True
        rest = tail_raw.strip()
    else:
        words = t.split()
        if words and (words[0] in NUMBER_WORDS or words[0] in ("a", "an")):
            base = 1.0 if words[0] in ("a", "an") else float(NUMBER_WORDS[words[0]])
            used = 1
            if len(words) > 1 and words[1] == "dozen" and words[0] != "dozen":
                base, used = base * 12, 2
            item.value, item.explicit_value = base, True
            rest = " ".join(words[used:])
        else:
            mo = re.match(r"^([a-z]+)\s+of\s+(.+)$", t)        # "cup of milk", "glass of milk"
            if mo and (mo.group(1) in UNIT_ALIASES or mo.group(1) in HOUSEHOLD_UNITS):
                item.value, item.explicit_value = 1.0, True
                rest = t

    leading = item.explicit_value
    if not leading:
        # Trailing quantity: "paneer 100g", "spinach 50 g", "eggs x3".
        # Only accepted when what follows the number is a known unit or nothing,
        # so "qwxyz123abc" or "vitamin b12 pills" are not misread.
        mt = re.match(
            r"^(?P<food>.*?[a-z])\s*(?:x\s*)?(?P<num>\d+\s+\d+\s*/\s*\d+|\d+\s*/\s*\d+|\d+(?:\.\d+)?)\s*(?P<unit>[a-z]+)?$",
            t,
        )
        if mt and (mt.group("unit") is None or mt.group("unit") in UNIT_ALIASES
                   or mt.group("unit") in HOUSEHOLD_UNITS):
            item.value, item.explicit_value = _to_float(mt.group("num")), True
            rest = mt.group("food")
            word = mt.group("unit")
            if word in UNIT_ALIASES:
                item.unit = UNIT_ALIASES[word]
            elif word in HOUSEHOLD_UNITS:
                item.household_unit = word
            rest = re.sub(r"^(?:a|an|the|some|my|of)\s+", "", rest.strip())
    if leading:
        rest = re.sub(r"^x\s+", "", rest)                       # "2 x eggs" / "2x eggs"
        mu = re.match(r"^([a-z]+)\b\s*(?:of\b\s*)?", rest)
        if mu:
            word = mu.group(1)
            if word in UNIT_ALIASES:
                item.unit, rest = UNIT_ALIASES[word], rest[mu.end():]
            elif word in HOUSEHOLD_UNITS:
                item.household_unit, rest = word, rest[mu.end():]
        rest = re.sub(r"^(?:a|an|the|some|my|of)\s+", "", rest.strip())

    item.food_text = rest.strip()

    # A quantity-shaped leftover token ("e3" from "1e3 eggs", "x3", "10g", "b12") means we
    # could not read the amount, while alphanumeric food names like "7up" (or "omega 3") are allowed.
    def _has_ambiguous_qty_token(food_str: str) -> bool:
        if split_bad_token:
            return True
        for tok in food_str.split():
            if tok == "7up":
                continue
            m_tok = re.fullmatch(r"([a-z]{0,2})(\d+(?:\.\d+)?)([a-z]{0,2})", tok)
            if not m_tok:
                continue
            prefix, _num, suffix = m_tok.groups()
            if not prefix and len(suffix) >= 2 and suffix not in UNIT_ALIASES and suffix not in HOUSEHOLD_UNITS:
                # e.g. "7up", "5pc"? "pc" IS in UNIT_ALIASES so "5pc" is caught as ambiguous_quantity!
                continue
            return True
        return False

    if _has_ambiguous_qty_token(item.food_text):
        item.problems.append({"code": "ambiguous_quantity",
                              "message": f"I couldn't read the amount in '{item.raw}'. "
                                         "Please write it like '2 eggs' or '150 g paneer'."})
    if not item.food_text:
        item.problems.append({"code": "no_food_name",
                              "message": f"I couldn't find a food name in '{item.raw}'."})

    if item.unit is not None:
        problem = plausibility_problem(item.value, item.unit)
        if problem:
            item.problems.append(problem)
    elif item.value is not None and item.value <= 0 and not any(
            p["code"] == "invalid_quantity" for p in item.problems):
        item.problems.append(plausibility_problem(item.value, None))
    return item


# ---------------------------------------------------------------------------
# Splitting a whole sentence into food phrases
# ---------------------------------------------------------------------------
_SEPARATORS = re.compile(r"\s*,\s*|\s*;\s*|\s+and\s+|\s+with\s+|\s*\+\s*|\s+&\s+|\s+plus\s+", re.I)
_MEAL_WORD = re.compile(r"(?:\bfor\s+)?(?:\bmy\s+)?\b(breakfast|brunch|lunch|dinner|supper|snacks?)\b\s*(?:was|is|:|-)?", re.I)
_LEAD = re.compile(
    r"^(?:please\s+|can you\s+|could you\s+)?"
    r"(?:(?:i|we)(?:\s*'ve|\s+have|\s+had)?\s+)?(?:just\s+)?"
    r"(?:(?:ate|eaten|eat|had|have|consumed|drank|add(?:ed)?|log(?:ged)?|record(?:ed)?|track|enter(?:ed)?)\b[:,]?\s*)+"
    r"(?:to\s+my\s+(?:log|diary)\s*[:,]?\s*)?", re.I)
_TRAILING_TIME = re.compile(r"\b(?:this morning|this afternoon|this evening|tonight|just now|earlier)\b", re.I)
_ISO_DATE_TOKEN = re.compile(r"(?:\b(?:on|for|dated?)\s+)?\b(\d{4})-(\d{1,2})-(\d{1,2})\b(?:'s)?", re.I)
_SLASH_DATE_TOKEN = re.compile(r"(?:\b(?:on|for|dated?)\s+)?\b(\d{1,4})/(\d{1,2})/(\d{1,4})\b(?:'s)?", re.I)
_REL_DATE_TOKEN = re.compile(r"(?:\b(?:on|for)\s+)?\b(yesterday|today)\b(?:'s)?", re.I)


def extract_date_from_text(text: str) -> tuple[str | None, str | None, str]:
    """Extract & strip explicit ISO (`YYYY-MM-DD`), slash (`DD/MM/YYYY` / `YYYY/MM/DD`), or relative (`today`/`yesterday`) dates.

    Returns `(date_hint, invalid_date_error, cleaned_text)`. Never misreads fractions like `1/2 cup` or food names like `7up`.
    """
    t = " ".join(str(text).split())
    date_hint: str | None = None
    invalid_err: str | None = None

    def _check_iso(y: int, m: int, d: int, raw: str) -> None:
        nonlocal date_hint, invalid_err
        try:
            date_hint = date(y, m, d).isoformat()
        except ValueError:
            invalid_err = f"'{raw}' is not a valid calendar date."

    m_iso = _ISO_DATE_TOKEN.search(t)
    if m_iso:
        y, mo, d = int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3))
        _check_iso(y, mo, d, f"{y:04d}-{mo:02d}-{d:02d}")
        t = (t[:m_iso.start()] + " " + t[m_iso.end():]).strip(" ,:-")
    else:
        m_slash = _SLASH_DATE_TOKEN.search(t)
        if m_slash:
            a, b, c = m_slash.group(1), m_slash.group(2), m_slash.group(3)
            raw_token = f"{a}/{b}/{c}"
            ai, bi, ci = int(a), int(b), int(c)
            if len(a) == 4:
                _check_iso(ai, bi, ci, raw_token)
            elif len(c) == 4:
                if ai > 12:
                    _check_iso(ci, bi, ai, raw_token)
                elif bi > 12:
                    _check_iso(ci, ai, bi, raw_token)
                else:
                    _check_iso(ci, bi, ai, raw_token)
            if date_hint is not None or invalid_err is not None:
                t = (t[:m_slash.start()] + " " + t[m_slash.end():]).strip(" ,:-")
        else:
            m_rel = _REL_DATE_TOKEN.search(t)
            if m_rel:
                date_hint = m_rel.group(1).lower()
                t = (t[:m_rel.start()] + " " + t[m_rel.end():]).strip(" ,:-")

    t = _TRAILING_TIME.sub("", t)
    t = " ".join(t.split()).strip(" ,:-")
    return date_hint, invalid_err, t


def split_meal_text_with_meta(text):
    """-> (meal_type | None, date_hint | None, invalid_date_error | None, [food phrases]). Never raises."""
    date_hint, invalid_err, t = extract_date_from_text(text)

    meal_type = None
    mm = _MEAL_WORD.search(t)
    if mm:
        meal_type = MEAL_TYPES[mm.group(1).lower()]
        t = (t[:mm.start()] + " " + t[mm.end():]).strip()
    t = " ".join(t.split()).strip(" ,")
    for _ in range(2):                       # "I just had", "please add my ..." etc.
        t = _LEAD.sub("", t).strip(" :,")
        if t.startswith("- "):
            t = t[2:].strip()
    t = re.sub(r"^(?:my|the)\s+", "", t, flags=re.I)

    phrases = [re.sub(r"^and\s+", "", p.strip(" .,!?;:"), flags=re.I) for p in _SEPARATORS.split(t)]
    phrases = [p for p in phrases if p and not re.fullmatch(r"(?:for|and|with|my|the|a|an|on)", p, re.I)]
    return meal_type, date_hint, invalid_err, phrases


def split_meal_text(text):
    """-> (meal_type | None, date_hint | None, [food phrases]). Never raises."""
    meal_type, date_hint, _, phrases = split_meal_text_with_meta(text)
    return meal_type, date_hint, phrases


# A phrase that asks for something instead of naming a food ("calculate how much
# protein I have consumed"). The intake tool reports these as ignored so the
# manager can route them to the right specialist.
NON_FOOD_REQUEST = re.compile(
    r"\b(calculate|how much|how many|what|show|tell|check|recommend|suggest|total|summary|summari[sz]e|"
    r"compare|analy[sz]e|remaining|left|budget|target|goal)\b", re.I)


def is_non_food_request(phrase):
    return bool(NON_FOOD_REQUEST.search(phrase))
