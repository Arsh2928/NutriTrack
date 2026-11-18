"""
recommendation_data.py
----------------------
CLASSIFICATION and PORTION data for the Meal Recommendation Agent.

This module contains NO nutrition values. It only says
  * what role a food plays in a meal (protein / carb / veg ...),
  * which portion sizes are sensible to suggest,
  * which diets a food is not allowed in.
Every calorie and macro figure is still scaled from the food database, so
the values shown to the user are always the database's. The portion lists are
engineering assumptions ("suggested portions"), documented here so they can be
reviewed and changed in one place.
"""

ROLES = {
    "egg": "protein", "chicken breast": "protein", "chicken leg": "protein", "paneer": "protein",
    "soya chunks": "protein", "whey protein": "protein",
    "dal": "legume", "rajma": "legume", "chickpeas": "legume",
    "rice": "carb", "brown rice": "carb", "roti": "carb", "bread": "carb", "poha": "carb", "oats": "carb",
    "potato": "carb", "sweet potato": "carb",
    "banana": "fruit", "apple": "fruit", "orange": "fruit", "mango": "fruit", "grapes": "fruit",
    "spinach": "veg", "broccoli": "veg", "carrot": "veg", "tomato": "veg", "onion": "veg",
    "milk": "dairy", "skimmed milk": "dairy", "curd": "dairy", "cheese": "dairy",
    "almonds": "fat", "peanuts": "fat", "walnuts": "fat",
    # not suggested on their own: butter, orange juice, green tea
}

MEAT = frozenset({"chicken breast", "chicken leg"})
EGG = frozenset({"egg"})


def _n(unit, *quantities):
    return [(q, unit) for q in quantities]


LADDERS = {
    "egg": _n("piece", 1, 2, 3, 4),
    "chicken breast": _n("g", 100, 150, 200), "chicken leg": _n("g", 100, 150, 200),
    "paneer": _n("g", 50, 100, 150), "soya chunks": _n("g", 30, 50), "whey protein": _n("scoop", 1, 2),
    "dal": _n("g", 150, 200, 250), "rajma": _n("g", 150, 200), "chickpeas": _n("g", 100, 150, 200),
    "rice": _n("g", 100, 150, 200), "brown rice": _n("g", 100, 150, 200),
    "roti": _n("piece", 1, 2, 3), "bread": _n("slice", 1, 2, 3), "poha": _n("g", 30, 50),
    "oats": _n("g", 30, 40, 50), "potato": _n("piece", 1), "sweet potato": _n("piece", 1),
    "banana": _n("piece", 1, 2), "apple": _n("piece", 1, 2), "orange": _n("piece", 1, 2),
    "mango": _n("piece", 1), "grapes": _n("g", 100, 150),
    "spinach": _n("g", 50, 100), "broccoli": _n("g", 100, 150), "carrot": _n("piece", 1, 2),
    "tomato": _n("piece", 1, 2), "onion": _n("piece", 1),
    "milk": _n("ml", 200, 250, 300), "skimmed milk": _n("ml", 200, 250, 300),
    "curd": _n("g", 100, 150, 200), "cheese": _n("g", 20, 30, 40),
    "almonds": _n("g", 10, 20, 30), "peanuts": _n("g", 10, 20, 30), "walnuts": _n("g", 10, 20, 30),
}

# Meal shapes: which roles are combined (one food from each role).
TEMPLATES = (
    ("protein",), ("protein", "carb"), ("protein", "veg"), ("protein", "carb", "veg"), ("protein", "fruit"),
    ("legume",), ("legume", "carb"),
    ("dairy",), ("dairy", "fruit"), ("carb", "dairy"), ("fruit", "fat"),
    ("other",),                                     # the user's own custom foods, on their own
)


def fallback_ladder(entry):
    """Portions for a food that has no entry in LADDERS (the user's custom foods)."""
    unit, base = entry["unit"], entry["base_amount"]
    if unit in ("piece", "slice", "scoop"):
        return [(1, unit), (2, unit)]
    return [(base, unit)]


def diet_allows(diet, food_key):
    if diet == "vegetarian":
        return food_key not in MEAT and food_key not in EGG
    if diet == "eggetarian":
        return food_key not in MEAT
    return True
