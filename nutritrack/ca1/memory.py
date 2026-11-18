"""
memory.py
---------
Session/conversation memory for the Meal Calorie Helper agent.

The agent stores all state here programmatically — it does NOT rely on the
language model remembering text from earlier turns.

State stored:
    - daily_budget     : kcal target set by the user
    - meals            : list of logged meal entries (each entry is a dict)
    - total_calories   : running kcal sum
    - total_protein    : running protein sum (g)
    - total_carbs      : running carbs sum (g)
    - total_fat        : running fat sum (g)
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data class for a single meal entry
# ---------------------------------------------------------------------------
@dataclass
class MealEntry:
    food: str
    quantity_str: str          # human-readable e.g. "3 pieces" / "200 g"
    calories: float
    protein: float
    carbs: float
    fat: float

    def to_dict(self) -> dict:
        return {
            "food": self.food,
            "quantity": self.quantity_str,
            "calories": round(self.calories, 1),
            "protein": round(self.protein, 1),
            "carbs": round(self.carbs, 1),
            "fat": round(self.fat, 1),
        }


# ---------------------------------------------------------------------------
# Session Memory
# ---------------------------------------------------------------------------
class SessionMemory:
    """
    Holds all state for the current conversation/session.

    Usage:
        memory = SessionMemory()
        memory.set_budget(2200)
        memory.add_meal(MealEntry(...))
        print(memory.remaining_calories())
    """

    def __init__(self) -> None:
        self.daily_budget: Optional[float] = None
        self.meals: list[MealEntry] = []

    # ------------------------------------------------------------------
    # Budget management
    # ------------------------------------------------------------------
    def set_budget(self, budget_kcal: float) -> None:
        """Set the daily calorie budget."""
        self.daily_budget = float(budget_kcal)

    def has_budget(self) -> bool:
        """Return True if a budget has been set."""
        return self.daily_budget is not None

    # ------------------------------------------------------------------
    # Totals -- always recomputed from the meal list (audit defect D6: the old
    # incrementally-updated totals drifted after many add/remove cycles).
    # ------------------------------------------------------------------
    @property
    def _total_calories(self) -> float:
        return math.fsum(m.calories for m in self.meals)

    @property
    def _total_protein(self) -> float:
        return math.fsum(m.protein for m in self.meals)

    @property
    def _total_carbs(self) -> float:
        return math.fsum(m.carbs for m in self.meals)

    @property
    def _total_fat(self) -> float:
        return math.fsum(m.fat for m in self.meals)

    # ------------------------------------------------------------------
    # Meal management
    # ------------------------------------------------------------------
    def add_meal(self, entry: MealEntry) -> None:
        """Append a meal. (Totals are computed from self.meals on demand.)"""
        self.meals.append(entry)

    def reset_meals(self) -> None:
        """Clear all meals for a fresh day."""
        self.meals.clear()

    def remove_meal(self, index: int) -> Optional[MealEntry]:
        """
        Remove a meal by its 1-based position (as shown in summary()'s
        "meals" list / the numbered list printed by 'show today's summary').

        Totals follow automatically (they are computed from self.meals).
        Returns the removed MealEntry, or None if the index is out of range.
        """
        if index < 1 or index > len(self.meals):
            return None
        return self.meals.pop(index - 1)

    def remove_meal_by_food(self, food_name: str) -> Optional[MealEntry]:
        """
        Remove the MOST RECENTLY logged meal whose food name matches
        `food_name` (case-insensitive substring match).

        Useful for "remove roti" style requests where the user doesn't
        know / doesn't want to look up the numeric index. Returns the
        removed MealEntry, or None if no match was found.
        """
        needle = food_name.strip().lower()
        for i in range(len(self.meals) - 1, -1, -1):
            if needle in self.meals[i].food.lower():
                return self.remove_meal(i + 1)
        return None

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------
    @property
    def total_calories(self) -> float:
        return round(self._total_calories, 1)

    @property
    def total_protein(self) -> float:
        return round(self._total_protein, 1)

    @property
    def total_carbs(self) -> float:
        return round(self._total_carbs, 1)

    @property
    def total_fat(self) -> float:
        return round(self._total_fat, 1)

    def remaining_calories(self) -> Optional[float]:
        """Return kcal remaining (None if budget not set)."""
        if not self.has_budget():
            return None
        return round(self.daily_budget - self._total_calories, 1)

    # ------------------------------------------------------------------
    # Macro split helpers
    # ------------------------------------------------------------------
    def macro_split(self) -> dict:
        """
        Compute approximate percentage contribution of each macro.

        Formulas (standard):
            Protein  cal = protein_g  × 4
            Carbs    cal = carbs_g    × 4
            Fat      cal = fat_g      × 9

        Returns a dict with macro grams, kcal from each, and % split.
        """
        protein_kcal = self._total_protein * 4
        carbs_kcal = self._total_carbs * 4
        fat_kcal = self._total_fat * 9
        total_macro_kcal = protein_kcal + carbs_kcal + fat_kcal

        if total_macro_kcal == 0:
            return {
                "protein_g": 0, "carbs_g": 0, "fat_g": 0,
                "protein_pct": 0, "carbs_pct": 0, "fat_pct": 0,
                "total_from_macros_kcal": 0,
            }

        return {
            "protein_g": round(self._total_protein, 1),
            "carbs_g": round(self._total_carbs, 1),
            "fat_g": round(self._total_fat, 1),
            "protein_pct": round(protein_kcal / total_macro_kcal * 100, 1),
            "carbs_pct": round(carbs_kcal / total_macro_kcal * 100, 1),
            "fat_pct": round(fat_kcal / total_macro_kcal * 100, 1),
            "total_from_macros_kcal": round(total_macro_kcal, 1),
        }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def summary(self) -> dict:
        """Return a snapshot of the current session state."""
        return {
            "daily_budget": self.daily_budget,
            "total_calories": self.total_calories,
            "remaining_calories": self.remaining_calories(),
            "total_protein": self.total_protein,
            "total_carbs": self.total_carbs,
            "total_fat": self.total_fat,
            "meal_count": len(self.meals),
            "meals": [m.to_dict() for m in self.meals],
        }

    def __repr__(self) -> str:
        budget_str = (f"{self.daily_budget} kcal" if self.has_budget()
                      else "not set")
        return (f"<SessionMemory budget={budget_str} "
                f"consumed={self.total_calories} kcal "
                f"meals={len(self.meals)}>")
