"""
agent.py

NOTE (Windows): Set PYTHONIOENCODING=utf-8 in your terminal, or run via
  Jupyter Notebook (which uses UTF-8 by default) to see emoji correctly.
--------
Core agent logic for the T17: Meal Calorie Helper.

Architecture: Plan-Act Loop
-----------------------------
The agent follows a ReAct-style loop:
  1. UNDERSTAND — parse the user's message and identify the goal/intent.
  2. PLAN       — decide which tool(s) are needed to satisfy the goal.
  3. ACT        — call the tool(s) and collect the results.
  4. OBSERVE    — read tool results and update reasoning.
  5. RESPOND    — compose a clear final response using tool results + memory.

The agent does NOT rely on an LLM to "remember" values — all state is stored
in SessionMemory and retrieved via tool calls.

This file exposes:
    MealCaloricAgent   — the main agent class
    run_interactive()  — a REPL loop for terminal usage

The agent uses RULE-BASED intent recognition combined with genuine tool
calls whose results drive the final answer. This keeps the project simple
enough for a CA1 viva while genuinely demonstrating agentic behaviour.
"""

from __future__ import annotations
import builtins
import io
import re
import sys
import threading

_capture_local = threading.local()


def print(*args, **kwargs) -> None:  # noqa: A001
    """Route module print calls to the active thread's `chat_capture` buffer when set."""
    buf = getattr(_capture_local, "buf", None)
    if buf is not None and "file" not in kwargs:
        builtins.print(*args, file=buf, **kwargs)
    else:
        builtins.print(*args, **kwargs)

# ── Windows UTF-8 fix ──────────────────────────────────────────────────────
# Reconfigure stdout to UTF-8 so emoji/Unicode characters display correctly
# on Windows terminals that default to cp1252.
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
# ─────────────────────────────────────────────────────────────────────────
from typing import Optional

# Import our tools (which also manage session memory)
from . import tools as T
from . import daily_store


# ---------------------------------------------------------------------------
# Intent definitions
# ---------------------------------------------------------------------------
# Each intent is a (name, pattern_list) pair.
# The agent tests these in order; the first match wins.

_INTENTS = [
    ("ADD_CUSTOM_FOOD", [
        r"^add custom food",
        r"^add my own food",
        r"^add custom meal",
    ]),
    ("SAVE_LAST_LOOKUP", [
        r"^save (this|it|that|last)\b",
        r"\bsave.*(?:to|in).*database\b",
        r"^save this food\b",
    ]),
    ("DELETE_MEAL", [
        r"^delete\b",
        r"^remove\b",
        r"\bdelete.*(?:meal|food|item)\b",
        r"\bremove.*(?:meal|food|item)\b",
        r"\bwrong(?:ly)?\s*(?:added|logged)\b",
        r"\bundo.*(?:meal|add|log)\b",
    ]),
    ("SHOW_HISTORY", [
        r"\b(?:show|view)\s+(?:my\s+)?history\b",
        r"\bpast\s+days\b",
        r"\byesterday'?s?\s+(?:log|summary|total)\b",
        r"\blast\s+\d+\s+days\b",
    ]),
    ("SET_BUDGET",   [
        r"budget.*?(\d+[\.,]?\d*)",
        r"(\d+[\.,]?\d*)\s*(?:kcal|calorie|cal).*budget",
        r"my.*(\d+[\.,]?\d*)\s*calorie",
        r"set.*(\d+[\.,]?\d*)",
        r"(\d+[\.,]?\d*)\s*is.*(?:my|the).*budget",
    ]),
    ("ADD_MEAL",     [
        r"^add\b",
        r"\bi(?:'ve| have)? eaten\b",
        r"\bi(?:'ve| have)? had\b",
        r"^i ate\b",
        r"^ate\b",
        r"\blog\b.*\bmeal\b",
        # Bare food strings: "3 eggs", "200g chicken", "3 eggs, 250 ml milk"
        r"^\d+(?:\.\d+)?\s*(?:g|gm|gram|grams|ml|kg)?\s+\w",
        r"^\d+(?:\.\d+)?\s+\w+.*,",   # starts with qty and has a comma → multi-food
    ]),
    ("CHECK_SNACK",  [
        r"\bcan i (?:eat|have|drink|take)\b",
        r"\bcan i (?:eat|have|drink|take)\b",
        r"\bshould i (?:eat|have)\b",
        r"\bwill .+ fit\b",
        r"\bdoes .+ fit\b",
        r"\bis .+ ok\b",
        r"\bfit.*budget\b",
    ]),
    ("RESET",        [
        r"\breset\b",
        r"\bclear\b.*(?:meal|log|today)",
        r"\bstart.*fresh\b",
        r"\bnew day\b",
    ]),
    ("SHOW_SUMMARY", [
        r"\bsummar",
        r"\bshow.*today",
        r"\bwhat.*(?:have i|i have).*(?:eaten|had|logged)\b",
        r"\bmy\s+meals?\b(?!.*\breset\b)",
        r"\bhow.*(?:many|much).*calorie",
        r"\bremaining",
        r"\btotal",
        r"\bmacro",
    ]),
    ("HELP",         [
        r"\bhelp\b",
        r"\bwhat can you do\b",
        r"\bcommand",
        r"^hi\b",
        r"^hello\b",
        r"^hey\b",
    ]),
]


def _detect_intent(text: str) -> str:
    """
    Return the intent label for the given user message.
    Falls back to UNKNOWN.
    """
    lower = text.lower().strip()
    for intent_name, patterns in _INTENTS:
        for pat in patterns:
            if re.search(pat, lower):
                return intent_name
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Multi-food splitter
# ---------------------------------------------------------------------------

# Words that can join multiple foods in one sentence
_FOOD_SEPARATORS = re.compile(
    r"\s*,\s*|\s+and\s+|\s+with\s+|\s+\+\s+",
    re.IGNORECASE,
)

# Format: "add custom food: <name> | calories=250 | protein=10 | carbs=30 | fat=5 | serving=1 bowl"
_CUSTOM_FOOD_PATTERN = re.compile(
    r"add\s+(?:custom\s+food|my\s+own\s+food|custom\s+meal)\s*:?\s*(?P<name>[^|]+)\|"
    r"\s*calories?\s*=\s*(?P<calories>\d+(?:\.\d+)?)\s*\|"
    r"\s*protein\s*=\s*(?P<protein>\d+(?:\.\d+)?)\s*\|"
    r"\s*carbs?\s*=\s*(?P<carbs>\d+(?:\.\d+)?)\s*\|"
    r"\s*fat\s*=\s*(?P<fat>\d+(?:\.\d+)?)"
    r"(?:\s*\|\s*serving\s*=\s*(?P<serving>.+))?",
    re.IGNORECASE,
)


def _split_multi_food(query: str) -> list[str]:
    """
    Split a natural-language multi-food string into individual food items.

    Examples:
        '3 eggs, 250 ml milk with 50g oats'
            → ['3 eggs', '250 ml milk', '50g oats']
        '2 roti and 100g curd'
            → ['2 roti', '100g curd']

    Returns a list of stripped, non-empty strings.
    """
    parts = _FOOD_SEPARATORS.split(query)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_macro_split(ms: dict) -> str:
    """Format macro split for display."""
    if ms.get("total_from_macros_kcal", 0) == 0:
        return "  (No macros logged yet.)"
    lines = [
        "  📊 Macro Split (approximate):",
        f"     Protein : {ms['protein_g']} g  → {ms['protein_pct']}% of macro calories",
        f"     Carbs   : {ms['carbs_g']} g  → {ms['carbs_pct']}% of macro calories",
        f"     Fat     : {ms['fat_g']} g  → {ms['fat_pct']}% of macro calories",
        f"  ⚡ Total from macros: ~{ms['total_from_macros_kcal']} kcal",
        "  (Protein×4 + Carbs×4 + Fat×9 — approximate; for informational use only)",
    ]
    return "\n".join(lines)


def _fmt_daily_bar(consumed: float, budget: Optional[float]) -> str:
    """Show a simple text progress bar for daily calories."""
    if budget is None or budget == 0:
        return f"  Consumed today: {consumed} kcal (no budget set)"
    pct = min(consumed / budget * 100, 100)
    filled = int(pct / 5)          # 20 block bar
    bar = "█" * filled + "░" * (20 - filled)
    remaining = round(budget - consumed, 1)
    emoji = "✅" if remaining >= 0 else "⚠️"
    return (
        f"  [{bar}] {pct:.1f}%\n"
        f"  Consumed : {consumed} / {budget} kcal\n"
        f"  Remaining: {remaining} kcal {emoji}"
    )


def _infer_unit_base(serving: str) -> tuple[str, float]:
    """
    Infer (unit, base_amount) from a human-written serving string.

    Examples:
        "100 g"              -> ("g", 100)
        "per 100g"           -> ("g", 100)
        "250 ml"             -> ("ml", 250)
        "1 bowl (250 g)"     -> ("piece", 1)   # weight is inside parens,
                                                 # not the leading token —
                                                 # treated as one-off dish
        "1 bowl"             -> ("piece", 1)

    This lets a food described "per 100 g" (e.g. a packaged product's
    nutrition label) scale correctly if the user later logs it by grams,
    while a one-off home-cooked dish ("1 bowl of rajma") stays as a
    single indivisible serving.
    """
    m = re.match(r"^\s*(?:per\s+)?(\d+(?:\.\d+)?)\s*(g|gm|gram|grams|ml)\b",
                 serving.strip(), re.IGNORECASE)
    if m:
        qty = float(m.group(1))
        u = m.group(2).lower()
        u = "g" if u in ("g", "gm", "gram", "grams") else "ml"
        return u, qty
    return "piece", 1


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class MealCaloricAgent:
    """
    T17: Meal Calorie Helper Agent.

    The agent maintains a Plan-Act loop:
        observe(user_message)
        → detect_intent()
        → plan(tool_sequence)
        → call tools in sequence
        → read tool results
        → compose final response

    All state is stored in tools.py's session memory.
    """

    DISCLAIMER = (
        "\n⚠️  All nutritional values are estimated. "
        "Actual values vary by brand, preparation, and serving size. "
        "This tool is for informational purposes only."
    )

    def __init__(self, verbose: bool = True) -> None:
        """
        Parameters:
            verbose (bool): If True, display plan-act trace during reasoning.
        """
        self.verbose = verbose
        self._turn = 0

    # ------------------------------------------------------------------
    # Logging helpers (shown to user to make agentic trace visible)
    # ------------------------------------------------------------------

    def _log(self, tag: str, msg: str) -> None:
        if self.verbose:
            print(f"  [{tag}] {msg}")

    def _section(self, title: str) -> None:
        if self.verbose:
            print(f"\n{'─'*55}")
            print(f"  🤖 AGENT — {title}")
            print(f"{'─'*55}")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> None:
        """
        Process one user turn. All output is printed directly.
        Returns None so Jupyter does not display the raw response string.

        Plan-Act loop steps are printed if verbose=True.
        """
        self._turn += 1

        # --- Day persistence / rollover ------------------------------------
        # Restores today's earlier meals if this is a fresh process, or
        # archives yesterday into history and starts a clean day if the
        # calendar date has changed since the last saved entry.
        mem = T.get_memory()
        rollover = daily_store.load_or_rollover(mem)
        if rollover["action"] == "rolled_over":
            print(
                f"\n🌅 New day detected — {rollover['previous_date']}'s log has "
                f"been saved to history. Starting a fresh day"
                + (f" with the same {mem.daily_budget:.0f} kcal budget.\n"
                   if mem.daily_budget else ".\n")
            )
        elif rollover["action"] == "restored":
            print(f"\n🔁 Restored today's earlier meals from disk ({len(mem.meals)} logged so far).\n")

        print(f"\n{'═'*60}")
        print(f"  👤 USER (Turn {self._turn}): {user_message}")
        print(f"{'═'*60}")

        intent = _detect_intent(user_message)
        self._log("PLAN", f"Intent detected → {intent}")

        # Route to the correct handler
        if intent == "ADD_CUSTOM_FOOD":
            self._handle_add_custom_food(user_message)
        elif intent == "SAVE_LAST_LOOKUP":
            self._handle_save_last_lookup()
        elif intent == "DELETE_MEAL":
            self._handle_delete_meal(user_message)
        elif intent == "SHOW_HISTORY":
            self._handle_show_history(user_message)
        elif intent == "SET_BUDGET":
            self._handle_set_budget(user_message)
        elif intent == "ADD_MEAL":
            # Strip command prefix and check if multiple foods remain
            food_part = re.sub(
                r"^(add|ate|eaten|had|i ate|i had|i've eaten|i've had|log meal|logged)\s+",
                "", user_message.strip(), flags=re.IGNORECASE
            ).strip()
            foods = _split_multi_food(food_part)
            if len(foods) > 1:
                self._log("PLAN", f"Multi-food input detected — {len(foods)} items to log")
                self._handle_add_multi_meal(foods)
            else:
                self._handle_add_meal(user_message)
        elif intent == "CHECK_SNACK":
            self._handle_check_snack(user_message)
        elif intent == "SHOW_SUMMARY":
            self._handle_summary()
        elif intent == "RESET":
            self._handle_reset()
        elif intent == "HELP":
            self._handle_help()
        elif intent == "UNKNOWN":
            # Last-chance: if message looks like a food list, try adding it
            foods = _split_multi_food(user_message)
            if foods and re.search(r"\d", user_message):
                self._log("PLAN", f"Bare food input — treating as multi-meal add ({len(foods)} items)")
                self._handle_add_multi_meal(foods)
            else:
                self._handle_unknown(user_message)
        else:
            self._handle_unknown(user_message)

        # Persist today's state after every turn, in case anything changed
        # (add/delete meal, set/keep budget, reset). Cheap and safe to call
        # even on turns that didn't mutate anything.
        daily_store.sync_after_change(T.get_memory())

    # ------------------------------------------------------------------
    # Web / API entry point
    # ------------------------------------------------------------------

    def chat_capture(self, user_message: str) -> dict:
        """
        Same behaviour as chat(), but captures all printed output instead
        of writing straight to the terminal, and returns it as structured
        data. Used by the Flask web frontend (app.py) so the browser can
        render the response and the plan-act trace separately.

        Does not change chat()'s own behaviour or return value at all —
        this only wraps it.

        Returns:
            dict with keys:
                response : the "🤖 RESPONSE:" text shown to the user
                trace    : everything printed BEFORE the response (the
                           [PLAN]/[ACT]/[TOOL RESULT]/[OBSERVE]/[MEMORY]
                           plan-act trace lines), as one string
                raw      : the full unsplit captured output, for debugging
        """
        buf = io.StringIO()
        prev_buf = getattr(_capture_local, "buf", None)
        _capture_local.buf = buf
        try:
            self.chat(user_message)
        finally:
            _capture_local.buf = prev_buf
        text = buf.getvalue()

        if "🤖 RESPONSE:" in text:
            before, _, after = text.partition("🤖 RESPONSE:")
            trace_text = before.strip()
            response_text = after.strip()
        else:
            # Some paths (e.g. a parse error) print only a short message
            # with no separate RESPONSE marker — treat it all as the reply.
            trace_text = ""
            response_text = text.strip()

        return {
            "response": response_text,
            "trace": trace_text,
            "raw": text,
        }

    # ------------------------------------------------------------------
    # Intent handlers
    # ------------------------------------------------------------------

    def _handle_set_budget(self, msg: str) -> str:
        self._section("Setting daily calorie budget")

        # Extract the number from the message
        numbers = re.findall(r"\d+[\.,]?\d*", msg)
        if not numbers:
            return (
                "❌ I couldn't find a calorie number in your message.\n"
                "   Please say something like: 'My calorie budget is 2200 kcal'"
            )

        budget = float(numbers[0].replace(",", ""))
        self._log("ACT", f"Calling set_budget({budget})")

        result = T.set_budget(budget)
        self._log("TOOL RESULT", f"Status = {result['status']}, budget = {result.get('daily_budget')} kcal")

        if result["status"] == "error":
            return f"❌ {result['error']}"

        response = (
            f"✅ Daily calorie budget set to {budget:.0f} kcal.\n"
            f"   Start logging your meals and I'll track calories and macros for you.\n"
            + self.DISCLAIMER
        )
        print(f"\n  🤖 RESPONSE:\n{response}")
        return response

    # ------------------------------------------------------------------

    def _handle_add_meal(self, msg: str) -> str:
        self._section("Logging a meal")

        # Strip command words to isolate the food part
        food_query = re.sub(
            r"^(add|ate|eaten|had|i ate|i had|i've eaten|i've had|log meal|logged)\s+",
            "", msg.strip(), flags=re.IGNORECASE
        ).strip()

        if not food_query:
            return (
                "❌ Please tell me what food to add.\n"
                "   Example: 'Add 3 boiled eggs' or 'I had 200 g chicken breast'"
            )

        # STEP 1: Lookup calories first (so the agent "reads" the result)
        self._log("ACT",     f"Calling lookup_calories('{food_query}')")
        lookup = T.lookup_calories(food_query)
        self._log("TOOL RESULT", f"Status = {lookup['status']}")

        if lookup["status"] == "not_found":
            response = (
                f"❌ Food not found: {lookup['error']}\n"
                "   Tip: Try a more common food name, or add it yourself:\n"
                "   add custom food: <name> | calories=250 | protein=10 | "
                "carbs=30 | fat=5 | serving=1 bowl"
            )
            print(f"\n  🤖 RESPONSE:\n{response}")
            return response

        if lookup.get("source") == "api":
            self._log("OBSERVE", "Not in local database — fetched from ONLINE API instead")

        self._log("OBSERVE",
                  f"Found '{lookup['food']}' — {lookup['calories']} kcal, "
                  f"P:{lookup['protein']}g C:{lookup['carbs']}g F:{lookup['fat']}g "
                  f"(source: {lookup.get('source', 'local_database')})")

        # STEP 2: Add the meal to memory
        self._log("ACT", f"Calling add_meal('{food_query}')")
        result = T.add_meal(food_query)
        self._log("TOOL RESULT",
                  f"Status = {result['status']}, "
                  f"Daily total = {result['daily_totals']['calories']} kcal")

        # STEP 3: Read memory state
        mem = T.get_memory()
        self._log("MEMORY",
                  f"Consumed = {mem.total_calories} / "
                  f"{mem.daily_budget or '(no budget)'} kcal | "
                  f"Meals logged = {len(mem.meals)}")

        # STEP 4: Compose response
        meal = result["meal_added"]
        totals = result["daily_totals"]
        remaining = result["remaining_kcal"]
        ms = result["macro_split"]

        budget_line = ""
        if mem.has_budget():
            rem_str = f"{remaining} kcal" if remaining is not None else "N/A"
            budget_line = (
                f"\n  📅 Today: {totals['calories']} / {mem.daily_budget} kcal"
                f"\n  ⏳ Remaining: {rem_str}"
            )

        online_hint = (
            "\n  🌐 This was fetched online, not from your local database. "
            "Say 'save this food' to store it permanently for offline use.\n"
            if lookup.get("source") == "api" else ""
        )

        response = (
            f"✅ Added: {meal['food'].title()} ({meal['quantity']})\n"
            f"  🔥 Calories: {meal['calories']} kcal\n"
            f"  🥩 Protein : {meal['protein']} g\n"
            f"  🍞 Carbs   : {meal['carbs']} g\n"
            f"  🧈 Fat     : {meal['fat']} g"
            + budget_line + online_hint + "\n\n"
            + _fmt_macro_split(ms)
            + "\n" + self.DISCLAIMER
        )
        print(f"\n  🤖 RESPONSE:\n{response}")
        return response

    # ------------------------------------------------------------------

    def _handle_add_multi_meal(self, food_items: list[str]) -> str:
        """
        Add multiple foods in one turn.
        Shows a per-item trace and a combined final summary.
        """
        self._section(f"Logging {len(food_items)} foods")
        self._log("PLAN", f"Will call lookup_calories + add_meal for each: {food_items}")

        added = []
        failed = []

        for item in food_items:
            self._log("ACT", f"Calling lookup_calories('{item}')")
            lookup = T.lookup_calories(item)
            self._log("TOOL RESULT", f"Status = {lookup['status']}")

            if lookup["status"] == "not_found":
                self._log("OBSERVE", f"'{item}' not found in database — skipping")
                failed.append(item)
                continue

            self._log("OBSERVE",
                      f"Found '{lookup['food']}' — {lookup['calories']} kcal "
                      f"P:{lookup['protein']}g C:{lookup['carbs']}g F:{lookup['fat']}g")

            self._log("ACT", f"Calling add_meal('{item}')")
            result = T.add_meal(item)
            self._log("TOOL RESULT",
                      f"Status = {result['status']}, "
                      f"Running total = {result['daily_totals']['calories']} kcal")
            added.append(result["meal_added"])

        # Read memory once after all additions
        mem = T.get_memory()
        self._log("MEMORY",
                  f"Total consumed = {mem.total_calories} / "
                  f"{mem.daily_budget or '(no budget)'} kcal | "
                  f"Meals logged = {len(mem.meals)}")

        # Build response
        lines = [f"✅ Added {len(added)} food(s):\n"]
        for m in added:
            lines.append(
                f"  • {m['food'].title()} ({m['quantity']}) — "
                f"{m['calories']} kcal | P:{m['protein']}g C:{m['carbs']}g F:{m['fat']}g"
            )

        if failed:
            lines.append(f"\n  ⚠️  Not found (skipped): {', '.join(failed)}")
            lines.append("     Try individual names like 'muesli' → 'oats', or add "
                          "them yourself: add custom food: <name> | calories=X | "
                          "protein=X | carbs=X | fat=X")

        ms = mem.macro_split()
        if mem.has_budget():
            remaining = mem.remaining_calories()
            lines.append(
                f"\n  📅 Today: {mem.total_calories} / {mem.daily_budget} kcal"
                f"\n  ⏳ Remaining: {remaining} kcal"
            )

        lines.append("\n" + _fmt_macro_split(ms))
        lines.append(self.DISCLAIMER)

        response = "\n".join(lines)
        print(f"\n  🤖 RESPONSE:\n{response}")
        return response

    # ------------------------------------------------------------------

    def _handle_check_snack(self, msg: str) -> str:
        self._section("Checking snack feasibility")

        # Extract the food part from "can I eat/have X?"
        food_query = re.sub(
            r"^(?:can\s+i|should\s+i)\s+(?:eat|have|drink|take)\s+(?:a\s+|an\s+|some\s+)?",
            "", msg.strip(), flags=re.IGNORECASE
        ).strip()
        food_query = re.sub(r"\?$", "", food_query).strip()

        if not food_query:
            return (
                "❌ Please specify what you want to eat.\n"
                "   Example: 'Can I eat a banana?'"
            )

        # STEP 1: Check if budget is set
        mem = T.get_memory()
        self._log("MEMORY", f"Budget = {mem.daily_budget}, Consumed = {mem.total_calories} kcal")

        # STEP 2: Lookup snack calories
        self._log("ACT", f"Calling lookup_calories('{food_query}')")
        lookup = T.lookup_calories(food_query)
        self._log("TOOL RESULT", f"Status = {lookup['status']}")

        if lookup["status"] == "not_found":
            response = (
                f"❌ Food not found: {lookup['error']}\n"
                "   Add it yourself: add custom food: <name> | calories=X | "
                "protein=X | carbs=X | fat=X | serving=..."
            )
            print(f"\n  🤖 RESPONSE:\n{response}")
            return response

        if lookup.get("source") == "api":
            self._log("OBSERVE", "Not in local database — fetched from ONLINE API instead")

        self._log("OBSERVE",
                  f"Snack = '{lookup['food']}' → {lookup['calories']} kcal "
                  f"(source: {lookup.get('source', 'local_database')})")

        # STEP 3: Check fit
        self._log("ACT", f"Calling check_snack_fit('{food_query}')")
        result = T.check_snack_fit(food_query)
        self._log("TOOL RESULT", f"Status = {result['status']}")

        if result["status"] == "no_budget":
            response = (
                "⚠️  No budget set yet.\n"
                "   Please set a daily calorie budget first:\n"
                "   'My calorie budget is 2000 kcal'"
            )
            print(f"\n  🤖 RESPONSE:\n{response}")
            return response

        # STEP 4: Read remaining budget
        remaining = result["remaining_kcal"]
        snack_cal = result["snack_calories"]
        self._log("OBSERVE",
                  f"Remaining budget = {remaining} kcal | "
                  f"Snack = {snack_cal} kcal | "
                  f"Surplus/deficit = {result['surplus_deficit']} kcal")

        # STEP 5: Make decision
        decision = "FITS ✅" if result["fits"] else "DOES NOT FIT ❌"
        self._log("DECIDE", decision)

        macro_p = result["macro_preview"]
        snack_macro_line = (
            f"  🥩 Protein: {macro_p['protein']} g | "
            f"🍞 Carbs: {macro_p['carbs']} g | "
            f"🧈 Fat: {macro_p['fat']} g"
        )

        online_hint = (
            "\n  🌐 This was fetched online, not from your local database. "
            "Say 'save this food' to store it permanently for offline use.\n"
            if lookup.get("source") == "api" else ""
        )

        response = (
            f"{result['decision']}\n\n"
            f"  Snack nutrition:\n{snack_macro_line}\n"
            + online_hint + "\n"
            f"  Budget overview:\n"
            + _fmt_daily_bar(result["consumed_kcal"], result["budget"])
            + "\n" + self.DISCLAIMER
        )
        print(f"\n  🤖 RESPONSE:\n{response}")
        return response

    # ------------------------------------------------------------------

    def _handle_summary(self) -> str:
        self._section("Retrieving daily summary")

        self._log("ACT", "Calling get_daily_summary()")
        summary = T.get_daily_summary()
        self._log("TOOL RESULT",
                  f"Budget={summary['daily_budget']}, "
                  f"Consumed={summary['total_calories']} kcal, "
                  f"Meals={summary['meal_count']}")

        if summary["meal_count"] == 0:
            response = (
                "📋 No meals logged yet today.\n"
                "   Add a meal: 'Add 3 boiled eggs'"
            )
            print(f"\n  🤖 RESPONSE:\n{response}")
            return response

        # Format meals list
        meal_lines = []
        for i, m in enumerate(summary["meals"], 1):
            meal_lines.append(
                f"   {i}. {m['food'].title()} ({m['quantity']}) — {m['calories']} kcal | "
                f"P:{m['protein']}g C:{m['carbs']}g F:{m['fat']}g"
            )

        ms = summary["macro_split"]

        response = (
            "📋 Today's Meal Log:\n"
            + "\n".join(meal_lines) + "\n\n"
            + _fmt_daily_bar(summary["total_calories"], summary["daily_budget"]) + "\n\n"
            + _fmt_macro_split(ms)
            + "\n" + self.DISCLAIMER
        )
        print(f"\n  🤖 RESPONSE:\n{response}")
        return response

    # ------------------------------------------------------------------

    def _handle_reset(self) -> str:
        self._section("Resetting daily log")
        self._log("ACT", "Clearing all meals from session memory")

        mem = T.get_memory()
        mem.reset_meals()

        self._log("MEMORY", "Meals cleared. Budget preserved.")
        response = (
            "🔄 Daily meal log cleared.\n"
            f"   Budget is still set to {mem.daily_budget} kcal.\n"
            "   Ready to log a fresh day!"
        )
        print(f"\n  🤖 RESPONSE:\n{response}")
        return response

    # ------------------------------------------------------------------

    def _handle_show_history(self, msg: str) -> str:
        """
        Show a summary of past (fully-closed) days. Today's in-progress
        log is not included here — use 'show today's summary' for that.
        """
        self._section("Looking up your daily history")

        numbers = re.findall(r"\d+", msg)
        limit = int(numbers[0]) if numbers else 7

        self._log("ACT", f"Calling get_daily_history(limit={limit})")
        history = T.get_daily_history(limit=limit)
        self._log("TOOL RESULT", f"{len(history)} past day(s) found")

        if not history:
            response = (
                "📅 No past days saved yet — history builds up automatically "
                "once a full day has gone by."
            )
            print(f"\n  🤖 RESPONSE:\n{response}")
            return response

        lines = [f"📅 Last {len(history)} day(s):\n"]
        for d, day in history.items():
            budget = day.get("budget")
            budget_str = f"/{budget:.0f}" if budget else ""
            lines.append(
                f"  {d} — {day['total_calories']:.0f}{budget_str} kcal | "
                f"P:{day['total_protein']:.0f}g C:{day['total_carbs']:.0f}g "
                f"F:{day['total_fat']:.0f}g | {len(day['meals'])} meal(s)"
            )
        response = "\n".join(lines)
        print(f"\n  🤖 RESPONSE:\n{response}")
        return response

    # ------------------------------------------------------------------

    def _handle_delete_meal(self, msg: str) -> str:
        """
        Remove a single wrongly-logged meal, by number or by food name,
        and recompute totals + macros. Does NOT touch the budget.

        Examples:
            'delete meal 2'   → removes item #2 from today's list
            'remove roti'     → removes the most recent 'roti' entry
        """
        self._section("Deleting a logged meal")

        numbers = re.findall(r"\d+", msg)

        if numbers:
            index = int(numbers[0])
            self._log("ACT", f"Calling delete_meal(index={index})")
            result = T.delete_meal(index=index)
        else:
            food_query = re.sub(
                r"^(delete|remove|undo)\s+(the\s+)?(meal|food|item)?\s*",
                "", msg.strip(), flags=re.IGNORECASE
            ).strip()
            if not food_query:
                response = (
                    "❌ Tell me which meal to delete — by number or food name.\n"
                    "   Example: 'delete meal 2' or 'remove roti'\n"
                    "   (Say 'show today's summary' first to see the numbered list.)"
                )
                print(f"\n  🤖 RESPONSE:\n{response}")
                return response
            self._log("ACT", f"Calling delete_meal(food_name='{food_query}')")
            result = T.delete_meal(food_name=food_query)

        self._log("TOOL RESULT", f"Status = {result['status']}")

        if result["status"] != "deleted":
            response = f"❌ {result['error']}"
            print(f"\n  🤖 RESPONSE:\n{response}")
            return response

        removed = result["removed"]
        mem = T.get_memory()
        self._log("MEMORY",
                  f"Removed '{removed['food']}' — new total = "
                  f"{result['daily_totals']['calories']} kcal | "
                  f"Meals logged = {len(mem.meals)}")

        budget_line = ""
        if mem.has_budget():
            budget_line = (
                f"\n  📅 Today: {result['daily_totals']['calories']} / {mem.daily_budget} kcal"
                f"\n  ⏳ Remaining: {result['remaining_kcal']} kcal"
            )

        response = (
            f"🗑️  Removed: {removed['food'].title()} ({removed['quantity']}) — "
            f"{removed['calories']} kcal"
            + budget_line + "\n\n"
            + _fmt_macro_split(result["macro_split"])
        )
        print(f"\n  🤖 RESPONSE:\n{response}")
        return response

    # ------------------------------------------------------------------

    def _handle_add_custom_food(self, msg: str) -> str:
        """
        Let the user permanently add their own food to the local database
        with manually supplied nutrition info.

        Expected format:
            add custom food: <name> | calories=250 | protein=10 |
                              carbs=30 | fat=5 | serving=1 bowl
        """
        self._section("Adding a custom food to your database")

        m = _CUSTOM_FOOD_PATTERN.search(msg)
        if not m:
            response = (
                "❌ I couldn't parse that. Use this format:\n"
                "   add custom food: <name> | calories=250 | protein=10 | "
                "carbs=30 | fat=5 | serving=1 bowl\n\n"
                "   Example:\n"
                "   add custom food: mom's rajma | calories=320 | protein=14 | "
                "carbs=45 | fat=8 | serving=1 bowl (250g)"
            )
            print(f"\n  🤖 RESPONSE:\n{response}")
            return response

        name = m.group("name").strip()
        calories = float(m.group("calories"))
        protein = float(m.group("protein"))
        carbs = float(m.group("carbs"))
        fat = float(m.group("fat"))
        serving = (m.group("serving") or "1 serving").strip()

        # Infer whether this is a per-weight/volume product (e.g. "100 g",
        # "per 100g") vs a one-off dish (e.g. "1 bowl") so later lookups by
        # grams/ml scale correctly instead of always returning the full
        # entry's calories.
        unit, base_amount = _infer_unit_base(serving)

        self._log("PLAN", f"Will call add_custom_food_tool('{name}', ...)")
        self._log("ACT", f"Calling add_custom_food_tool('{name}', {calories} kcal, "
                          f"P:{protein}g C:{carbs}g F:{fat}g, unit={unit}, base={base_amount})")
        result = T.add_custom_food_tool(name, calories, protein, carbs, fat,
                                         serving, unit=unit, base_amount=base_amount)
        self._log("TOOL RESULT", f"Status = {result['status']}")

        if result["status"] == "error":
            response = f"❌ {result['error']}"
        else:
            self._log("MEMORY", f"'{result['food_key']}' now permanently in local database")
            response = (
                f"✅ {result['message']}\n"
                f"   You can now say: 'Add {name}' any time — found locally,\n"
                f"   no internet required."
                + "\n" + self.DISCLAIMER
            )
        print(f"\n  🤖 RESPONSE:\n{response}")
        return response

    # ------------------------------------------------------------------

    def _handle_save_last_lookup(self) -> str:
        """
        Save the most recent ONLINE (API) lookup permanently into the
        local database, so it's available offline next time.
        """
        self._section("Saving last online lookup to local database")

        self._log("ACT", "Calling save_last_lookup()")
        result = T.save_last_lookup()
        self._log("TOOL RESULT", f"Status = {result['status']}")

        if result["status"] == "error":
            response = f"⚠️ {result['error']}"
        else:
            self._log("MEMORY", f"'{result['food_key']}' now permanently in local database")
            response = (
                f"✅ {result['message']}"
                + "\n" + self.DISCLAIMER
            )
        print(f"\n  🤖 RESPONSE:\n{response}")
        return response

    # ------------------------------------------------------------------

    def _handle_help(self) -> str:
        response = (
            "👋 Hello! I'm your Meal Calorie Helper Agent.\n\n"
            "  What I can do:\n"
            "  • Set budget  : 'My calorie budget is 2200 kcal'\n"
            "  • Log a meal  : 'Add 3 boiled eggs'\n"
            "                  'I had 200 g chicken breast'\n"
            "                  'Add 2 slices of bread'\n"
            "  • Check snack : 'Can I eat a banana?'\n"
            "                  'Can I have 2 slices of bread?'\n"
            "  • Summary     : 'Show today's summary'\n"
            "                  'How many calories have I eaten?'\n"
            "  • Reset       : 'Reset my meals'\n"
            "  • Delete meal : 'delete meal 2'  (see numbered list in summary)\n"
            "                  'remove roti'    (removes most recent match)\n"
            "  • Unknown food: if a food isn't in my database I'll try an\n"
            "                  online lookup automatically (if configured).\n"
            "                  Say 'save this food' afterwards to keep it\n"
            "                  permanently, offline.\n"
            "  • Your own food: 'add custom food: mom's rajma | calories=320 |\n"
            "                  protein=14 | carbs=45 | fat=8 | serving=1 bowl'\n\n"
            "  I track calories + macros (protein, carbs, fat) for you. 🥗"
        )
        print(f"\n  🤖 RESPONSE:\n{response}")
        return response

    # ------------------------------------------------------------------

    def _handle_unknown(self, msg: str) -> str:
        self._log("OBSERVE", f"Could not determine intent for: '{msg}'")
        response = (
            "🤔 I'm not sure what you mean. Here are some things you can say:\n"
            "  • 'My calorie budget is 2000 kcal'\n"
            "  • 'Add 3 eggs'\n"
            "  • 'Can I eat a banana?'\n"
            "  • 'Show summary'\n"
            "  • Type 'help' for more options."
        )
        print(f"\n  🤖 RESPONSE:\n{response}")
        return response


# ---------------------------------------------------------------------------
# Interactive REPL (for terminal use)
# ---------------------------------------------------------------------------

def run_interactive() -> None:
    """
    Start an interactive terminal session with the agent.
    Type 'quit' or 'exit' to stop.
    """
    agent = MealCaloricAgent(verbose=True)
    print("=" * 60)
    print("  🍽️  T17: Meal Calorie Helper — Interactive Session")
    print("  Type 'help' to see commands. Type 'quit' to exit.")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye! Stay healthy! 🥗")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("  Goodbye! Stay healthy! 🥗")
            break

        agent.chat(user_input)


if __name__ == "__main__":
    run_interactive()
