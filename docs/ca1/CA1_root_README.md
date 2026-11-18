# T17: Meal Calorie Helper

**CSE 476 — Agentic AI and Intelligent Automation | CA1 Assignment**

An AI agent that tracks a user's daily meals against a calorie budget.
It is not a chatbot — it calls real Python tools to look up nutrition
data, remembers state across the whole conversation (and across days),
and makes decisions (like whether a planned snack fits the remaining
budget) by actually computing from tool results, not from a canned
reply.


## Contents

1. [Quick Start](#1-quick-start)
2. [Workflow — how a request actually gets handled](#2-workflow--how-a-request-actually-gets-handled)
3. [File-by-file guide](#3-file-by-file-guide)
4. [Every function, what it's for](#4-every-function-what-its-for)
5. [Memory — two layers](#5-memory--two-layers)
6. [Group of 3 Add-on — Macro Split](#6-group-of-3-add-on--macro-split)
7. [Honest Failures Found & Fixed](#7-honest-failures-found--fixed)
8. [How to Run — every option](#8-how-to-run--every-option)
9. [Supported Foods](#9-supported-foods)

---

## 1. Quick Start

```bash
pip install -r requirements.txt
python agent.py
```

Then just type things like:
```
My calorie budget is 2200 kcal
Add 3 boiled eggs
Can I eat a banana?
Show today's summary
quit
```

That's it — no API keys, no signup, no internet required for the core
features. See [section 8](#8-how-to-run--every-option) for the notebook
and web-UI options too.

---

## 2. Workflow — how a request actually gets handled

Every message goes through the same pipeline, regardless of whether
you're using the terminal, the notebook, or the web UI:

```
User types a message
        │
        ▼
┌───────────────────────┐
│  daily_store checks:    │   ← is this a NEW calendar day since the
│  same day, or rollover? │     last message? If so, archive yesterday
└───────────┬─────────────┘     and start today fresh (budget carries over)
            ▼
┌───────────────────────┐
│  Intent detection        │   ← regex-based: matches the message against
│  (agent.py)               │     patterns to decide what the user wants
└───────────┬───────────────┘
            ▼
┌───────────────────────┐
│  Call the right tool(s)  │   ← e.g. lookup_calories(), then add_meal(),
│  (tools.py)                │     then read back the updated totals
└───────────┬───────────────┘
            ▼
┌───────────────────────┐
│  Read the tool's RESULT  │   ← e.g. "not found locally, try online" or
│  and decide next step     │     "found — now check if it fits the budget"
└───────────┬───────────────┘
            ▼
┌───────────────────────┐
│  Update memory            │   ← SessionMemory (this conversation) +
│  (memory.py)                │     daily_store (saved to disk)
└───────────┬───────────────┘
            ▼
┌───────────────────────┐
│  Compose & return reply   │
└───────────────────────┘
```

**The important part for "why is this an agent, not a chatbot":** step
3 and step 4 aren't fixed. `lookup_calories()` might succeed locally,
might need a second tool call to an online source, or might fail
entirely — and what the agent does NEXT depends on which of those
happened. That branching, driven by real tool output, is the whole
point.

### Example trace: "Add 3 boiled eggs"

```
[PLAN] Intent detected → ADD_MEAL
[ACT] Calling lookup_calories('3 boiled eggs')
[TOOL RESULT] Status = found
[OBSERVE] Found 'egg' — 234.0 kcal P:18.0g C:1.8g F:15.9g
[ACT] Calling add_meal('3 boiled eggs')
[TOOL RESULT] Status = added, Daily total = 234.0 kcal
[MEMORY] Consumed = 234.0 / 2200.0 kcal | Meals logged = 1
```

### Example trace: "Can I eat a banana?" (fits vs. doesn't fit)

```
[ACT] Calling lookup_calories('banana')
[TOOL RESULT] 105.0 kcal
[ACT] Reading remaining_calories() from memory
[MEMORY] 1966.0 kcal remaining
[OBSERVE] 105 < 1966 → decision: FITS
```
If the same food is asked about later in the day after more meals have
been logged, the exact same code path can return "does NOT fit" instead
— because the decision is computed fresh each time from current memory,
never hard-coded.

---

## 3. File-by-file guide

| File | Purpose |
|---|---|
| `agent.py` | The decision layer. Reads the user's message, figures out the intent (regex matching), calls the right tool(s) in the right order, and formats the reply. This is the "brain" that drives the plan-act loop. |
| `tools.py` | The 9 real tool functions. All actual logic — nutrition lookup, quantity scaling, budget math — lives here, not in `agent.py`. |
| `memory.py` | `SessionMemory` class: budget, logged meals, running totals for the CURRENT conversation. |
| `food_database.py` | Built-in nutrition data for 40+ common foods, the food-matching logic, and permanent storage for custom/user-added foods (`custom_foods.json`). |
| `daily_store.py` | Cross-**day** persistence: saves today's state to disk, detects when a new calendar day has started, archives yesterday into history, and resets today automatically. |
| `config.py` | Settings for the online nutrition fallback (Open Food Facts — free, no key needed) used only when a food isn't in the local database. |
| `app.py` | Flask backend for the optional web UI. Wraps `agent.py` in a small HTTP API. |
| `frontend/index.html` | Browser chat interface — talks to `app.py`. |
| `T17_Meal_Calorie_Helper.ipynb` | Demo notebook showing the multi-step trace for several example goals. |
| `TESTING.md` | Full manual + automated test checklist covering every feature. |

---

## 4. Every function, what it's for

### Tool functions (`tools.py`) — what the agent actually calls

| Function | What it does |
|---|---|
| `lookup_calories(query)` | Looks up estimated calories + macros for a food **without logging it**. Tries the local database first, then an online fallback if not found. Read-only — used both for plain nutrition questions and as the first step before logging or checking a snack. |
| `add_meal(query)` | Logs a food to today's record. Internally reuses the same lookup logic as `lookup_calories`, then writes a `MealEntry` into `SessionMemory` and updates running totals. |
| `check_snack_fit(query)` | Looks up a food's calories, reads the remaining budget from memory, computes the difference, and returns a fits/doesn't-fit decision — all without logging anything. |
| `get_daily_summary()` | Returns today's full state: budget, every logged meal, totals, remaining calories, macro percentage split. |
| `set_budget(budget_kcal)` | Sets the daily calorie target in memory. |
| `delete_meal(index, food_name)` | Removes one previously-logged meal (by its position in the list, or by matching the food name) and recomputes all totals. |
| `add_custom_food_tool(name, calories, protein, carbs, fat, serving_size)` | Permanently saves a food the user describes themselves — a homemade dish or a branded product not in the built-in list — so future lookups find it instantly, offline. |
| `save_last_lookup()` | Caches the most recent food found via the ONLINE fallback into the permanent local database, so it doesn't need the internet again next time. |
| `get_daily_history(limit)` | Returns a summary of past (already-closed) days — NOT today — from `daily_store`. |

### Memory functions (`memory.py`)

| Function | What it does |
|---|---|
| `set_budget(kcal)` / `has_budget()` | Store / check whether a daily target is set. |
| `add_meal(entry)` | Append a logged meal and update running totals. |
| `remove_meal(index)` / `remove_meal_by_food(name)` | Remove one meal and recompute totals. |
| `reset_meals()` | Clear all of today's meals (keeps the budget). |
| `remaining_calories()` | `budget − total_calories`. This is what `check_snack_fit` reads. |
| `macro_split()` | Computes protein/carb/fat percentage of total calories (see [section 6](#6-group-of-3-add-on--macro-split)). |
| `summary()` | Full snapshot dict of the current day's state. |

### Day-persistence functions (`daily_store.py`)

| Function | What it does |
|---|---|
| `sync_after_change(memory)` | Writes today's current state (budget + meals) to `daily_log.json` — called after every change so nothing is lost if the program closes. |
| `load_or_rollover(memory)` | Called at the start of every turn. If it's still the same day, restores any state saved earlier today (e.g. after restarting the terminal). If the calendar date has changed, archives yesterday into history and starts today empty (budget carries over). |
| `get_history(limit)` | Returns saved past days, most recent first. |

### Food database functions (`food_database.py`)

| Function | What it does |
|---|---|
| `find_food(query)` | Matches free-text like "3 boiled eggs" to a canonical food entry — whole-word matching with typo tolerance, and prefers the most specific match when several could apply (see Failure 1 & 2 below for why this matters). |
| `add_custom_food(...)` | Adds a new entry to the in-memory database AND persists it to `custom_foods.json` so it survives restarts. |

---

## 5. Memory — two layers

1. **`SessionMemory`** (in-process) — the current conversation's budget
   and meal list. This is what makes "add eggs" → "can I eat a banana"
   → correct remaining-calorie math work, all within one chat.

2. **`daily_store.py`** (on disk, `daily_log.json`) — survives closing
   the program entirely:
   - Reopen the terminal/notebook the same day → today's meals are
     still there.
   - Come back tomorrow → yesterday's log is archived into history,
     today starts empty, and the budget carries over automatically.

**Proof memory is really used, not just displayed:** `check_snack_fit`
never asks the user how many calories they've had — it reads
`_memory.remaining_calories()` directly, which only has the correct
value because of meals added in earlier turns.

---

## 6. Group of 3 Add-on — Macro Split

After meals are logged, the agent computes:

```
Protein calories = protein_g × 4
Carb calories    = carbs_g   × 4
Fat calories      = fat_g     × 9

Protein % = protein_cal / total_macro_cal × 100
Carbs %   = carbs_cal   / total_macro_cal × 100
Fat %     = fat_cal     / total_macro_cal × 100
```

Always labelled as an approximate, informational estimate — never
presented as medically exact.

---

## 7. Honest Failures Found & Fixed

### Failure 1 — false-positive substring matching
**What happened:** `find_food()` originally checked
`if alias in query` as a raw substring. The query `"pintola high
protien muesli"` matched the alias `"roti"` — because the letters
r-o-t-i happen to appear inside "p**roti**en", a pure text coincidence.
The wrong food got logged silently (Roti, 104 kcal, instead of the
actual muesli).
**Fix:** switched to whole-word tokenized matching — an alias only
matches if every one of its words appears as a standalone word in the
query, never buried inside an unrelated word.

### Failure 2 — typo + an over-generic fallback winning
**What happened:** even after fixing Failure 1, a custom food named
"pintola high protein muesli" plus a typo ("protien") still matched the
generic single-word alias "muesli" (→ Oats) instead of the user's own
4-word custom entry, because "protien" ≠ "protein" exactly.
**Fix:** added typo tolerance (`difflib.SequenceMatcher`, ≥0.8
similarity, for words 4+ letters) AND a "most specific alias wins"
rule — a 4-word match now always beats a 1-word match when both
technically fire.

### Failure 3 — intent-ordering bug ("Reset my meals")
**What happened:** the regex for showing a summary included a pattern
that also matched inside "Reset **my meal**s" — and because that
pattern was checked before the reset pattern, typing "Reset my meals"
showed a summary instead of actually resetting.
**Fix:** reordered the intent-checking list (more specific/rarer
commands checked first) and tightened the overly broad pattern.
**Found via:** an automated regression test script (see `TESTING.md`) —
not just manual clicking, which is why it was caught at all.

### Failure 4 — online lookup failing ≠ food doesn't exist
**Situation:** if the online fallback (Open Food Facts) times out or
errors, that's a NETWORK problem, not proof the food is invalid.
**Fix:** `lookup_calories` distinguishes an `api_error` status
internally and tells the user the *lookup* failed (with the real
reason), never claiming the food itself is unknown.

### Failure 5 — no budget set
**Situation:** user asks "Can I eat a banana?" before ever setting a
daily budget.
**Fix:** `check_snack_fit` returns a clear `no_budget` status and asks
the user to set one first, instead of crashing or guessing.

---

## 8. How to Run — every option

### Prerequisites
```bash
pip install -r requirements.txt
```
No API keys or accounts needed for any of the options below.

### Option A — Terminal
```bash
python agent.py
```

### Option B — Jupyter Notebook (best for demonstrating the trace)
```bash
jupyter notebook T17_Meal_Calorie_Helper.ipynb
```
Run cells in order. Each `agent.chat(...)` cell prints the full
`[PLAN]/[ACT]/[TOOL RESULT]/[OBSERVE]/[MEMORY]` trace.

### Option C — Import programmatically
```python
import tools
from agent import MealCaloricAgent

tools.reset_session()
agent = MealCaloricAgent(verbose=True)

agent.chat("My calorie budget is 2200 kcal")
agent.chat("Add 3 boiled eggs")
agent.chat("Can I eat a banana?")
```

### Option D — Web UI
```bash
python app.py
```
Then open **http://127.0.0.1:5000**. Gives you a chat window plus a
live "Today" sidebar (budget bar, macro breakdown, numbered meal
ledger with delete buttons), a History panel showing past days, and a
"Show reasoning ▾" toggle under every reply that reveals the exact
plan-act trace for that turn.

### File structure
```
T17_Meal_Calorie_Helper/
├── agent.py                 # Decision layer — plan-act loop
├── tools.py                 # 9 tool functions
├── memory.py                # SessionMemory — this conversation's state
├── daily_store.py           # Cross-day persistence + auto-rollover
├── food_database.py         # Built-in foods + custom-food storage
├── config.py                # Online nutrition fallback settings
├── app.py                   # Flask backend for the web UI
├── frontend/index.html      # Browser chat UI
├── custom_foods.json        # Auto-created — your saved custom foods
├── daily_log.json           # Auto-created — today's log + day history
├── T17_Meal_Calorie_Helper.ipynb
├── TESTING.md                # Full test checklist
└── README.md                 # This file
```

---

## 9. Supported Foods

Eggs, chicken breast, chicken leg, paneer, milk, skimmed milk, curd,
cheese, butter, rice, brown rice, roti/chapati, bread, oats/muesli,
poha, soya chunks, dal, rajma, chickpeas, banana, apple, orange, mango,
grapes, potato, sweet potato, spinach, broccoli, carrot, tomato, onion,
almonds, peanuts, walnuts, whey protein, orange juice, green tea, and
more — plus anything you add yourself via `add custom food`, or that's
found through the online fallback and saved with `save this food`.

---

*This project was developed for CSE 476 — Agentic AI and Intelligent
Automation, CA1 Assignment, T17: Meal Calorie Helper.*
