# T17: Meal Calorie Helper — README

## 1. Project Name

**T17: Meal Calorie Helper**
CSE 476 — Agentic AI and Intelligent Automation | CA1 Assignment

---

## 2. Goal

Build a real AI agent that helps users track daily meals against a calorie budget. The agent:

- Remembers all meals logged during the conversation
- Uses tool functions (not text generation) to look up and store nutrition
- Makes multi-step decisions based on tool results and session memory
- Provides a macro split (protein / carbs / fat) after each food is logged
- Tells the user whether a planned snack fits their remaining budget

> ⚠️ **Informational Only**: All nutritional values are estimated. Actual values vary by brand, preparation, and serving size. This tool is not medical advice.

---

## 3. Tools Used

### Tool 1 — `lookup_calories(query)`

**Purpose:** Look up estimated calories and macronutrients for a food item from the built-in database.

**Input:** Natural-language food string — e.g. `"3 boiled eggs"`, `"200 g chicken breast"`, `"banana"`

**Output (dict):**
```python
{
  "status": "found",
  "food": "egg",
  "quantity_str": "3 pieces",
  "serving_size": "1 large egg (~50 g)",
  "calories": 234.0,
  "protein": 18.0,
  "carbs": 1.8,
  "fat": 15.9,
  "note": "Estimated nutritional values; actual values can vary..."
}
```

If the food is not in the database:
```python
{
  "status": "not_found",
  "error": "'xyz' is not in the food database. Please try a different name."
}
```

### Tool 2 — `add_meal(query)`

**Purpose:** Log a food to today's meal record and update running totals.

**Input:** Same natural-language food string as `lookup_calories`.

**What it does internally:**
1. Calls `lookup_calories()` to get nutrition data
2. Creates a `MealEntry` object
3. Appends it to `SessionMemory`
4. Updates running calorie and macro totals

**Output (dict):**
```python
{
  "status": "added",
  "meal_added": { "food": "egg", "quantity": "3 pieces", "calories": 234.0, ... },
  "daily_totals": { "calories": 234.0, "protein": 18.0, "carbs": 1.8, "fat": 15.9 },
  "remaining_kcal": 1966.0,
  "macro_split": { "protein_g": 18.0, "protein_pct": 30.7, ... }
}
```

### Tool 3 — `check_snack_fit(query)`

**Purpose:** Decide whether a planned snack fits the remaining calorie budget.

**Multi-step process inside the tool:**
1. Calls `lookup_calories()` for the snack
2. Reads `remaining_calories()` from `SessionMemory`
3. Computes `surplus_deficit = remaining - snack_calories`
4. Returns status `"fits"` or `"does_not_fit"`

### Tool 4 — `get_daily_summary()`

Returns the full session state including all logged meals, totals, and macro split.

### Tool 5 — `set_budget(budget_kcal)`

Sets the daily calorie budget in session memory.

### Tool 6 — `add_custom_food_tool(name, calories, protein, carbs, fat, serving_size)`

**Purpose:** Let the user permanently add their own food (e.g. a homemade dish like "mom's rajma") to the local database with manually supplied nutrition values.

**Chat syntax:**
```
add custom food: mom's rajma | calories=320 | protein=14 | carbs=45 | fat=8 | serving=1 bowl (250g)
```

The food is written to `custom_foods.json` on disk, so it is remembered in every future run — not just this session.

### Tool 7 — `save_last_lookup()`

**Purpose:** After a food is found through the **online API fallback** (see below), this permanently caches that exact result into the local database, so the next lookup for the same food is instant and works offline.

**Chat syntax:** `save this food`

### Tool 8 — `delete_meal(index, food_name)`

**Purpose:** Remove a single wrongly-logged meal from today's log — e.g. one added by mistake, or a genuine matcher bug like the one described in Failure 5 below — and recompute daily totals + macro split.

**Chat syntax:**
```
delete meal 2      # removes the 2nd item shown by "show today's summary"
remove roti         # removes the most recent meal matching "roti" by name
```
Only the matched meal is removed; the budget and every other logged meal stay untouched.

---

## 3b. Online API Fallback (optional)

If a food is missing from the local `food_database.py`, `lookup_calories()` now tries one more source before giving up. Two providers are supported (`config.py` → `NUTRITION_API_PROVIDER`):

| Provider | Key needed? | Notes |
|---|---|---|
| `openfoodfacts` (**default**) | **No** | Free, crowdsourced, no signup at all. Best for packaged/branded products. Zero setup friction — this is why it's the default. |
| `calorieninjas` | Yes (free) | Natural-language quantity parsing ("3 boiled eggs" understood directly). Now issued via **api-ninjas.com** — CalorieNinjas merged into that platform in 2025, so sign up there if the old calorieninjas.com key doesn't work anymore. |

**How it fits into the tool chain:**
```
lookup_calories(food)
   │
   ├── 1. Check local food_database.py  → found? return it (instant, offline)
   │
   ├── 2. Not found locally?
   │        → call the configured online provider
   │        → found online? scale to the requested quantity, return it,
   │            flagged source="api"
   │        → network/API error? return not_found with a clear reason
   │            (distinct from "food doesn't exist")
   │
   └── 3. Still nothing → return not_found with can_add_manually=True
            so the agent offers "add custom food"
```

**Setup:** nothing required for the default (Open Food Facts). To switch to CalorieNinjas/API Ninjas instead:
```bash
export NUTRITION_API_PROVIDER=calorieninjas
export CALORIE_API_KEY=your_key_here   # get one free at https://api-ninjas.com/register
```
No key configured and provider set to `calorieninjas`? Nothing breaks — the online step is simply skipped and the agent falls straight to "add custom food".

**Why this matters for the assignment:** it's a real extra plan-act branch — the agent doesn't just fail on an unknown food, it tries a second tool (the API), reads that tool's result, and only then decides whether to ask the user to add the food themselves. This is exactly the kind of "tool result changes what happens next" behaviour the rubric is looking for.

---

## 4. Memory

Memory is stored in the `SessionMemory` class (`memory.py`) — **not in LLM text context**.

### What is stored:

| Field | Type | Description |
|---|---|---|
| `daily_budget` | `float` | Calorie target for the day |
| `meals` | `list[MealEntry]` | All meals logged this session |
| `_total_calories` | `float` | Running kcal sum |
| `_total_protein` | `float` | Running protein sum (g) |
| `_total_carbs` | `float` | Running carbs sum (g) |
| `_total_fat` | `float` | Running fat sum (g) |

Memory persists across all turns in the same Python session. To start fresh, call `tools.reset_session()`.

---

## 4b. Day-to-Day Persistence & Auto-Reset

`SessionMemory` (above) only lives for one Python process. `daily_store.py` adds a second, disk-backed layer on top of it, in `daily_log.json`:

- **Same-day persistence** — every meal/budget change is written to disk immediately, so today's log survives closing and reopening the notebook, restarting the terminal agent, or restarting the Flask server. The next process to start restores today's meals automatically.
- **Automatic day rollover** — the first agent turn of a new calendar day detects the date has changed, archives yesterday's final totals into `history`, and starts a clean log for today. The daily **budget carries over** automatically (you don't have to re-set it every morning), but meals do **not** — today always starts empty.
- **History** — every past day is kept, so `show my history` (or `GET /api/history` in the web app) can show a running record, not just "today".

This is triggered automatically at the start of every `agent.chat(...)` call — nothing needs to be called manually. It's demonstrated by backdating `daily_log.json` and confirming the agent archives the old day and starts fresh (see Failure 7 below).

---

## 5. Agentic Behaviour

The agent follows a **Plan-Act-Observe-Respond** loop:

```
User Message
    │
    ▼
detect_intent()          ← PLAN: what does the user want?
    │
    ├── SET_BUDGET  → set_budget()
    ├── ADD_MEAL    → lookup_calories() → add_meal() → read memory
    ├── CHECK_SNACK → lookup_calories() → check_snack_fit() → read memory → decide
    ├── SHOW_SUMMARY → get_daily_summary()
    └── RESET        → memory.reset_meals()
    │
    ▼
Read tool result         ← OBSERVE: what did the tool return?
    │
    ▼
Compose response         ← RESPOND: use tool result + memory to answer
```

### Why it is an agent and not a chatbot

- The agent **calls tools** — tool results are real Python dicts, not generated text
- The agent **reads tool results** before deciding its next action
- The agent **uses memory** — `remaining_calories()` is computed from stored values, not recalled from text
- For snack decisions: the agent calls `lookup_calories()`, reads kcal, reads memory, does arithmetic, then decides — this is a multi-step reasoning chain with real computation
- For unknown foods: the agent tries the local database, then (if a key is configured) the online API, reads whether *that* succeeded or failed, and only *then* decides whether to prompt the user to add the food manually — a genuine three-branch decision driven entirely by tool results, not a scripted reply

---

## 6. Group of 3 Add-on — Macro Split

After every meal is logged, the agent computes and displays:

- Protein (g) and % of macro calories
- Carbohydrates (g) and % of macro calories  
- Fat (g) and % of macro calories

**Formulas used:**
```
Protein calories  = protein_g  × 4
Carb calories     = carbs_g    × 4
Fat calories      = fat_g      × 9

Protein % = protein_cal / total_macro_cal × 100
Carbs %   = carbs_cal   / total_macro_cal × 100
Fat %     = fat_cal     / total_macro_cal × 100
```

This is clearly labelled as approximate and informational.

---

## 7. Honest Failure and Handling

### Failure 1 — Unknown food

**Situation:** User asks to add or check a food not in the database (e.g., "samosa", "biryani", "pizza").

**How the system handles it:**
```python
# lookup_calories() returns:
{
  "status": "not_found",
  "error": "'biryani' is not in the food database. Please try a different name."
}
```
The agent reads `status == "not_found"` and tells the user the food is unavailable. It **never invents or guesses** nutritional values.

### Failure 2 — Ambiguous quantity

**Situation:** User says just "some chicken" without a quantity.

**How the system handles it:**  
The parser defaults to `1 piece` (100 g equivalent) and the agent includes the assumed quantity in its response so the user can correct it if needed.

### Failure 3 — Online API unreachable

**Situation:** A food isn't in the local database, an API key *is* configured, but the request fails (no internet, invalid key, timeout).

**How the system handles it:** `_fetch_from_api()` catches the exception and returns `{"status": "api_error", ...}` instead of letting the program crash. `lookup_calories()` treats this differently from "food doesn't exist" — it tells the user the *lookup* failed, not that the food is invalid, and still offers `add custom food` as a way forward.

### Failure 4 — No budget set

**Situation:** User asks "Can I eat a banana?" without setting a budget first.

**How the system handles it:**
```python
# check_snack_fit() returns:
{
  "status": "no_budget",
  "decision": "No daily calorie budget has been set. Please set your budget first."
}
```

### Failure 5 — False-positive food matching (found and fixed during testing)

**Situation:** `find_food()` originally matched aliases using a raw substring check (`alias in query`). This misfired on real input: the query `"pintola high protien muesli"` matched the alias `"roti"`, because the letters r-o-t-i happen to appear inside the word "p**roti**en" — a text coincidence with no relation to food. The agent silently logged the wrong food (Roti, 104 kcal) instead of the intended muesli.

**How it was found:** caught during manual testing in the notebook — the trace showed `[OBSERVE] Found 'roti'` for an input that clearly wasn't roti.

**How it was fixed:** `find_food()` now tokenizes the query into whole words and only matches an alias if *every word* of that alias appears as a standalone word in the query, never as a substring buried inside an unrelated word. `"muesli"` now correctly matches the `"muesli"` alias (→ Oats) instead of accidentally matching `"roti"`.

### Failure 6 — Specific custom food losing to a generic one-word alias (found and fixed during testing)

**Situation:** After the Failure 5 fix, a *different* mismatch showed up: a user added a custom food `"pintola high protein muesli"` and then logged `"add 100gm pintola high protien muesli"` (note the typo — "protien"). Because "protien" ≠ "protein", the 4-word custom-food alias didn't fully match, but the single word `"muesli"` still matched the generic built-in `"muesli"` alias → Oats. So a *more specific, user-added* food lost to a *more generic* fallback, and a small typo silently changed which food got logged.

**How it was found:** caught during manual testing — the response showed "Oats" instead of the just-added custom food.

**How it was fixed:** two changes to `find_food()`:
1. **Typo tolerance:** words of 4+ letters are compared with `difflib.SequenceMatcher` (similarity ≥ 0.8) instead of requiring an exact match, so "protien" now matches "protein".
2. **Most-specific-alias-wins:** when multiple aliases match, the one made of the *most words* is preferred. A 4-word custom food now always outranks a 1-word generic alias like "muesli" when both technically match, so a specific, user-added food is never silently overridden by a generic database entry.

### Failure 7 — Day rollover needed to be tested without waiting a real day

**Situation:** `daily_store.load_or_rollover()` is only exercised once per real calendar day in normal use, which makes it awkward to verify by hand — waiting until midnight isn't a practical test.

**How it was handled:** the rollover logic reads the "current date" purely from the persisted `daily_log.json`, never from a live clock check baked into memory. So it can be tested deterministically by editing `daily_log.json`'s `current_date` field back one day and then sending any message — the agent detects the mismatch against today's real date, archives the old entry into `history`, and starts a clean day. This was used to verify the feature (see the "Day-to-Day Persistence" section above) without any special test mode or waiting.

---

## 8. How to Run

### Prerequisites

```bash
pip install jupyter requests
```

**No API key needed.** The online fallback defaults to Open Food Facts, which requires no signup at all — the project works fully offline (local database + manual "add custom food") or with the free online fallback, out of the box.

### Option A — Run the Jupyter Notebook (Recommended)

```bash
cd T17_Meal_Calorie_Helper
jupyter notebook T17_Meal_Calorie_Helper.ipynb
```

Run cells in order. The notebook contains 5 scenarios with full plan-act traces.

### Option B — Run the Interactive Terminal Agent

```bash
cd T17_Meal_Calorie_Helper
python agent.py
```

Example session:
```
You: My calorie budget is 2200 kcal
You: Add 3 boiled eggs
You: Add 200 g chicken breast
You: Can I eat a banana?
You: Show today's summary
You: quit
```

### Option C — Import and Use Programmatically

```python
import sys
sys.path.insert(0, "T17_Meal_Calorie_Helper")

import tools
from agent import MealCaloricAgent

tools.reset_session()
agent = MealCaloricAgent(verbose=True)

agent.chat("My calorie budget is 2200 kcal")
agent.chat("Add 3 boiled eggs")
agent.chat("Can I eat a banana?")
```

### Option D — Web Frontend (browser UI)

A small Flask backend + single-page frontend, so the agent can be used from a browser instead of a notebook or terminal.

```bash
python app.py
```
Then open **http://127.0.0.1:5000** in your browser.

What it gives you beyond the terminal:
- A chat window on the left; a live "Today" panel on the right with a budget progress bar, a macro breakdown, and a numbered meal ledger — each meal has a **✕** button that calls `delete_meal` for you.
- Every agent reply has a **"Show reasoning ▾"** toggle that reveals the exact plan-act trace (`[PLAN]` → `[ACT]` → `[TOOL RESULT]` → `[OBSERVE]` → `[MEMORY]`) for that turn — useful for demonstrating the agentic behaviour in your viva without needing the notebook open.
- A **History** panel showing past (fully-closed) days.
- A banner that appears automatically when a new calendar day rolls over.

The frontend only talks to three endpoints (`POST /api/chat`, `GET /api/summary`, `GET /api/history`) — all the actual agent logic still lives in `agent.py`/`tools.py`, completely unchanged. `app.py` is a thin HTTP adapter, not a second implementation.


### File Structure

```
T17_Meal_Calorie_Helper/
├── food_database.py         # Built-in nutrition data + custom-food storage
├── memory.py                # SessionMemory class — stores all state
├── daily_store.py           # Cross-day persistence + auto-rollover
├── config.py                # Online nutrition API provider settings
├── tools.py                 # 9 tool functions the agent calls
├── agent.py                 # Plan-Act loop + intent detection
├── app.py                   # Flask backend for the web frontend
├── frontend/index.html      # Browser chat UI (talks to app.py)
├── custom_foods.json        # Auto-created; your saved custom/API foods
├── daily_log.json           # Auto-created; today's log + day history
├── T17_Meal_Calorie_Helper.ipynb  # Demo notebook
└── README.md                # This file
```

---

## Supported Foods (sample)

Eggs, chicken breast, chicken leg, paneer, milk, skimmed milk, curd, cheese, butter, rice, brown rice, roti/chapati, bread, oats, poha, soya chunks, dal, rajma, chickpeas, banana, apple, orange, mango, grapes, potato, sweet potato, spinach, broccoli, carrot, tomato, onion, almonds, peanuts, walnuts, whey protein, orange juice, green tea, and more.

---

*This project was developed for CSE 476 — Agentic AI and Intelligent Automation, CA1 Assignment, T17: Meal Calorie Helper.*
