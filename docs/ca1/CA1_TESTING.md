# T17 Meal Calorie Helper — Full Test Checklist

Run these in order (terminal `python agent.py`, notebook, or the web UI at
`http://127.0.0.1:5000`). Each block sets up the state the next block
needs, so don't skip around. This doubles as a viva script — every
command here maps to a rubric requirement.

Tip: in the web UI, click **"Show reasoning ▾"** under any agent reply to
see the exact plan-act trace for that command.

---

## 1. Budget + basic memory

| # | Type this | Expect |
|---|---|---|
| 1.1 | `My calorie budget is 2200 kcal` | Confirms budget set to 2200 |
| 1.2 | `Show today's summary` | "Nothing logged yet" but budget shows 2200 |

**Proves:** memory persists a value (budget) and a later command reads it back.

---

## 2. Adding meals (local database)

| # | Type this | Expect |
|---|---|---|
| 2.1 | `Add 3 boiled eggs` | ~234 kcal, updates running total |
| 2.2 | `Add 200 g chicken breast` | ~330 kcal, total now ~564 kcal |
| 2.3 | `Add 2 slices of bread` | ~158 kcal added |
| 2.4 | `Show today's summary` | Lists all 3 meals, correct running total, remaining kcal, macro % split |

**Proves:** tool #1 (`lookup_calories`) and #2 (`add_meal`) both get called for real, and results chain — `add_meal` reads `lookup_calories`'s output before deciding what to store.

---

## 3. Snack-fit decision (the "must be agentic" requirement)

| # | Type this | Expect |
|---|---|---|
| 3.1 | `Can I eat a banana?` | Should say it fits (plenty of budget left), shows the exact math |
| 3.2 | `Add 10 boiled eggs` *(deliberately overshoot)* | Adds ~780 kcal, likely pushes near/over budget |
| 3.3 | `Can I eat 200 g chicken breast?` | Should now say it does NOT fit, with the real numbers |

**Proves:** the plan-act loop — look up calories, read remaining budget from memory, compute, decide — not a canned reply. 3.1 and 3.3 must give *opposite* verdicts using the same food-lookup tool, which only happens if it's actually computing each time.

---

## 4. Delete a meal (bug-fix feature)

| # | Type this | Expect |
|---|---|---|
| 4.1 | `Show today's summary` | Note the numbered list (e.g. 1. Egg, 2. Chicken, 3. Bread, 4. Egg×10) |
| 4.2 | `delete meal 4` | Confirms removal, recalculates totals |
| 4.3 | `remove chicken` | Removes the most recent "chicken" entry by name |
| 4.4 | `Show today's summary` | Confirms both are gone, totals correct |

**Proves:** a tool can mutate memory and later reads reflect the change correctly.

---

## 5. Reset (the bug we just found and fixed)

| # | Type this | Expect |
|---|---|---|
| 5.1 | `Reset my meals` | ⚠️ **This exact phrase used to wrongly trigger "show summary" instead of resetting — confirm it now says "Daily meal log cleared"** |
| 5.2 | `Show today's summary` | "Nothing logged yet", budget still 2200 (reset keeps budget) |

**Proves:** intent detection correctly disambiguates similar-sounding phrases — good one to mention in viva as an "honest failure you caught and fixed."

---

## 6. Custom food (your own dishes / branded products)

| # | Type this | Expect |
|---|---|---|
| 6.1 | `add custom food: mom's rajma \| calories=320 \| protein=14 \| carbs=45 \| fat=8 \| serving=1 bowl (250g)` | Confirms saved to database |
| 6.2 | `Add 1 mom's rajma` | Logs exactly 320 kcal — found LOCALLY now, no internet needed |
| 6.3 | Restart Python (`quit()` then `python agent.py` again, or restart the notebook kernel) | `Add mom's rajma` should still work — proves it's saved to disk (`custom_foods.json`), not just memory |

**Proves:** genuinely durable custom-food storage, not just session-only.

---

## 7. Online API fallback (no key needed — Open Food Facts)

| # | Type this | Expect |
|---|---|---|
| 7.1 | `Add 100g Nutella` *(or any specific branded product not in your local DB)* | Should attempt an online lookup automatically (needs real internet on your machine) |
| 7.2 | If found online | Response includes a note that it was fetched online, and suggests `save this food` |
| 7.3 | `save this food` | Persists it locally — try 7.1's food again, should now say "local_database" not "api" in the trace |
| 7.4 | Try a nonsense food: `Add qwxyz123abc` | Should clearly say not found, and suggest `add custom food` — must NOT crash |

**Proves:** the 3-branch decision (local → online → manual) that's the strongest "real agent, not chatbot" evidence in this project.

---

## 8. Day rollover + persistence (can't wait a real day — simulate it)

| # | Do this | Expect |
|---|---|---|
| 8.1 | Log a couple of meals today, note the total | — |
| 8.2 | Close Python. Open `daily_log.json` in a text editor. Change `"current_date"` to yesterday's date (e.g. if today is `2026-09-01`, set it to `2026-08-31`) | — |
| 8.3 | Restart the agent, send any message (e.g. `hi`) | Should print/show a "🌅 New day detected" banner |
| 8.4 | `Show today's summary` | Today is now EMPTY, but budget carried over |
| 8.5 | `show my history` | Yesterday's meals + totals now appear under history |

**Proves:** the auto-reset-next-day feature actually works, tested deterministically without waiting.

---

## 9. Web frontend (if using `app.py`)

| # | Do this | Expect |
|---|---|---|
| 9.1 | `python app.py`, open `http://127.0.0.1:5000` | Page loads, chat panel + "Today" sidebar visible |
| 9.2 | Send several messages back to back | Chat window scrolls smoothly, stays scrolled to latest message (this was the bug we just fixed — confirm it's smooth now) |
| 9.3 | Add a few meals via chat | Sidebar's budget bar, macro bars, and meal ledger update live after every message |
| 9.4 | Click the ✕ next to any meal in the ledger | Deletes it (calls the same `delete_meal` tool as typing `delete meal N`) |
| 9.5 | Click "Show past days ▾" in the sidebar | Expands history; click a specific day to see the actual foods logged that day (not just the total) |
| 9.6 | Click "Show reasoning ▾" under any agent reply | Reveals the `[PLAN]/[ACT]/[TOOL RESULT]/[OBSERVE]` trace for that exact turn |
| 9.7 | Check the small badge under the header | Should say "Rule-based agent (no GITHUB_TOKEN set)" until you configure the LLM (see below) |

---

## 10. LLM agent (optional — needs a free GitHub token, see below)

| # | Do this | Expect |
|---|---|---|
| 10.1 | Set `GITHUB_TOKEN`, restart `python app.py` | Server console prints "using the LLM agent"; badge in the UI turns green and shows the model name |
| 10.2 | `My budget is 2200, add 3 eggs, and tell me if I can also have a banana` *(all in one message, deliberately)* | The LLM should decide on its own to call `set_budget`, then `add_meal`, then `check_snack_fit` — in one turn, without you specifying each step. This is the strongest possible viva demo since it's the model doing real multi-step planning, not you triggering pre-written intents one at a time |
| 10.3 | Compare the trace of 10.2 against any rule-based trace above | The LLM trace shows the MODEL choosing tools each round; the rule-based trace shows a fixed regex match — good to have both ready to contrast in viva |

---

## Quick automated version (paste into a fresh Python session)

```python
import sys; sys.path.insert(0, ".")
import tools
tools.reset_session()
from agent import MealCaloricAgent

agent = MealCaloricAgent(verbose=False)
checks = []

def check(name, cond):
    checks.append((name, cond))
    print(("PASS" if cond else "FAIL"), "-", name)

r = agent.chat_capture("My calorie budget is 2200 kcal")
check("set budget", "2200" in r["response"])

r = agent.chat_capture("Add 3 boiled eggs")
check("add meal", "234" in r["response"])

r = agent.chat_capture("Can I eat a banana?")
check("snack fit check", "kcal" in r["response"].lower())

r = agent.chat_capture("Show today's summary")
check("summary shows meal", "Egg" in r["response"])

r = agent.chat_capture("delete meal 1")
check("delete by index", "removed" in r["response"].lower())

r = agent.chat_capture("Reset my meals")
check("reset works (not swallowed by summary intent)", "clear" in r["response"].lower())

print(f"\n{sum(1 for _, c in checks if c)}/{len(checks)} passed")
```

This is exactly the script used to catch the "Reset my meals" bug — run it any time after making changes to confirm nothing regressed.
