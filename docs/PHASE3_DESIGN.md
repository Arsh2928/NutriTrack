# Phase 3 - shared state, tools and logging

## What exists now
```
nutritrack/core/      quantity.py  nutrition.py        parse "two roti" / "250gm curd"; resolve food + amount -> nutrition
nutritrack/state/     nutrition_state.py  run_log.py   the ONE shared state; append-only logs/agent_runs.jsonl
nutritrack/tools/     22 tools in 5 modules + base.py  what the agents will call (Phases 4-5)
```

## The shared state (`NutritionState`, one per user, `data/users/<user_id>.json`)
`snapshot()` returns the structure from the brief: `targets, consumed, remaining, meals, weekly_history,
recommendations, agent_results, workflow_status` (+ `preferences`, `targets_source`, `meta`).

| Guarantee | How |
|---|---|
| Totals can never disagree with the meals | `consumed`/`remaining` are **never stored** - recomputed from the meal list on every read |
| A failed save is never silent, memory and disk never diverge | every change is a transaction: validate -> change -> atomic save; on any error roll back and raise `StateUpdateError` |
| No duplicate meals | `request_id` idempotency; tools derive one per run, so an LLM that calls `add_meal_to_state` twice logs once; eating the same food again in a new run is allowed |
| No fabricated history | a day exists only if meals were saved for it; no records = "no data", never 0 kcal |
| Users are separate | one file per user; ids validated (`[A-Za-z0-9_-]{1,64}`) |
| Corrupt file is not destroyed | renamed `*.corrupt-<time>`, warning surfaced in `snapshot()["meta"]` |
| Run data is not persisted | `agent_results`, `workflow_status`, open questions, candidate pool reset on restart |

Targets: only `demo_user` starts with **labelled demonstration defaults** (2300 kcal / 150 g, source `demo_default`);
every other user has none and is asked. Accepted ranges (500-10000 kcal, 10-500 g) only catch typos.
A day keeps the targets in effect when it was saved. CA1 history is imported once (`data/daily_log.json`),
skipping CA1's empty archived days.

## Tool contract (`tools/base.py`)
* Every tool takes an explicit `ctx` first (user, run id, state, log). `bind(tool, ctx)` removes it from the signature -
  that is what an agent receives. **No globals, no context variables.**
* Returns `{"status": "success"|"failed", "code", "message", ...}`; user-fixable problems carry
  `needs_user_input=True` + a `question`. Exceptions never escape (`internal_error`, `bad_arguments`, `state_update_failed`).
* The decorator logs every call (agent, tool, input/output summary, status, error, duration) and records it in the state.
* **The LLM never supplies numbers:** `add_meal_to_state` takes only `{food, quantity, unit}` and recomputes nutrition.
  Recommendation candidates are passed between tools as ids (`c07`).

| Agent | Tools |
|---|---|
| Meal Intake | parse_meal_entry, lookup_food_nutrition, add_meal_to_state, remove_meal_from_state, *register_custom_food* |
| Nutrition Analysis | calculate_daily_totals, calculate_macro_split, get_weekly_trends, compare_intake_with_targets |
| Goal and Budget | set_calorie_target, set_protein_target, calculate_remaining_targets, check_meal_against_budget |
| Meal Recommendation | get_recommendation_candidates, estimate_meal_nutrition, rank_meal_candidates, validate_recommendation_fit |
| Nutrition Manager | read_shared_state, delegate_to_specialist, validate_agent_results, update_workflow_status, append_run_log |

## Changes to CA1 code (defects fixed at the root, shared by CA1 and CA2)
| Defect | Fix |
|---|---|
| D1 `200 g banana` = 21,000 kcal | `core/nutrition.py` converts only when it is exact (g<->kg, ml<->l/cup/tbsp/tsp); otherwise asks |
| D2 `2 curd` = 2 kcal | bare number on a weight/volume food -> asks for the unit |
| D3 zero / negative / absurd quantities, "two" read as 1 | parser rejects them; understands number words, fractions, trailing quantities |
| D4 brand matched to plain food | `match_details()` reports leftover words; any unexplained word -> asks |
| D5 budget lost, silent / non-atomic saves, shared default | `daily_store.py`: restore budget, atomic write, logged failures, quarantine, deepcopy |
| D6 float drift | `memory.py`: totals computed from the meal list |
| D7 online lookup on by default | opt-in via `NUTRITRACK_ONLINE_LOOKUP=1` |

## Assumptions to review (all in one place each)
* Weekly "within target" = calories **at or below** that day's target. Unusual days: under 50% / over 125% of the
  day's target (or 50% / 150% of the user's median if no target, 3+ days); today is never flagged. (`analysis_tools.py`)
* Recommendation scoring weights and the 600 kcal "meal size" (`recommendation_tools.py`); portion lists and food
  roles (`recommendation_data.py`) - classification only, no nutrition values.
* 26 of the 37 built-in foods use reference values, not CA1's originals (see `food_database.py` header).

## Known limits
* Custom foods are shared by all users (meals and targets are per user).
* One process owns the data: thread-safe, but two processes writing the same user file is last-write-wins.
* `delegate_to_specialist` calls a **synchronous** runner; Phase 5 adapts it once the framework's async/tool behaviour is verified.
* Tool schemas for Microsoft Agent Framework are **unverified** (framework not installable in the build environment).
