# NutriTrack — Multi-Agent Nutrition Tracker (CA2)

NutriTrack is a verified multi-agent nutrition assistant built with **Flask**, **Microsoft Agent Framework (`agent-framework`)**, deterministic Python calculation tools, per-user JSON persistence (`data/users/<user_id>.json`), and append-only audit logs (`logs/agent_runs.jsonl`, `logs/ai_provider_events.jsonl`).

## Architecture Overview

1. **Five Microsoft Agent Framework Agents (`nutritrack/agents/`)**
   - `NutritionManagerAgent` (`nutrition_manager_agent.py`): Coordinator that reads shared state, selectively delegates only relevant specialists sequentially (`allow_concurrent_invocation=False`, bounded iterations), validates `run_id` results, and formats grounded replies.
   - `MealIntakeAgent` (`meal_intake_agent.py`): Parses food text, resolves items against `FOOD_DB` + `custom_foods.json`, logs/removes meals with deterministic date extraction (`YYYY-MM-DD`, slash dates, `today`, `yesterday`, `on <date>`), and asks the user when a food is unknown.
   - `GoalBudgetAgent` (`goal_budget_agent.py`): Sets daily calorie/protein targets, computes remaining budget, and checks whether proposed meals fit.
   - `MealRecommendationAgent` (`meal_recommendation_agent.py`): Runs the 4-stage candidate generation, constraint filtering (`diet`, `avoid`, `prefer`), scoring, and ground-truth re-validation pipeline.
   - `NutritionAnalysisAgent` (`nutrition_analysis_agent.py`): Computes daily totals (calories, protein, carbs, fat, dietary fiber), macro calorie splits (`4/4/9`), target comparisons, and honest partial-period weekly trends without counting missing days as zero.

2. **Strict 3-Mode AI Configuration (`nutritrack/ai/`)**
   - **`gemini`**: Calls Google Gemini exclusively with explicit configured model ID, per-provider circuit breaker, bounded transient retries, and fail-fast on HTTP 400/401/403/404 and daily quota exhaustion. Preserves `gemini` mode on failure.
   - **`groq`**: Calls Groq exclusively with strict configured model ID, fail-fast on HTTP 400 `tool_use_failed` / `failed_generation`, isolated breaker/retries, and preserves `groq` mode on failure.
   - **`offline`**: Uses deterministic local routing (`DataOnlyService`) with zero cloud LLM requests and blocks external food lookups.
   - **No automatic mode, no hidden providers, no cross-provider fallback, and no model auto-rotation.**

3. **Fast Deterministic Path & Production Concurrency (`nutritrack/tools/fast_log.py`, `nutritrack/state/`)**
   - UI meal diary mutations (`POST /api/nutritrack/meals`, `/api/nutritrack/meals/fast`, `PUT/DELETE`) execute `fast_log_meal` / `fast_remove_meal` directly in Python (`~14 ms` warm mean offline) while logging cleanly and monitoring latency (`PerformanceMonitor` at `/api/nutritrack/perf`).
   - Unknown foods never receive LLM-invented nutrition values; external USDA FDC / Open Food Facts lookups require explicit opt-in (`NUTRITRACK_ONLINE_LOOKUP=1` in cloud mode) and are never silently saved.
   - Concurrency & I/O safety: per-user `RLock` + cross-process advisory file locks (`nutritrack/state/io.py`), atomic temp-file + `os.replace()` writes, disk mtime change reloading, chunked EOF tail reads + 5 MiB rotation in `RunLog`, strict JSON/input validation, security headers, and `request_id` idempotency.

## Setup & Commands

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env   # Optional: set GEMINI_API_KEY / GROQ_API_KEY
```

- **Run Web Application (`http://127.0.0.1:5000`)**: `python app.py`
- **Run Full Test Suite**: `python -m pytest -v`
- **Execute Demo Notebook cleanly**: `jupyter nbconvert --to notebook --execute notebooks/CA2_Demo.ipynb`
- **Environment Overrides**: `NUTRITRACK_AI_MODE=gemini|groq|offline`, `NUTRITRACK_DATA_DIR`, `NUTRITRACK_LOG_DIR`, `NUTRITRACK_ONLINE_LOOKUP=1`
