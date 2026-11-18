# NutriTrack — Engineering Overhaul Report

## 1. Audit Summary & Root Causes Resolved

1. **Slow Diary Meal Logging (~76 s live cloud / 196.87 ms offline baseline):** `/api/nutritrack/meals` routed structured food-diary form entries through multi-agent MAF orchestration and multi-model retries/rotation. Added deterministic `fast_log_meal()` and `fast_remove_meal()` (`nutritrack/tools/fast_log.py`) for structured CRUD mutations, reserving MAF multi-agent orchestration for conversational `/api/nutritrack/chat`.
2. **Complex Multi-Provider Failover & Hidden Model Rotation:** Removed `automatic`/`online` modes, cross-provider switching, `openai`/`github` auto-detection, `_FALLBACK_MODELS`/`_RESOLVED_MODELS`, and online model discovery. Simplified `nutritrack/ai/` to three explicit isolated modes: **`gemini`**, **`groq`**, and **`offline`**.
3. **Groq HTTP 429 & HTTP 400 Malformed Tool-Call Handling:** Added explicit `malformed_tool_call` classification (`tool_use_failed`, `failed_generation`, `tool call validation failed`) failing fast without retries, scoped Gemini `thought_signature` (`extra_content`) handling strictly to `provider == "gemini"`, and preserved the user's selected mode and typed chat input on failure.
4. **Date Extraction & Snapshot Alignment Bugs:** Replaced bare `"yesterday"` string replacement with `extract_date_from_text()` (`YYYY-MM-DD`, slash dates, `today`, `yesterday`, `on <date>`), validating calendar dates (rejecting impossible dates like `2026-02-30` without leaving date tokens in `food_text`) while ignoring `7up` / `1/2 cup`, and aligned `NutritionManagerAgent.run_turn()` snapshots to extracted dates (`snap_day`).
5. **Missing Dietary Fiber Tracking:** Added full dietary `fiber` resolution across `FOOD_DB`, USDA FDC (`#291`/`1079`), Open Food Facts (`fiber_100g`), `custom_foods.json`, `NutritionState` (`day_totals["fiber"]`, `snapshot()["fiber"]`), `analysis_tools`, and the UI Food Diary (`#diary-side-fiber`), while keeping `snapshot()["consumed"]` strictly at its 4-key contract (`calories`, `protein`, `carbs`, `fat`).
6. **Concurrency, Stale Cache & Disk I/O Bottlenecks:** Added sibling `.lock` cross-process advisory locks (`msvcrt`/`fcntl` in `nutritrack/state/io.py`), automatic disk mtime reload in `StateManager.get()` and `_txn()`, per-user Flask `RLock` (`_user_lock`), `request_id` idempotency, insights caching (`_insights_cache`), reverse EOF chunked `RunLog` tail reads with 5 MiB size rotation, and opt-in fsync (`NUTRITRACK_LOG_FSYNC=1`).
7. **Duplicate Frontend Requests & Input Validation Gaps:** Eliminated duplicate `loadWeeklyInsights()` boot calls via single-flight promise deduplication and `insightsDirty` tracking; added `_parse_json_dict()`, `_resolve_user_id()`, JSON `404`/`405`/`500` error handlers, and response security headers (`X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, `Cache-Control: no-store` on `/api/*`).

## 2. Simplified 3-Mode AI Architecture & Fast-Path Tradeoff

| Mode | Behaviour | Failure Handling |
| --- | --- | --- |
| `gemini` | Google Gemini only, explicit configured model ID | Per-provider breaker + bounded transient retry; fail-fast on `400`/`401`/`403`/`404` & daily quota; mode stays `gemini`; returns HTTP 503 (`llm_unavailable`) |
| `groq` | Groq only, explicit configured model ID | Per-provider breaker + bounded retry; fail-fast on HTTP `400` `tool_use_failed`/`failed_generation` & auth errors; mode stays `groq`; returns HTTP 503 |
| `offline` | `DataOnlyService` deterministic local routing (`0` cloud calls, external food lookup blocked) | Never contacts cloud APIs |

**Design Tradeoff:** `/api/nutritrack/meals`, `/api/nutritrack/meals/fast`, and `PUT/DELETE /api/nutritrack/meals/<id>` use deterministic Python resolution directly (`fast_log_meal`/`fast_remove_meal`) for single-digit millisecond latency, while `/api/nutritrack/chat` retains the genuine 5-agent Microsoft Agent Framework coordinator/worker pipeline (`NutritionManagerAgent` + 4 specialists).

## 3. Quantitative Performance Comparison

All optimized timings were measured locally via `scratch/benchmark_stage8.py` in isolated temp storage under `NUTRITRACK_AI_MODE=offline` after final code edits.

| Operation | Baseline Reference | Optimized Cold | Optimized Warm Mean | Optimized Median | Optimized P95 |
| --- | --- | --- | --- | --- | --- |
| `POST /api/nutritrack/meals` (fast path) | 196.87 ms offline (~76 s live LLM) | 17.07 ms | 5.05 ms | 4.78 ms | 8.17 ms |
| `POST /api/nutritrack/chat` (MAF offline loop) | — | 124.15 ms | 118.70 ms | 123.59 ms | 155.27 ms |
| `GET /api/nutritrack/state` | 30.54 ms | 8.36 ms | 14.57 ms | 17.04 ms | 20.39 ms |
| `GET /api/nutritrack/insights` (7 days) | 8.34 ms | 3.25 ms | 0.70 ms | 0.66 ms | 0.92 ms |
| `RunLog.read(limit=25)` (1,560 lines) | 14.59 ms median (30.74 ms p95) | 14.26 ms | 0.22 ms | 0.21 ms | 0.34 ms |

## 4. Verification Status

- **Automated Test Suite (`pytest`):** Covers strict 3-mode isolation (`tests/test_ai_modes.py`, `tests/test_ai_failover.py`, `tests/test_ai_recovery.py`, `tests/test_provider_isolation.py`), date extraction / dietary fiber / `fast_log_meal` (`tests/test_date_fiber_fast_log.py`), cross-process locking / idempotency / tail log rotation / validation (`tests/test_concurrency_io_validation.py`), and core CA1/CA2 agent/web contracts.
- **Live HTTP & Headless Edge DOM:** All tested routes (`GET /api/*`, `POST /api/nutritrack/meals/fast`, `meals`, `chat`, `recommendations`, `goals`, `preferences`, `water`, `profile`, `custom-food`, `PUT`/`DELETE` meals, `404` JSON, and single non-duplicated `/api/nutritrack/insights` call on boot) verified cleanly, alongside clean headless `jupyter nbconvert --execute` execution of `notebooks/CA2_Demo.ipynb`.
