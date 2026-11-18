# NutriTrack — Complete Presentation & Viva Reference Guide
**Models, Agents, Functions (with Exact Code Locations) & Data Storage Architecture (Locations + Schemas)**

---

## Part 1: AI Models, Providers & Routing Architecture

NutriTrack enforces a **Strict 3-Mode AI Architecture** (`gemini`, `groq`, `offline`) built on the **Microsoft Agent Framework (`agent_framework`)** and **OpenAI Async SDK (`AsyncOpenAI`)**. There is no hidden "Automatic" or cross-provider silent fallback.

### 1.1 All Configured Models & Engines

| Mode / Provider | Default Model ID | Base URL / Runtime | Where Configured in Code | Purpose & How It Works |
|---|---|---|---|---|
| **1. Google Gemini** (`mode = "gemini"`) | `gemini-flash-lite-latest` *(auto-normalizes legacy `.env` aliases like `gemini-2.5-flash`, `gemini-2.0-flash`, `gemini-1.5-flash`)* | `https://generativelanguage.googleapis.com/v1beta/openai/` | `nutritrack/ai/config.py` (`L26–L70`), `nutritrack/ai/provider_factory.py` (`L212–L248`) | Cloud LLM via Google's OpenAI-compatible Chat Completions endpoint wrapped in `OpenAIChatCompletionClient`. Preserves Gemini `extra_content` (`thought_signature`) across tool turns via `_THOUGHT_SIGNATURES`. |
| **2. Groq Cloud** (`mode = "groq"`) | `openai/gpt-oss-120b` *(auto-normalizes retired aliases like `llama-3.3-70b-versatile`, `llama-3.1-70b-versatile`)* | `https://api.groq.com/openai/v1` | `nutritrack/ai/config.py` (`L29–L70`), `nutritrack/ai/provider_factory.py` (`L249–L283`) | High-speed cloud LLM via Groq's OpenAI-compatible endpoint wrapped in `OpenAIChatCompletionClient`. |
| **3. Offline / Data-Only Engine** (`mode = "offline"`, default) | `deterministic-data-only-v1` / `deterministic-tool-planner-v1` | Local Python process (`DeterministicNutriTrackChatClient`) | `nutritrack/ai/data_only_service.py` (`L1–L101`), `nutritrack/agents/runtime.py` (`L742–L1251`) | **100% local deterministic MAF `BaseChatClient` + `FunctionInvocationLayer`**. Executes all 5 agents and all 22 Python tools locally with zero external network calls (`assert_network_allowed()` hard-blocks outbound sockets). |
| **4. Multi-Objective Recommendation Scoring Model** | Weighted linear utility model (`score` in `[0, 1]`) | Local Python (`_score()` in `recommendation_tools.py`) | `nutritrack/tools/recommendation_tools.py` (`L34–L109`) | `score = 0.50 * protein_score + 0.25 * calorie_fill + 0.15 * variety + 0.10 * preference`, with `MEAL_TARGET_KCAL = 600.0`, `MIN_FILL = 0.35`, `MAX_PER_MAIN_FOOD = 4`. |
| **5. Lexical & Fuzzy Information Retrieval (IR) Model** | Token specificity + `SequenceMatcher` (`>= 0.80`) + `get_close_matches` (`>= 0.60`) | Local Python (`match_details()`, `did_you_mean()`) | `nutritrack/ca1/food_database.py` (`L317–L413`), `nutritrack/core/nutrition.py` (`L75–L91`) | Ranks food aliases by `(len(alias_words), len(alias))`, blocks look-alike composites via `_NOT_SAME_AS`, prevents known-word hijacking via `exact_vocab`, and flags leftover unknown tokens as `low_confidence_match`. |

### 1.2 AI Routing, Resilience & Isolation Functions (`nutritrack/ai/`)

| File | Key Classes & Functions | Exact Lines | What to Explain in Viva |
|---|---|---|---|
| `nutritrack/ai/config.py` | `normalize_ai_mode()`, `normalize_provider_model()`, `load_persisted_mode()`, `save_persisted_mode()`, `ai_mode_scope()`, `turn_execution_scope()`, `is_turn_cancelled()`, `remaining_turn_timeout()`, `assert_network_allowed()`, `AIProviderConfig.from_env()` | `L43–L405` | Central policy guard. Enforces `VALID_AI_MODES = ("gemini", "groq", "offline")`, binds per-request `ContextVar` mode and deadline scopes, and blocks any cross-provider or Offline external network call. |
| `nutritrack/ai/provider_factory.py` | `_wrap_sdk_chat_completions()`, `_wrap_live_maf_client()`, `ProviderFactory.build_client()` | `L30–L289` | Builds fresh `AsyncOpenAI` + `OpenAIChatCompletionClient` per call (closed in `finally:` so `httpx` never leaks event loops). `_wrap_live_maf_client()` runs intermediate NutriTrack tool steps deterministically and invokes the cloud LLM **once** at final synthesis (~300 tokens). |
| `nutritrack/ai/provider_manager.py` | `RoutedNutriTrackChatClient`, `AIProviderManager.route_chat_request()`, `_invoke_cloud_provider()`, `_attempt_provider()`, `async_probe_providers()`, `set_mode()` | `L48–L750` | Shared MAF chat client across all 5 agents. Routes requests strictly to the user's selected mode, coordinates retries and circuit breakers, and logs provider events. |
| `nutritrack/ai/circuit_breaker.py` | `CircuitBreaker.allow_request()`, `record_success()`, `record_failure()`, `release_probe()`, `reset()` | `L11–L168` | 3-state machine (`closed` -> `open` after 3 failures or immediate trip -> `half_open` after 60s cooldown with single-probe lock). |
| `nutritrack/ai/failure_classifier.py` | `ClassifiedFailure`, `classify_provider_failure()` | `L1–L409` | Classifies HTTP/socket exceptions into `invalid_api_key` (401), `permission_denied` (403), `unsupported_model` (404), `quota_exhausted_daily` (429), `rate_limit_temporary` (429), `malformed_tool_call` (400), `server_error` (5xx), `timeout`, `dns_failure`, or `internet_disconnected`. |
| `nutritrack/ai/retry_policy.py` | `RetryPolicy.evaluate()` | `L1–L80` | Exponential backoff with jitter (`base=1.0s`, `max=8.0s`, `max_retries=2`), honoring upstream `Retry-After` headers and never retrying config/auth/daily-quota errors. |
| `nutritrack/ai/provider_health.py` & `provider_status.py` | `ProviderHealthTracker`, `ProviderStatusSnapshot`, `compute_effective_label()` | `L1–L135`, `L1–L83` | Tracks per-provider health (`healthy`, `available`, `rate_limited`, `quota_exhausted`, `config_error`, `unreachable`, `circuit_open`, `disabled_offline`) and computes UI badge labels (`Gemini Connected`, `Groq Connected`, `Offline Mode Active`). |

---

## Part 2: All 5 Microsoft Agent Framework (MAF) Agents & 22 `@agent_tool` Functions

| Agent Key & Name | Agent Module | Registered `@agent_tool` Functions & Tool Module | Causal Responsibility |
|---|---|---|---|
| **Agent 5 (Coordinator)**: `nutrition_manager` — **Nutrition Manager Agent** | `nutritrack/agents/nutrition_manager_agent.py` (`L23–L378`) | `nutritrack/tools/manager_tools.py` (`L60–L249`): `read_shared_state` (`L61`), `delegate_to_specialist` (`L81`), `validate_agent_results` (`L125`), `update_workflow_status` (`L211`), `append_run_log` (`L227`) | Plans required specialists (`plan_specialists_for_request()`), delegates in strict causal order (`meal_intake` -> `nutrition_analysis` -> `goal_budget` -> `meal_recommendation`), cross-checks recomputed meal sums against `snapshot()`, and compiles the final response. |
| **Agent 1 (Worker)**: `meal_intake` — **Meal Intake Agent** | `nutritrack/agents/meal_intake_agent.py` (`L18–L71`) | `nutritrack/tools/intake_tools.py` (`L29–L223`): `parse_meal_entry` (`L30`), `lookup_food_nutrition` (`L69`), `add_meal_to_state` (`L98`), `remove_meal_from_state` (`L144`), `register_custom_food` (`L174`) | Splits natural-language meal text, resolves nutrition from the database, saves or deletes meals in `NutritionState`, and registers user-defined custom foods after physical-plausibility checks (`<= 10` kcal/g). |
| **Agent 2 (Worker)**: `nutrition_analysis` — **Nutrition Analysis Agent** | `nutritrack/agents/nutrition_analysis_agent.py` (`L18–L70`) | `nutritrack/tools/analysis_tools.py` (`L61–L285`): `calculate_daily_totals` (`L62`), `calculate_macro_split` (`L74`), `get_weekly_trends` (`L199`), `compare_intake_with_targets` (`L248`) | Computes daily calorie/macro/fiber totals, 4/4/9 caloric percentages, honest multi-day trends (empty days are **never** averaged as 0 kcal), unusual-day detection, and week-over-week deltas. |
| **Agent 3 (Worker)**: `goal_budget` — **Goal & Budget Agent** | `nutritrack/agents/goal_budget_agent.py` (`L18–L69`) | `nutritrack/tools/budget_tools.py` (`L32–L128`): `set_calorie_target` (`L33`), `set_protein_target` (`L44`), `calculate_remaining_targets` (`L55`), `check_meal_against_budget` (`L89`) | Updates daily calorie/protein targets, computes unclipped remaining budget, and simulates whether a proposed meal fits remaining calories. |
| **Agent 4 (Worker)**: `meal_recommendation` — **Meal Recommendation Agent** | `nutritrack/agents/meal_recommendation_agent.py` (`L18–L69`) | `nutritrack/tools/recommendation_tools.py` (`L166–L382`): `get_recommendation_candidates` (`L167`), `estimate_meal_nutrition` (`L243`), `rank_meal_candidates` (`L281`), `validate_recommendation_fit` (`L334`) | Executes the 4-step deterministic recommendation pipeline: builds role-template combinations (`c01..c35`), recomputes macros, scores/diversifies options, and re-validates fit from scratch before saving to `NutritionState`. |

---

## Part 3: Where All Data Is Saved & In Which Format

| # | Data Store Purpose | Exact File Path on Disk | Storage Format | Owner Module & Functions |
|---|---|---|---|---|
| **1** | **Per-User Shared Nutrition State** | `data/users/<user_id>.json` *(+ `.lock`)* | **UTF-8 JSON Object** (`schema_version: 1`) | `nutritrack/state/nutrition_state.py` (`StateManager._load_or_create`, `NutritionState._persist`) |
| **2** | **Per-User Profile & Daily Water Logs** | `data/profiles/<user_id>.json` *(+ `.lock`)* | **UTF-8 JSON Object** | `nutritrack/web/app.py` (`_load_profile`, `_save_profile`, `nt_water`) |
| **3** | **Per-User Custom Foods Database** | `data/custom_foods/<user_id>.json` *(and `data/custom_foods.json` for `demo_user`)* | **UTF-8 JSON Object** | `nutritrack/ca1/food_database.py` (`custom_path_for_user`, `add_custom_food`, `delete_custom_food`) |
| **4** | **Durable Idempotency & Operation Store** | `data/operations/<user_id>.json` *(+ `.lock`)* | **UTF-8 JSON Object** (max 200 ops/user) | `nutritrack/state/operations.py` (`DurableOperationStore.begin`, `complete`) |
| **5** | **Persisted AI Execution Mode** | `data/ai_mode_settings.json` *(+ `.lock`)* | **UTF-8 JSON Object** | `nutritrack/ai/config.py` (`load_persisted_mode`, `save_persisted_mode`) |
| **6** | **Registered User Credentials** | `data/auth/users.json` *(+ `.lock`)* | **UTF-8 JSON Object** (PBKDF2-HMAC-SHA256) | `nutritrack/web/auth.py` (`AuthManager.register`, `authenticate`) |
| **7** | **Active Authentication Sessions** | `data/auth/sessions.json` *(+ `.lock`)* | **UTF-8 JSON Object** (24h TTL + CSRF token) | `nutritrack/web/auth.py` (`create_session`, `verify_session`, `revoke_session`) |
| **8** | **Session HMAC Signing Key** | `data/auth/session_secret.key` *(+ `.lock`)* | **64-char Hex ASCII Text** (32 random bytes, `0o600`) | `nutritrack/web/auth.py` (`_get_signing_key`) |
| **9** | **Login Rate-Limit Buckets** | `data/auth/rate_limits.json` *(+ `.lock`)* | **UTF-8 JSON Object** (60s sliding window) | `nutritrack/web/auth.py` (`_check_login_rate_limit`, `_record_login_failure`) |
| **10** | **Multi-Agent Execution Audit Log** | `logs/agent_runs.jsonl` *(rotates `.1`..`.3` at 5 MB)* | **Append-Only JSON Lines (`.jsonl`)** | `nutritrack/state/run_log.py` (`RunLog.append`, `read`) |
| **11** | **AI Provider Routing & Circuit Event Log** | `logs/ai_provider_events.jsonl` *(rotates at 5 MB)* | **Append-Only JSON Lines (`.jsonl`)** | `nutritrack/ai/provider_manager.py` (`AIProviderManager._log_event`) |
