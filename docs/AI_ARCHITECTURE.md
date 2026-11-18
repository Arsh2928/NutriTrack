# NutriTrack Strict Three-Mode AI Architecture

## Core Invariant: Three Manual Modes Only

NutriTrack enforces exactly three user-selectable execution modes (`VALID_AI_MODES = ("gemini", "groq", "offline")`):

1. **`gemini`** — Routes LLM orchestration requests exclusively to Google Gemini (pinned `GEMINI_MODEL`). Never calls Groq, OpenAI, or GitHub Models, and never rotates models automatically.
2. **`groq`** — Routes LLM orchestration requests exclusively to Groq (pinned `GROQ_MODEL`). Never calls Gemini, OpenAI, or GitHub Models, and never rotates models automatically.
3. **`offline`** — Runs deterministically in local Python using the local food database and state engine. Zero external LLM calls and zero external nutrition API calls (`USDA FoodData Central` / `Open Food Facts` are strictly blocked).

There is **no** `automatic`, `auto`, `online`, or hidden provider mode, and API keys never auto-select a provider.

## Configuration Precedence & Persistence

- **Default First-Run Mode:** Always `offline` when no valid choice has been persisted and no explicit `NUTRITRACK_AI_MODE` (`gemini` | `groq` | `offline`) is configured.
- **Legacy Value Quarantine:** Any legacy `"automatic"` or `"online"` value in `ai_mode_settings.json` is quarantined/ignored and falls back to `"offline"`.
- **Fail-Loud Persistence:** Switching modes via `AIProviderManager.set_mode(mode, persist=True)` writes `ai_mode_settings.json` atomically under cross-process `user_file_lock()` **before** updating in-memory state. Disk write errors raise `ProviderConfigError` (`HTTP 500` on `/api/nutritrack/ai/mode`).

## Runtime Isolation & Centralized Network Guard

```text
HTTP Request -> _authorize_request() -> acquire selected_mode snapshot
                                              │
                     with ai_mode_scope(selected_mode):
                         ├── Straightforward chat / meal / summary -> route_chat_fast() [0 LLM calls]
                         └── Complex query / Recommendations       -> orch.run_turn() [pinned provider only]
```

- **Request-Scoped Mode Snapshot (`ai_mode_scope`)**: Prevents mid-request mode switches from causing mixed-provider calls inside a single turn.
- **Network Policy Guard (`assert_network_allowed`)**: Invoked inside `ProviderFactory._wrap_sdk_chat_completions`, `LoopSafeOpenAIChatClient`, `external_nutrition._http_get_json`, and `ca1/tools._fetch_from_api` to enforce:
  - `offline` blocks both external AI SDK calls and external food lookups (`PermissionError` / disabled).
  - `gemini` permits only Gemini AI calls (plus opted-in external food lookup if `NUTRITRACK_ONLINE_LOOKUP=1`).
  - `groq` permits only Groq AI calls (plus opted-in external food lookup if `NUTRITRACK_ONLINE_LOOKUP=1`).
- **Decommissioned Fourth Provider Path**: `GITHUB_TOKEN` auto-detection and `LLMMealAgent` initialization were removed from `/api/chat` and `llm_config.py`. Obsolete legacy mode requests (`github`, `openai`, `automatic`, `online`) return `HTTP 400`.
