# NutriTrack Measured Performance Benchmarks

## Measurement Environment & Methodology

All benchmarks below were directly measured after the final edits by running [scratch/benchmark_final.py](file:///C:/Users/arshd/.gemini/antigravity/brain/99ba1d79-256c-42ed-979a-7624f251a719/scratch/benchmark_final.py) against Flask `app.test_client()` backed by real disk persistence (`NUTRITRACK_DATA_DIR` / `NUTRITRACK_LOG_DIR` in an isolated temporary directory on Windows NTFS) with Offline mode active (`0` external cloud calls). No timings are extrapolated or simulated.

## End-to-End Endpoint Latency Results

| Operation / Endpoint | Cold (ms) | Warm Mean (ms) | Median (ms) | p95 (ms) | Min (ms) | Max (ms) | LLM Calls | Agents Used |
|---|---|---|---|---|---|---|---|---|
| `GET /api/nutritrack/foods` (catalog) | 0.85 | 0.65 | 0.63 | 0.72 | 0.58 | 1.01 | 0 | Direct DB |
| `GET /api/nutritrack/insights` (7d cached) | 2.71 | 1.11 | 1.03 | 1.59 | 0.95 | 1.65 | 0 | Cache hit |
| `GET /api/nutritrack/runs` | 17.88 | 1.47 | 1.30 | 2.31 | 1.22 | 3.07 | 0 | Tail cache |
| `GET /api/nutritrack/foods/preview` | 2.84 | 1.86 | 1.67 | 2.69 | 1.58 | 3.02 | 0 | Resolver |
| `GET /api/nutritrack/insights` (cold single-pass) | 1.89 | 2.06 | 2.00 | 2.58 | 1.10 | 3.70 | 0 | `compute_insights_bundle` |
| `POST /api/nutritrack/chat` (fast daily summary) | 3.29 | 3.31 | 3.08 | 4.20 | 2.89 | 5.02 | 0 | Fast analytics |
| `POST /api/nutritrack/chat` (fast weekly summary) | 3.48 | 3.83 | 3.53 | 5.00 | 3.32 | 5.68 | 0 | Fast analytics |
| `POST /api/nutritrack/chat` (idempotent replay) | 26.98 | 10.47 | 8.99 | 11.76 | 8.63 | 42.57 | 0 | `DurableOperationStore` |
| `POST /api/nutritrack/water` | 7.71 | 13.15 | 12.40 | 21.60 | 6.49 | 21.67 | 0 | Atomic profile |
| `GET /api/nutritrack/state` | 16.23 | 18.41 | 18.40 | 20.37 | 15.76 | 21.20 | 0 | `StateManager` + `RunLog` |
| `POST /api/nutritrack/custom-food` | 5.52 | 19.13 | 19.07 | 20.62 | 17.38 | 20.64 | 0 | Per-user JSON |
| `DELETE /api/nutritrack/meals/<id>` | 23.87 | 23.29 | 23.47 | 26.80 | 19.98 | 28.19 | 0 | `fast_remove_meal` |
| `POST /api/nutritrack/profile` | 46.52 | 46.57 | 44.44 | 56.04 | 41.22 | 62.03 | 0 | Atomic profile+state |
| `POST /api/nutritrack/chat` (recommendations offline) | 53.68 | 47.11 | 40.70 | 67.50 | 38.28 | 67.50 | 0 | `NutriTrackOrchestrator` |
| `POST /api/nutritrack/meals` (fast quick add) | 38.02 | 72.38 | 73.21 | 87.44 | 57.76 | 89.56 | 0 | `fast_log_meal` + op store |
| `POST /api/nutritrack/chat` (fast meal log) | 67.65 | 89.12 | 88.52 | 108.43 | 64.84 | 111.87 | 0 | `route_chat_fast` + op store |

## Key Engineering Takeaways

1. **Zero-MAF Fast Analytics & Summaries:** Daily summary and weekly summary chat turns executed in **3.31 ms** and **3.83 ms** warm mean (`0` LLM calls, `0` multi-agent specialist wakeups).
2. **Single-Pass Insights & Midnight Cache:** Uncached 7-day trend + macro split + daily totals ran in **2.06 ms** mean; warm cache hits dropped to **1.11 ms** mean.
3. **Crash-Safe Persistence Cost Profile:** Write mutations (`meals`, `chat` with `request_id`, `profile`) perform cross-process `user_file_lock` + temporary-file atomic `os.replace()` across Windows NTFS (guaranteeing cross-process durability and `DurableOperationStore` persistence across restarts).
