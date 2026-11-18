# CA1 -> NutriTrack migration map (Phase 2)

Goal of this phase: new project layout, **no behaviour change**. Every CA1 feature
still works; the new multi-agent code (Phases 3-5) is built next to it.

## Where everything went

| CA1 file | New location | Changed? |
|---|---|---|
| `agent.py` | `nutritrack/ca1/agent.py` | 2 import lines |
| `tools.py` | `nutritrack/ca1/tools.py` | 4 import lines |
| `daily_store.py` | `nutritrack/ca1/daily_store.py` | 1 import + storage path |
| `food_database.py` (rebuilt in Phase 1) | `nutritrack/ca1/food_database.py` | 1 import + storage path |
| `llm_agent.py` | `nutritrack/ca1/llm_agent.py` | 4 import lines |
| `memory.py`, `config.py`, `llm_config.py`, `tool_schemas.py` | `nutritrack/ca1/` | **byte-identical** |
| `app.py` | `nutritrack/web/app.py` (+ thin top-level `app.py`) | imports, static path, `main()` wrapper |
| `frontend/index.html` | `nutritrack/web/static/index.html` | **byte-identical** |
| `README.md`, `TESTING.md`, notebook | `docs/ca1/` (archived) | unchanged |
| `daily_log.json`, `custom_foods.json` (next to the code) | `data/` | see "Your existing data" |

New in this phase: `nutritrack/paths.py`, `scripts/migrate_ca1_data.py`, `tests/`,
`pytest.ini`, `.gitignore`, empty `agents/ tools/ state/` packages (filled in Phases 3-5),
`data/`, `logs/`, `notebooks/`.

The exact edits are provable: diff any file in `nutritrack/ca1/` against the Phase 1 version -
only the import lines and storage paths above differ.

## How to run things now

| Task | Command (from the project root) |
|---|---|
| Web app | `python app.py` -> http://127.0.0.1:5000 |
| CA1 terminal chat | `python -m nutritrack.ca1.agent` (was `python agent.py`) |
| Tests | `python -m unittest discover -s tests -t .` (or `pytest`, once installed) |
| Move your CA1 data | `python scripts/migrate_ca1_data.py --from "<old CA1 code folder>"` (add `--dry-run` first) |

## Your existing data

If you ran CA1 on your own machine you have real `custom_foods.json` and `daily_log.json`
files next to the old code. Run the migration script above. It only reads the old folder,
validates each file, never overwrites existing files without `--force`, and lists any custom
food it cannot load. It was tested against files in the format this project writes; **your
real files are the first test against CA1's original format** - if it reports incompatible
entries, nothing is lost (the originals are untouched); re-add those foods once.

## Defects found in the audit (status)

Phase 2 left the CA1 defects in place on purpose, to prove "identical behaviour", and tracked them as
`@known_defect` tests. **Phase 3 fixed D1-D7** (see `docs/PHASE3_DESIGN.md`); those markers became ordinary
regression tests (`TestFixedCA1Defects`). Still open: the web-layer items - XSS via `innerHTML`, `debug=True`,
`redirect_stdout` thread-safety - which belong to Phase 7, and the archived CA1 notebook's broken cell 37.

## Decisions and assumptions

* **Package, not flat modules.** CA1's generic module names (`tools`, `memory`, `agent`) would
  collide with the new code, so CA1 lives in the `nutritrack.ca1` sub-package.
* **Runtime data moved to `data/`**, overridable with `NUTRITRACK_DATA_DIR` (and
  `NUTRITRACK_LOG_DIR`); the folder is created on demand because CA1 silently lost saves when
  it was missing.
* **Tests are `unittest.TestCase` classes.** pytest collects them natively, and they run with the
  standard library - needed because the build environment could not install pytest.
  `pytest.ini` is written but has **not** been executed.
* **Test isolation:** `tests/__init__.py` redirects all storage to a temp folder and disables
  the network food lookup and the LLM agent before anything is imported.
