"""
paths.py
--------
Single source of truth for where NutriTrack keeps runtime files.

CA1 wrote `daily_log.json` and `custom_foods.json` next to the source code.
Runtime data now lives in `<project>/data/` and execution logs in
`<project>/logs/`, so source and data are separate and easy to back up.

Both locations can be redirected with environment variables (used by the
automated tests so they never touch real data):

    NUTRITRACK_DATA_DIR   default: <project>/data
    NUTRITRACK_LOG_DIR    default: <project>/logs

The directories are created on demand. (In CA1, a missing folder made the
daily-log save fail *silently*; creating it here removes that trap.)
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve(env_var: str, default_name: str) -> Path:
    override = os.environ.get(env_var, "").strip()
    path = Path(override).expanduser() if override else PROJECT_ROOT / default_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir() -> Path:
    """Folder for persistent user data (daily log, custom foods, ...)."""
    return _resolve("NUTRITRACK_DATA_DIR", "data")


def logs_dir() -> Path:
    """Folder for the append-only execution log (agent_runs.jsonl)."""
    return _resolve("NUTRITRACK_LOG_DIR", "logs")
