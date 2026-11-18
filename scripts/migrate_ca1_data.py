"""
migrate_ca1_data.py
-------------------
Copy your existing CA1 runtime data into NutriTrack's data/ folder.

CA1 kept two JSON files next to its source code:
    custom_foods.json   foods you added yourself (e.g. "mom's rajma")
    daily_log.json      today's meals, budget and the history of past days

NutriTrack keeps them in <project>/data/ instead. Usage:

    python scripts/migrate_ca1_data.py --from "path/to/AGENTIC AI CA1/code"
    python scripts/migrate_ca1_data.py --from "path/to/old/code" --dry-run

Safety rules
  * The source folder is only READ -- nothing there is modified or deleted.
  * A file is validated before copying; an unreadable or wrongly-shaped file
    is reported and NOT copied.
  * An existing destination file is never overwritten unless --force is given.

Note: this handles the Phase 2 layout, where the file formats are unchanged.
The custom-food check below tests the entries against what this project's
food_database.py can load; if your original file used a different layout,
the incompatible entries are listed so you can re-add them (nothing is lost
-- your original file is left untouched).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nutritrack.paths import data_dir  # noqa: E402

CUSTOM_REQUIRED = ("unit", "base_amount", "calories_per_unit", "protein", "carbs", "fat")
DAILY_REQUIRED = ("current_date", "budget", "meals", "history")


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _check_custom(data) -> tuple[bool, str, list[str]]:
    if not isinstance(data, dict):
        return False, "expected a JSON object of foods", []
    bad = []
    for key, entry in data.items():
        if not isinstance(entry, dict):
            bad.append(f"{key!r}: not an object")
            continue
        missing = [k for k in CUSTOM_REQUIRED if k not in entry]
        if missing:
            bad.append(f"{key!r}: missing {', '.join(missing)}")
            continue
        try:
            for k in CUSTOM_REQUIRED[1:]:
                float(entry[k])
        except (TypeError, ValueError):
            bad.append(f"{key!r}: non-numeric nutrition value")
    ok_count = len(data) - len(bad)
    return True, f"{ok_count} of {len(data)} custom food(s) compatible", bad


def _check_daily(data) -> tuple[bool, str, list[str]]:
    if not isinstance(data, dict):
        return False, "expected a JSON object", []
    missing = [k for k in DAILY_REQUIRED if k not in data]
    if missing:
        return False, f"missing key(s): {', '.join(missing)}", []
    hist = data["history"]
    if not isinstance(hist, dict) or not isinstance(data["meals"], list):
        return False, "'history' must be an object and 'meals' a list", []
    return True, (f"{len(data['meals'])} meal(s) today ({data['current_date']}), "
                  f"{len(hist)} past day(s) in history"), []


CHECKS = {"custom_foods.json": _check_custom, "daily_log.json": _check_daily}


def migrate(src_dir: Path, dry_run: bool = False, force: bool = False) -> int:
    """Returns 0 if everything found was handled, 1 if any file was invalid."""
    dest_dir = data_dir()
    exit_code = 0
    print(f"Source : {src_dir}\nTarget : {dest_dir}{'   [dry run]' if dry_run else ''}\n")
    if not src_dir.is_dir():
        print(f"ERROR: source folder does not exist: {src_dir}")
        return 1

    for name, check in CHECKS.items():
        src, dest = src_dir / name, dest_dir / name
        if not src.exists():
            print(f"- {name}: not found in source (skipped)")
            continue
        try:
            data = _load_json(src)
        except Exception as exc:
            print(f"- {name}: UNREADABLE ({exc}) -- NOT copied")
            exit_code = 1
            continue
        ok, summary, problems = check(data)
        if not ok:
            print(f"- {name}: INVALID ({summary}) -- NOT copied")
            exit_code = 1
            continue
        print(f"- {name}: valid -- {summary}")
        for p in problems:
            print(f"    incompatible entry (will not load until re-added): {p}")
        if dest.exists() and not force:
            print("    destination already exists -- left unchanged (use --force to overwrite)")
            continue
        if dry_run:
            print(f"    would copy -> {dest}")
        else:
            shutil.copy2(src, dest)
            print(f"    copied -> {dest}")
    return exit_code


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Copy CA1 runtime data into NutriTrack's data/ folder.")
    parser.add_argument("--from", dest="src", required=True, help="folder containing the old CA1 JSON files")
    parser.add_argument("--dry-run", action="store_true", help="show what would happen; copy nothing")
    parser.add_argument("--force", action="store_true", help="overwrite files that already exist in data/")
    args = parser.parse_args(argv)
    return migrate(Path(args.src).expanduser(), dry_run=args.dry_run, force=args.force)


if __name__ == "__main__":
    sys.exit(main())
