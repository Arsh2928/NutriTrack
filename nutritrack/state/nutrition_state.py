"""
nutrition_state.py
------------------
The ONE shared nutrition state every agent reads and writes.

    manager = StateManager()               # one per application
    state   = manager.get("demo_user")     # one NutritionState per user
    state.snapshot()                       # the structure from the CA2 brief

Design guarantees
  * DERIVED TOTALS -- `consumed` and `remaining` are never stored. They are
    recomputed from the meal list on every read, so they cannot drift or
    disagree with the meals (CA1's incremental totals could).
  * ALL-OR-NOTHING WRITES -- every mutation runs in a transaction: validate,
    change, save atomically. If the save fails the in-memory state is rolled
    back and StateUpdateError is raised, so memory and disk never diverge and a
    failure is never silent.
  * DUPLICATE PROTECTION -- add_meals() takes an optional request_id; replaying
    the same id changes nothing (double click, LLM retry).
  * PER-USER FILES -- data/users/<user_id>.json, one file per user.
  * HONEST HISTORY -- a day exists only if meals were saved for it. A day with
    no records is "no data", never "0 kcal" (CA1's rollover archived empty days
    as zeros).
  * EPHEMERAL RUN DATA -- agent_results, workflow_status and the recommendation
    candidate pool live in memory only; a restart starts from "idle".
"""
import copy
import json
import math
import re
import threading
import uuid
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path

from ..paths import data_dir
from .io import atomic_write_json, user_file_lock

SCHEMA_VERSION = 1
DEMO_USER_ID = "demo_user"
# Clearly identified DEMONSTRATION defaults -- not advice. Only the demo user gets
# them; every other new user starts with no targets and is asked to set them.
DEMO_DEFAULT_TARGETS = {"calories": 2300.0, "protein": 150.0}
# Sanity limits that catch typos (e.g. 23000 instead of 2300). Not medical advice.
TARGET_LIMITS = {"calories": (500.0, 10000.0), "protein": (10.0, 500.0)}
WORKFLOW_STATUSES = ("idle", "planning", "delegating", "validating", "compiling", "completed", "failed")
DIETS = ("any", "vegetarian", "eggetarian")
MEAL_TYPES = ("breakfast", "brunch", "lunch", "dinner", "snack")
USER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MAX_REQUEST_IDS = 500
MAX_ENTRIES_PER_REQUEST = 30
MAX_PREFERENCE_ITEMS = 25
MAX_BACKDATE_DAYS = 400
NUTRIENTS = ("calories", "protein", "carbs", "fat")


class StateError(ValueError):
    """The request itself is invalid (bad value, unknown id, ...). Nothing was changed."""


class StateUpdateError(RuntimeError):
    """The state could not be stored. The in-memory state was rolled back."""


# ---------------------------------------------------------------------------
# Small validation helpers
# ---------------------------------------------------------------------------
def resolve_date(value, today):
    """None/'today'/'yesterday'/ISO date -> ISO string. Future and ancient dates are rejected."""
    if value is None or (isinstance(value, str) and value.strip().lower() in ("", "today")):
        return today.isoformat()
    if isinstance(value, str) and value.strip().lower() == "yesterday":
        return (today - timedelta(days=1)).isoformat()
    try:
        d = value if isinstance(value, date) and not isinstance(value, datetime) else date.fromisoformat(str(value).strip())
    except ValueError:
        raise StateError(f"'{value}' is not a valid date. Use YYYY-MM-DD, 'today' or 'yesterday'.")
    if d > today:
        raise StateError(f"{d.isoformat()} is in the future; meals can only be logged for today or earlier.")
    if (today - d).days > MAX_BACKDATE_DAYS:
        raise StateError(f"{d.isoformat()} is more than {MAX_BACKDATE_DAYS} days ago.")
    return d.isoformat()


def _number(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise StateError(f"{name} must be a number.")
    if not low <= value <= high:
        raise StateError(f"{name} must be between {low:g} and {high:g} (got {value:g}).")
    return float(value)


def _validate_entry(entry):
    if not isinstance(entry, dict):
        raise StateError("Each meal entry must be an object.")
    food = str(entry.get("food", "")).strip()
    quantity = str(entry.get("quantity_str", "")).strip()
    if not food or len(food) > 100:
        raise StateError("A meal entry needs a food name of 1-100 characters.")
    if not quantity or len(quantity) > 60:
        raise StateError("A meal entry needs a quantity description of 1-60 characters.")
    out = {"food": food, "quantity_str": quantity}
    for key, high in (("calories", 10000), ("protein", 1000), ("carbs", 2000), ("fat", 1000)):
        out[key] = round(_number(entry.get(key), key, 0, high), 1)
    fiber_val = 0.0 if entry.get("fiber") is None else entry.get("fiber")
    out["fiber"] = round(_number(fiber_val, "fiber", 0, 500), 1)
    for key, limit in (("serving_size", 120), ("provenance", 40), ("external_source", 80)):
        if entry.get(key) is not None:
            out[key] = str(entry[key])[:limit]
    out["external"] = bool(entry.get("external", False))
    out["estimated"] = bool(entry.get("estimated", True))
    return out


def _sum(meals):
    return {k: round(math.fsum(m[k] for m in meals), 1) for k in NUTRIENTS}


def _sum_fiber(meals):
    return round(math.fsum(float(m.get("fiber", 0.0) or 0.0) for m in meals), 1)


def _clean_names(values, label):
    if values is None:
        return None
    if isinstance(values, str) or not isinstance(values, (list, tuple, set)):
        raise StateError(f"'{label}' must be a list of food names.")
    cleaned = []
    for v in values:
        name = str(v).strip().lower()
        if name and len(name) <= 60 and name not in cleaned:
            cleaned.append(name)
    if len(cleaned) > MAX_PREFERENCE_ITEMS:
        raise StateError(f"'{label}' can have at most {MAX_PREFERENCE_ITEMS} items.")
    return cleaned


# ---------------------------------------------------------------------------
class NutritionState:
    """One user's nutrition state. Create through StateManager.get()."""

    def __init__(self, user_id, path, clock, doc, load_warnings=None):
        self.user_id = user_id
        self._path = Path(path)
        self._lock_path = self._path.with_name(f"{self._path.name}.lock")
        self._clock = clock
        self._doc = doc
        self._lock = threading.RLock()
        self._disk_sig = self._stat_sig()
        self.load_warnings = list(load_warnings or [])
        self._runtime = {"agent_results": {}, "workflow_status": "idle",
                         "workflow_detail": None, "workflow_history": [], "open_questions": []}
        self.pool = {}                              # recommendation candidate pool (ephemeral)

    def _stat_sig(self):
        try:
            st = self._path.stat()
            return (st.st_mtime_ns, st.st_size)
        except OSError:
            return None

    def _reload_locked(self, *, force: bool = False):
        """Reload state JSON from disk under self._lock (and cross-process file lock)."""
        sig = self._stat_sig()
        if sig is None:
            return False
        if not force and sig == self._disk_sig:
            return False
        with user_file_lock(self._lock_path):
            sig = self._stat_sig()
            if sig is None:
                return False
            if not force and sig == self._disk_sig:
                return False
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                if force:
                    raise
                return False
            if not isinstance(raw, dict):
                if force:
                    raise ValueError("state file must contain a JSON object")
                return False
            for key in (
                "schema_version",
                "user_id",
                "created_at",
                "targets",
                "targets_source",
                "preferences",
                "days",
                "request_ids",
                "recommendations",
                "recommendations_context",
                "meta",
            ):
                if key in raw:
                    self._doc[key] = raw[key]
            # Ensure core nested containers remain valid dicts/lists even if partially omitted
            self._doc.setdefault("targets", {"calories": None, "protein": None})
            self._doc.setdefault("targets_source", {"calories": None, "protein": None})
            self._doc.setdefault("preferences", {"diet": "any", "avoid": [], "prefer": []})
            self._doc.setdefault("days", {})
            self._doc.setdefault("request_ids", {})
            self._doc.setdefault("recommendations", [])
            self._doc.setdefault("recommendations_context", None)
            self._doc.setdefault("meta", {"ca1_import": None})
            self._disk_sig = sig
            return True

    def reload_if_changed(self):
        """Reload state JSON from disk if another process modified the file externally."""
        with self._lock:
            return self._reload_locked(force=False)

    # -- time ------------------------------------------------------------
    def _now(self):
        return self._clock()

    def today(self):
        return self._now().date().isoformat()

    def resolve_date(self, value=None):
        return resolve_date(value, self._now().date())

    # -- transactions ----------------------------------------------------
    def _persist(self):
        atomic_write_json(self._path, self._doc)
        self._disk_sig = self._stat_sig()

    @contextmanager
    def _txn(self):
        """Mutate self._doc inside the block; lock cross-process, reload latest disk state, save atomically, roll back on error."""
        class _TxnState:
            dirty = True

        txn = _TxnState()
        with self._lock:
            with user_file_lock(self._lock_path):
                try:
                    self._reload_locked(force=True)
                except StateError:
                    raise
                except Exception as exc:
                    raise StateUpdateError(f"Could not reload the nutrition state before transaction: {exc}") from exc
                backup = copy.deepcopy(self._doc)
                try:
                    yield txn
                    if txn.dirty:
                        self._persist()
                except StateError:
                    self._doc = backup
                    raise
                except Exception as exc:
                    self._doc = backup
                    raise StateUpdateError(f"Could not update the nutrition state: {exc}") from exc

    # -- reading ---------------------------------------------------------
    def get_day_meals(self, day=None):
        d = self.resolve_date(day)
        with self._lock:
            self._reload_locked(force=False)
            return copy.deepcopy(self._doc["days"].get(d, {}).get("meals", []))

    def targets_for(self, day=None):
        """Targets in effect on a day: the day's saved snapshot where it has one,
        otherwise the current target."""
        d = self.resolve_date(day)
        with self._lock:
            self._reload_locked(force=False)
            current = self._doc["targets"]
            snap = self._doc["days"].get(d, {}).get("targets", {})
            return {k: (snap.get(k) if snap.get(k) is not None else current.get(k)) for k in ("calories", "protein")}

    def targets_source(self):
        with self._lock:
            self._reload_locked(force=False)
            return dict(self._doc["targets_source"])

    def day_totals(self, day=None):
        """Totals recomputed from the meals; includes fiber, meal_count, and has_data."""
        meals = self.get_day_meals(day)
        out = _sum(meals)
        out["fiber"] = _sum_fiber(meals)
        out["meal_count"] = len(meals)
        out["has_data"] = bool(meals)
        return out

    def recorded_dates(self):
        """Dates that have at least one saved meal, oldest first."""
        with self._lock:
            self._reload_locked(force=False)
            return sorted(d for d, rec in self._doc["days"].items() if rec.get("meals"))

    def remaining(self, day=None):
        t, c = self.targets_for(day), self.day_totals(day)
        return {k: (None if t[k] is None else round(t[k] - c[k], 1)) for k in ("calories", "protein")}

    def preferences(self):
        with self._lock:
            self._reload_locked(force=False)
            return copy.deepcopy(self._doc["preferences"])

    def history(self, end_day=None, days=7):
        """Saved records for the `days` days ending at end_day (inclusive). Days
        without records are simply absent -- they are not zero-filled."""
        end = date.fromisoformat(self.resolve_date(end_day))
        rows = []
        with self._lock:
            self._reload_locked(force=False)
            for offset in range(days - 1, -1, -1):
                d = (end - timedelta(days=offset)).isoformat()
                meals = self.get_day_meals(d) if d <= self.today() else []
                if meals:
                    t = self.targets_for(d)
                    rows.append({"date": d, **_sum(meals), "fiber": _sum_fiber(meals), "meal_count": len(meals),
                                 "calorie_target": t["calories"], "protein_target": t["protein"]})
        return rows

    def snapshot(self, day=None, history_days=7):
        """The shared-state structure from the CA2 brief, for one day (default today)."""
        d = self.resolve_date(day)
        with self._lock:
            self._reload_locked(force=False)
            meals = self.get_day_meals(d)
            consumed = _sum(meals)
            fiber = _sum_fiber(meals)
            targets = self.targets_for(d)
            remaining = self.remaining(d)
            recs = copy.deepcopy(self._doc["recommendations"])
            ctx = self._doc.get("recommendations_context")
            stale = bool(recs) and (ctx is None or ctx.get("date") != d
                                    or ctx.get("remaining") != remaining)
            return {
                "user_id": self.user_id,
                "date": d,
                "targets": targets,
                "targets_source": self.targets_source(),
                "consumed": consumed,
                "fiber": fiber,
                "remaining": remaining,
                "meals": meals,
                "weekly_history": self.history(d, history_days),
                "recommendations": recs,
                "recommendations_stale": stale,
                "preferences": self.preferences(),
                "agent_results": copy.deepcopy(self._runtime["agent_results"]),
                "workflow_status": self._runtime["workflow_status"],
                "workflow_detail": self._runtime["workflow_detail"],
                "meta": {"schema_version": SCHEMA_VERSION, "load_warnings": list(self.load_warnings),
                         "ca1_import": copy.deepcopy(self._doc.get("meta", {}).get("ca1_import"))},
            }

    # -- targets & preferences ------------------------------------------
    def set_target(self, kind, value, source="user"):
        if kind not in TARGET_LIMITS:
            raise StateError(f"Unknown target '{kind}'. Use 'calories' or 'protein'.")
        low, high = TARGET_LIMITS[kind]
        value = _number(value, f"The {kind} target", low, high)
        with self._txn():
            old = self._doc["targets"][kind]
            self._doc["targets"][kind] = value
            self._doc["targets_source"][kind] = source
            today_rec = self._doc["days"].get(self.today())
            if today_rec is not None:
                today_rec.setdefault("targets", {})[kind] = value     # past days keep their own targets
        return {"kind": kind, "old": old, "new": value, "source": source}

    def set_preferences(self, diet=None, avoid=None, prefer=None):
        if diet is not None and diet not in DIETS:
            raise StateError(f"diet must be one of {', '.join(DIETS)}.")
        avoid, prefer = _clean_names(avoid, "avoid"), _clean_names(prefer, "prefer")
        with self._txn():
            prefs = self._doc["preferences"]
            if diet is not None:
                prefs["diet"] = diet
            if avoid is not None:
                prefs["avoid"] = avoid
            if prefer is not None:
                prefs["prefer"] = prefer
        return self.preferences()

    # -- meals -----------------------------------------------------------
    def add_meals(self, entries, request_id=None, meal_type=None, day=None, source="agent"):
        """Add food entries. Returns {"duplicate": bool, "added": [...], "meal_ids": [...], "date", "batch_id"}."""
        if not isinstance(entries, list) or not entries:
            raise StateError("At least one meal entry is required.")
        if len(entries) > MAX_ENTRIES_PER_REQUEST:
            raise StateError(f"At most {MAX_ENTRIES_PER_REQUEST} entries can be added at once.")
        if meal_type is not None and meal_type not in MEAL_TYPES:
            raise StateError(f"meal_type must be one of {', '.join(MEAL_TYPES)}.")
        if request_id is not None and (not isinstance(request_id, str) or not 1 <= len(request_id) <= 200):
            raise StateError("request_id must be a string of 1-200 characters.")
        d = self.resolve_date(day)
        validated = [_validate_entry(e) for e in entries]

        with self._txn() as txn:
            known = self._doc.setdefault("request_ids", {})
            if request_id is not None and request_id in known:
                prior = known[request_id]
                txn.dirty = False
                return {"duplicate": True, "added": [], "meal_ids": list(prior["meal_ids"]),
                        "date": prior["date"], "batch_id": prior.get("batch_id")}
            now = self._now().isoformat(timespec="seconds")
            batch_id = "b-" + uuid.uuid4().hex[:8]
            created = [{"meal_id": "m-" + uuid.uuid4().hex[:8], "batch_id": batch_id, "meal_type": meal_type,
                        "logged_at": now, "date": d, "source": source, "request_id": request_id, **e}
                       for e in validated]
            days = self._doc.setdefault("days", {})
            rec = days.setdefault(d, {"targets": copy.deepcopy(self._doc["targets"]), "meals": []})
            rec["meals"].extend(copy.deepcopy(created))
            if request_id is not None:
                known[request_id] = {"meal_ids": [m["meal_id"] for m in created], "date": d,
                                     "batch_id": batch_id, "at": now}
                while len(known) > MAX_REQUEST_IDS:
                    known.pop(next(iter(known)))
        return {"duplicate": False, "added": created, "meal_ids": [m["meal_id"] for m in created],
                "date": d, "batch_id": batch_id}

    def remove_meal(self, meal_id=None, batch_id=None, food_name=None, day=None, remove_last=False):
        """Remove by meal_id, by batch_id, the most recent match of food_name on a day, or the last logged meal."""
        chosen = [x for x in (meal_id, batch_id, food_name) if x]
        if remove_last:
            if chosen:
                raise StateError("Give either remove_last=True or one of meal_id, batch_id or food_name.")
        elif len(chosen) != 1:
            raise StateError("Give exactly one of meal_id, batch_id or food_name.")

        target_day = self.resolve_date(day) if (food_name or (remove_last and day is not None)) else None

        with self._txn() as txn:
            days = self._doc.setdefault("days", {})
            targets = []                                           # (date, entry)
            extra = 0
            if meal_id:
                targets = [(d, m) for d, rec in days.items() for m in rec.get("meals", []) if m.get("meal_id") == meal_id]
            elif batch_id:
                targets = [(d, m) for d, rec in days.items() for m in rec.get("meals", []) if m.get("batch_id") == batch_id]
            elif remove_last:
                if target_day is not None:
                    meals = days.get(target_day, {}).get("meals", [])
                    if meals:
                        targets = [(target_day, meals[-1])]
                        extra = len(meals) - 1
                else:
                    today_key = self.today()
                    if days.get(today_key, {}).get("meals"):
                        meals = days[today_key]["meals"]
                        targets = [(today_key, meals[-1])]
                        extra = len(meals) - 1
                    else:
                        recorded = sorted(d for d, rec in days.items() if rec.get("meals"))
                        if recorded:
                            latest_d = recorded[-1]
                            meals = days[latest_d]["meals"]
                            targets = [(latest_d, meals[-1])]
                            extra = len(meals) - 1
            else:
                needle = food_name.strip().lower()
                def _sing(w: str) -> str:
                    if len(w) > 4 and w.endswith("ies"):
                        return w[:-3] + "y"
                    if len(w) > 4 and w.endswith("oes"):
                        return w[:-2]
                    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
                        return w[:-1]
                    return w
                needle_words = [_sing(w) for w in re.findall(r"[a-z0-9]+", needle)]
                needle_sing = " ".join(needle_words)
                day_meals = days.get(target_day, {}).get("meals", [])
                matches = [
                    m for m in day_meals
                    if needle in m["food"].lower()
                    or (needle_sing and needle_sing in " ".join(_sing(w) for w in re.findall(r"[a-z0-9]+", m["food"].lower())))
                ]
                if not matches and needle_words:
                    needle_set = set(needle_words)
                    matches = [
                        m for m in day_meals
                        if (m_words := [_sing(w) for w in re.findall(r"[a-z0-9]+", m["food"].lower())])
                        and set(m_words) <= needle_set
                    ]
                if matches:
                    targets = [(target_day, matches[-1])]
                    extra = len(matches) - 1
            if not targets:
                txn.dirty = False
                return {"removed": [], "remaining_matches": 0, "dates": []}
            removed_copy = copy.deepcopy([m for _, m in targets])
            affected_dates = sorted({d for d, _ in targets})
            ids = {m["meal_id"] for _, m in targets}
            for d in affected_dates:
                rec = days[d]
                rec["meals"] = [m for m in rec["meals"] if m["meal_id"] not in ids]
                if not rec["meals"]:
                    del days[d]                                # no meals => no record (never a 0-kcal day)
            alive = {m["meal_id"] for rec in days.values() for m in rec.get("meals", [])}
            req_ids = self._doc.setdefault("request_ids", {})
            for rid in [r for r, v in req_ids.items() if not set(v.get("meal_ids", [])) & alive]:
                del req_ids[rid]                               # a deleted meal can be logged again
        return {"removed": removed_copy, "remaining_matches": extra, "dates": affected_dates}

    def has_meal(self, meal_id):
        """Return True if a meal entry with `meal_id` exists on any recorded date."""
        if not meal_id:
            return False
        with self._lock:
            self._reload_locked(force=False)
            return any(m.get("meal_id") == meal_id for rec in self._doc["days"].values() for m in rec["meals"])

    # -- recommendations (persisted) -------------------------------------
    def set_recommendations(self, recommendations, day=None):
        if not isinstance(recommendations, list) or len(recommendations) > 10:
            raise StateError("recommendations must be a list of at most 10 items.")
        d = self.resolve_date(day)
        with self._txn():
            context = {"date": d, "remaining": self.remaining(d), "generated_at": self._now().isoformat(timespec="seconds")}
            self._doc["recommendations"] = copy.deepcopy(recommendations)
            self._doc["recommendations_context"] = context

    # -- ephemeral run data ----------------------------------------------
    def update_agent_result(self, agent_key, *, bump_tool_calls=False, **fields):
        with self._lock:
            rec = self._runtime["agent_results"].setdefault(agent_key, {"tool_calls": 0})
            if bump_tool_calls:
                rec["tool_calls"] += 1
            rec.update(fields)
            rec["updated_at"] = self._now().isoformat(timespec="seconds")
            return copy.deepcopy(rec)

    def add_open_question(self, run_id, agent, tool, code, message, question=None):
        """Remember that a tool needs something from the USER (an unknown food, a missing target...),
        even when the tool itself succeeded in part. Cleared by reset_run_data()."""
        with self._lock:
            items = self._runtime["open_questions"]
            items.append({"run_id": run_id, "agent": agent, "tool": tool, "code": code,
                          "message": str(message)[:300], "question": None if question is None else str(question)[:300]})
            del items[:-100]

    def get_open_questions(self, run_id=None):
        with self._lock:
            return copy.deepcopy([q for q in self._runtime["open_questions"] if run_id is None or q["run_id"] == run_id])

    def get_agent_results(self):
        with self._lock:
            return copy.deepcopy(self._runtime["agent_results"])

    def set_workflow_status(self, status, detail=None):
        if status not in WORKFLOW_STATUSES:
            raise StateError(f"workflow status must be one of {', '.join(WORKFLOW_STATUSES)}.")
        with self._lock:
            previous = self._runtime["workflow_status"]
            self._runtime["workflow_status"] = status
            self._runtime["workflow_detail"] = None if detail is None else str(detail)[:200]
            hist = self._runtime["workflow_history"]
            hist.append({"at": self._now().isoformat(timespec="seconds"), "from": previous, "to": status,
                         "detail": self._runtime["workflow_detail"]})
            del hist[:-50]
            return {"previous": previous, "current": status}

    def get_workflow(self):
        with self._lock:
            return {"status": self._runtime["workflow_status"], "detail": self._runtime["workflow_detail"],
                    "history": copy.deepcopy(self._runtime["workflow_history"])}

    def reset_run_data(self):
        """Start of a new run: forget the previous run's agent results, candidate pool, and stale workflow status."""
        with self._lock:
            self._runtime["agent_results"] = {}
            self._runtime["open_questions"] = []
            self._runtime["workflow_status"] = "idle"
            self._runtime["workflow_detail"] = None
            self.pool = {}


# ---------------------------------------------------------------------------
class StateManager:
    """Owns one NutritionState per user and the on-disk layout (data/users/<id>.json)."""

    def __init__(self, base_dir=None, clock=None, ca1_log_path="auto"):
        self._base = Path(base_dir) if base_dir else None
        self._clock = clock or datetime.now
        self._ca1_log_path = ca1_log_path        # "auto" | None (disabled) | explicit path
        self._states = {}
        self._lock = threading.Lock()

    def _users_dir(self):
        return (self._base or data_dir()) / "users"

    def path_for(self, user_id):
        if not isinstance(user_id, str) or not USER_ID_RE.match(user_id):
            raise StateError("user_id must be 1-64 characters: letters, digits, '_' or '-'.")
        return self._users_dir() / f"{user_id}.json"

    def list_users(self):
        d = self._users_dir()
        return sorted(p.stem for p in d.glob("*.json")) if d.exists() else []

    def get(self, user_id=DEMO_USER_ID):
        path = self.path_for(user_id)
        with self._lock:
            if user_id not in self._states:
                self._states[user_id] = self._load_or_create(user_id, path)
            else:
                self._states[user_id].reload_if_changed()
            return self._states[user_id]

    # -- creation / loading ------------------------------------------------
    def _new_doc(self, user_id):
        doc = {
            "schema_version": SCHEMA_VERSION, "user_id": user_id,
            "created_at": self._clock().isoformat(timespec="seconds"),
            "targets": {"calories": None, "protein": None},
            "targets_source": {"calories": None, "protein": None},
            "preferences": {"diet": "any", "avoid": [], "prefer": []},
            "days": {}, "request_ids": {}, "recommendations": [], "recommendations_context": None,
            "meta": {"ca1_import": None},
        }
        if user_id == DEMO_USER_ID:
            for kind, value in DEMO_DEFAULT_TARGETS.items():
                doc["targets"][kind], doc["targets_source"][kind] = value, "demo_default"
        return doc

    def _normalise(self, doc, user_id, warnings):
        if not isinstance(doc, dict):
            raise ValueError("state file must contain a JSON object")
        version = doc.get("schema_version", 1)
        if not isinstance(version, int) or version > SCHEMA_VERSION:
            raise StateUpdateError(f"{user_id}.json uses schema version {version!r}, newer than this "
                                   f"program understands ({SCHEMA_VERSION}). Refusing to load it.")
        fresh = self._new_doc(user_id)
        for key, default in fresh.items():
            doc.setdefault(key, copy.deepcopy(default))
        for key in ("calories", "protein"):
            doc["targets"].setdefault(key, None)
            doc["targets_source"].setdefault(key, None)
        doc.setdefault("meta", {}).setdefault("ca1_import", None)
        if doc.get("user_id") != user_id:
            warnings.append(f"state file said user_id {doc.get('user_id')!r}; using {user_id!r}")
            doc["user_id"] = user_id
        return doc

    def _load_or_create(self, user_id, path):
        warnings = []
        doc = None
        lock_path = path.with_name(f"{path.name}.lock")
        with user_file_lock(lock_path):
            if path.exists():
                try:
                    doc = self._normalise(json.loads(path.read_text(encoding="utf-8")), user_id, warnings)
                except StateUpdateError:
                    raise
                except Exception as exc:
                    backup = path.with_name(f"{path.name}.corrupt-{self._clock():%Y%m%d%H%M%S}")
                    try:
                        path.replace(backup)
                        warnings.append(f"The saved state for '{user_id}' was unreadable ({exc}); "
                                        f"kept as {backup.name} and started fresh.")
                    except OSError:
                        warnings.append(f"The saved state for '{user_id}' was unreadable ({exc}) and could not be backed up.")
                    doc = None
            if doc is None:
                doc = self._new_doc(user_id)
                if user_id == DEMO_USER_ID:
                    self._maybe_import_ca1(doc, warnings)
                try:
                    atomic_write_json(path, doc)
                except Exception as exc:
                    raise StateUpdateError(f"Could not create the state file {path}: {exc}") from exc
        # a restart always begins "idle" (workflow status is not persisted)
        return NutritionState(user_id, path, self._clock, doc, warnings)

    def _maybe_import_ca1(self, doc, warnings):
        """One-time import of the CA1 daily log (data/daily_log.json) into a NEW demo-user state."""
        from .ca1_import import import_ca1_daily_log
        source = data_dir() / "daily_log.json" if self._ca1_log_path == "auto" else self._ca1_log_path
        if not source or not Path(source).exists():
            return
        try:
            store = json.loads(Path(source).read_text(encoding="utf-8"))
            report = import_ca1_daily_log(doc, store, self._clock().date().isoformat())
            report.update(source=str(source), imported_at=self._clock().isoformat(timespec="seconds"))
            doc["meta"]["ca1_import"] = report
        except Exception as exc:
            warnings.append(f"Could not import the CA1 daily log {source}: {exc}. It was left untouched.")
