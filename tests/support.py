"""Shared test helpers."""
import functools
import os
import shutil
import tempfile
import unittest


class IsolatedCase(unittest.TestCase):
    """Fresh temp storage + clean CA1 session for every single test."""

    def setUp(self):
        from nutritrack.ca1 import config, daily_store, tools

        self.tmp = tempfile.mkdtemp(prefix="case_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.patch_attr(daily_store, "_STORE_FILE", os.path.join(self.tmp, "daily_log.json"))
        self.patch_attr(config, "API_PROVIDER", "calorieninjas")   # online lookup OFF
        self.patch_attr(config, "CALORIE_API_KEY", "")
        self.patch_env("NUTRITRACK_CUSTOM_FOODS", os.path.join(self.tmp, "custom_foods.json"))
        tools.reset_session()

    # -- tiny patch helpers (restored automatically after each test) -----
    def patch_attr(self, obj, name, value):
        old = getattr(obj, name)
        setattr(obj, name, value)
        self.addCleanup(setattr, obj, name, old)

    def patch_env(self, key, value):
        old = os.environ.get(key)
        os.environ[key] = value
        self.addCleanup(lambda: os.environ.pop(key, None) if old is None else os.environ.__setitem__(key, old))


def known_defect(defect_id, summary):
    """Mark a test that states the CORRECT behaviour for a defect found in the
    Phase 1 audit and not yet fixed.

    * assertion fails  -> defect still present: reported as SKIPPED with the
                          defect id (the suite stays green, the backlog stays visible)
    * test passes      -> the defect has been fixed: this FAILS, telling you to
                          remove the decorator so the test becomes a normal regression test
    * any other error  -> propagates as a real error (a broken test is never hidden)
    """
    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(self):
            try:
                fn(self)
            except AssertionError:
                self.skipTest(f"KNOWN DEFECT {defect_id} still present: {summary}")
            self.fail(f"{defect_id} appears to be FIXED -- remove @known_defect from {fn.__name__}")
        return wrapper
    return decorate


# ---------------------------------------------------------------------------
# Phase 3 helpers
# ---------------------------------------------------------------------------
from datetime import datetime, timedelta  # noqa: E402
from pathlib import Path  # noqa: E402


class FakeClock:
    """A clock the test controls, so 'today', 'yesterday' and timestamps are deterministic."""

    def __init__(self, start=None):
        self.now = start or datetime(2026, 11, 13, 12, 0, 0)

    def __call__(self):
        return self.now

    def advance(self, **kwargs):
        self.now += timedelta(**kwargs)

    @property
    def today(self):
        return self.now.date().isoformat()

    def days_ago(self, n):
        return (self.now.date() - timedelta(days=n)).isoformat()


def make_entry(food="egg", quantity_str="3 pieces", calories=234.0, protein=18.0, carbs=1.8, fat=15.9, **extra):
    return {"food": food, "quantity_str": quantity_str, "calories": calories,
            "protein": protein, "carbs": carbs, "fat": fat, **extra}


class StateCase(IsolatedCase):
    """IsolatedCase + a fresh StateManager/RunLog in the temp folder and a fake clock."""

    def setUp(self):
        super().setUp()
        from nutritrack.state import RunLog, StateManager
        self.clock = FakeClock()
        self.base = Path(self.tmp) / "state"
        self.manager = StateManager(base_dir=self.base, clock=self.clock, ca1_log_path=None)
        self.run_log = RunLog(Path(self.tmp) / "logs" / "agent_runs.jsonl", clock=self.clock)
        self.state = self.manager.get("demo_user")

    def restart(self, **manager_kwargs):
        """New StateManager over the same folder = an application restart."""
        from nutritrack.state import StateManager
        kwargs = dict(base_dir=self.base, clock=self.clock, ca1_log_path=None)
        kwargs.update(manager_kwargs)
        self.manager = StateManager(**kwargs)
        self.state = self.manager.get("demo_user")
        return self.state


class ToolCase(StateCase):
    """StateCase + a ToolContext and quick access to every tool by name."""

    def setUp(self):
        super().setUp()
        from nutritrack.tools import TOOLS_BY_AGENT, new_context
        self._new_context = new_context
        self.ctx = self.new_ctx(task="test task")
        self.tools = {t.__name__: t for tools in TOOLS_BY_AGENT.values() for t in tools}

    def new_ctx(self, **kwargs):
        kwargs.setdefault("task", "test task")
        return self._new_context(self.manager, self.run_log, **kwargs)

    def call(self, _tool, *args, ctx=None, **kwargs):
        return self.tools[_tool](ctx or self.ctx, *args, **kwargs)

    def logged(self, tool=None, run_id=None, status=None):
        out = self.run_log.read(run_id=run_id)
        return [e for e in out if (tool is None or e["tool"] == tool) and (status is None or e["status"] == status)]

    def seed(self, days_ago, calories, protein=0.0, carbs=0.0, fat=0.0, food="seed meal"):
        """Put an exact, known entry on a day (bypasses the food database on purpose)."""
        import uuid
        return self.state.add_meals(
            [make_entry(food, "1 serving", calories, protein, carbs, fat)],
            day=self.clock.days_ago(days_ago), request_id=f"seed-{uuid.uuid4().hex}")

    def forget_custom_food(self, key):
        from nutritrack.ca1.food_database import FOOD_DB
        self.addCleanup(FOOD_DB.pop, key, None)
