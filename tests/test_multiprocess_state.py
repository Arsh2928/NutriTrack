"""Cross-process StateManager, RunLog, DurableOperationStore, and LockRegistry stress tests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nutritrack.state import DurableOperationStore, RunLog, StateError, StateManager, StateUpdateError, compute_payload_hash
from nutritrack.state.io import atomic_write_json, get_lock_registry


def _mp_worker_add_same_request(base_dir: str, user_id: str, food: str, req_id: str, count: int) -> list[dict]:
    sm = StateManager(base_dir=base_dir, ca1_log_path=None)
    results = []
    for _ in range(count):
        st = sm.get(user_id)
        res = st.add_meals(
            [{"food": food, "quantity_str": "100 g", "calories": 150.0, "protein": 10.0, "carbs": 12.0, "fat": 5.0, "fiber": 2.0}],
            request_id=req_id,
        )
        results.append(res)
    return results


def _mp_worker_add_unique(base_dir: str, user_id: str, proc_idx: int, n_meals: int) -> list[str]:
    sm = StateManager(base_dir=base_dir, ca1_log_path=None)
    created_ids: list[str] = []
    for i in range(n_meals):
        st = sm.get(user_id)
        res = st.add_meals(
            [{"food": f"item_{proc_idx}_{i}", "quantity_str": "1 piece", "calories": 100.0, "protein": 5.0, "carbs": 10.0, "fat": 2.0}],
            request_id=f"mp-{proc_idx}-{i}",
        )
        # Replay once immediately in the same process to test cross-process idempotency
        res_dup = st.add_meals(
            [{"food": f"item_{proc_idx}_{i}", "quantity_str": "1 piece", "calories": 100.0, "protein": 5.0, "carbs": 10.0, "fat": 2.0}],
            request_id=f"mp-{proc_idx}-{i}",
        )
        assert res_dup.get("duplicate") is True
        if res.get("added"):
            created_ids.append(res["added"][0]["meal_id"])
    return created_ids


def _mp_worker_run_log(log_path: str, proc_idx: int, entries: int) -> None:
    rl = RunLog(path=log_path, max_bytes=2048)
    for i in range(entries):
        rl.append(
            run_id=f"p{proc_idx}-{i}",
            user_id="stress_u",
            agent="Stress Worker",
            task=f"step {i}",
            tool="stress_tool",
            status="success",
            output_summary=f"payload-{proc_idx}-{i}-" + ("x" * 64),
        )


class MultiprocessAndConcurrencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_bug_a_two_statemanagers_persist_idempotency_across_mutations_and_restarts(self) -> None:
        """Stale StateManager A must never drop idempotency keys or overwrite B's committed meals."""
        sm1 = StateManager(base_dir=self.base, ca1_log_path=None)
        sm2 = StateManager(base_dir=self.base, ca1_log_path=None)

        st1 = sm1.get("u1")
        st2 = sm2.get("u1")

        # St1 logs banana with request_id='req-ban-1'
        r1 = st1.add_meals(
            [{"food": "banana", "quantity_str": "1 piece", "calories": 105.0, "protein": 1.3, "carbs": 27.0, "fat": 0.4}],
            request_id="req-ban-1",
        )
        self.assertEqual(len(r1["added"]), 1)

        # Independently, st2 logs egg (leaving st1's previous in-memory doc stale)
        r2 = st2.add_meals(
            [{"food": "boiled egg", "quantity_str": "2 pieces", "calories": 156.0, "protein": 12.6, "carbs": 1.2, "fat": 10.6}],
            request_id="req-egg-1",
        )
        self.assertEqual(len(r2["added"]), 1)

        # Replay 'req-egg-1' via st1 (which didn't originate it!) -> must deduplicate via reloaded disk state!
        dup_from_other = st1.add_meals(
            [{"food": "boiled egg", "quantity_str": "2 pieces", "calories": 156.0, "protein": 12.6, "carbs": 1.2, "fat": 10.6}],
            request_id="req-egg-1",
        )
        self.assertTrue(dup_from_other.get("duplicate"))
        self.assertEqual(dup_from_other["added"], [])

        # Also replay via a brand-new StateManager simulating process restart
        sm3 = StateManager(base_dir=self.base, ca1_log_path=None)
        st3 = sm3.get("u1")
        dup_after_restart = st3.add_meals(
            [{"food": "banana", "quantity_str": "1 piece", "calories": 105.0, "protein": 1.3, "carbs": 27.0, "fat": 0.4}],
            request_id="req-ban-1",
        )
        self.assertTrue(dup_after_restart.get("duplicate"))
        snap = st3.snapshot()
        self.assertEqual(len(snap["meals"]), 2)
        self.assertAlmostEqual(snap["consumed"]["calories"], 261.0, places=1)

    def test_bug_b_remove_meal_persists_even_from_stale_statemanager_and_reports_persist_failure(self) -> None:
        sm1 = StateManager(base_dir=self.base, ca1_log_path=None)
        sm2 = StateManager(base_dir=self.base, ca1_log_path=None)

        st1 = sm1.get("u_del")
        st2 = sm2.get("u_del")

        st1.add_meals([{"food": "egg", "quantity_str": "1 piece", "calories": 78.0, "protein": 6.0, "carbs": 0.6, "fat": 5.3}])
        add_b = st2.add_meals([{"food": "banana", "quantity_str": "1 piece", "calories": 105.0, "protein": 1.3, "carbs": 27.0, "fat": 0.4}])
        banana_id = add_b["added"][0]["meal_id"]

        # st1 removes banana (added by st2!) -> must reload, delete, persist, and preserve egg!
        rem = st1.remove_meal(meal_id=banana_id)
        self.assertTrue(rem["removed"])
        self.assertEqual(len(rem["removed"]), 1)

        # Fresh process/StateManager sees banana permanently removed and egg intact
        fresh = StateManager(base_dir=self.base, ca1_log_path=None).get("u_del")
        meals = fresh.snapshot()["meals"]
        self.assertEqual([m["food"] for m in meals], ["egg"])
        self.assertAlmostEqual(fresh.snapshot()["consumed"]["calories"], 78.0, places=1)

        # Repeated deletion reports removed=[] without crashing
        rem_again = fresh.remove_meal(meal_id=banana_id)
        self.assertFalse(rem_again["removed"])
        self.assertEqual(len(rem_again["removed"]), 0)

        # Remove-last works deterministically
        rem_last = fresh.remove_meal(remove_last=True)
        self.assertTrue(rem_last["removed"])
        self.assertEqual(fresh.snapshot()["meals"], [])

        # If disk persistence fails during removal, exception propagates (never claims false success)
        st1.add_meals([{"food": "egg", "quantity_str": "1 piece", "calories": 78.0, "protein": 6.0, "carbs": 0.6, "fat": 5.3}])
        with patch("nutritrack.state.nutrition_state.atomic_write_json", side_effect=OSError("disk full")):
            with self.assertRaises((StateError, StateUpdateError, OSError)):
                st1.remove_meal(food_name="egg")
        self.assertEqual(len(StateManager(base_dir=self.base, ca1_log_path=None).get("u_del").snapshot()["meals"]), 1)

    def test_real_multiprocess_concurrent_adds_and_idempotency(self) -> None:
        """Four OS processes concurrently adding unique meals and retrying identical request_ids preserve exact totals."""
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=4) as pool:
            proc_results = pool.starmap(
                _mp_worker_add_unique,
                [(str(self.base), "mp_user", p, 8) for p in range(4)],
            )
        all_ids = [mid for sub in proc_results for mid in sub]
        self.assertEqual(len(all_ids), 32)
        self.assertEqual(len(set(all_ids)), 32)

        # Now spawn 4 processes hammering the SAME request_id -> must result in strictly +1 meal
        with ctx.Pool(processes=4) as pool:
            dup_res = pool.starmap(
                _mp_worker_add_same_request,
                [(str(self.base), "mp_user", "paneer", "shared-mp-req", 3) for _ in range(4)],
            )
        added_counts = [len(r["added"]) for sub in dup_res for r in sub]
        self.assertEqual(sum(added_counts), 1)

        final_snap = StateManager(base_dir=self.base, ca1_log_path=None).get("mp_user").snapshot()
        self.assertEqual(len(final_snap["meals"]), 33)
        self.assertAlmostEqual(final_snap["consumed"]["calories"], 32 * 100.0 + 150.0, places=1)

    def test_cross_process_run_log_rotation_and_fast_tail_read(self) -> None:
        log_path = self.base / "agent_runs.jsonl"
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=3) as pool:
            pool.starmap(
                _mp_worker_run_log,
                [(str(log_path), p, 25) for p in range(3)],
            )
        rl = RunLog(path=log_path, max_bytes=2048)
        recent = rl.read(limit=15)
        self.assertEqual(len(recent), 15)
        # Verify main and rotated files remain strict valid JSONL
        for suffix in ("", ".1", ".2", ".3"):
            p = Path(f"{log_path}{suffix}")
            if p.exists():
                for line in p.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        obj = json.loads(line)
                        self.assertIn("run_id", obj)
        self.assertEqual(rl.read(limit=0), [])

    def test_durable_operation_store_and_lock_registry_bounded(self) -> None:
        store = DurableOperationStore(base_dir=self.base)
        ph = compute_payload_hash("chat", {"message": "I ate 2 eggs"})
        act1, rec1 = store.begin("op_u", "op-1", endpoint="chat", payload_hash=ph)
        self.assertEqual(act1, "execute")
        self.assertEqual(rec1["status"], "processing")

        act_conflict, _ = store.begin("op_u", "op-1", endpoint="chat", payload_hash=compute_payload_hash("chat", {"message": "different"}))
        self.assertEqual(act_conflict, "conflict")

        act_busy, _ = store.begin("op_u", "op-1", endpoint="chat", payload_hash=ph)
        self.assertEqual(act_busy, "in_progress")

        store.complete("op_u", "op-1", status="succeeded", status_code=200, response={"status": "success", "calories": 156})
        # Fresh store instance simulating process restart still replays completed outcome
        restarted_store = DurableOperationStore(base_dir=self.base)
        act_replay, rec_replay = restarted_store.begin("op_u", "op-1", endpoint="chat", payload_hash=ph)
        self.assertEqual(act_replay, "replay")
        self.assertEqual(rec_replay["response"]["calories"], 156)

        # LockRegistry drops unused keys cleanly when released (zero memory leak across many users)
        reg = get_lock_registry()
        before = reg.active_count()
        for idx in range(500):
            with reg.lock(f"temp-user-lock-{idx}"):
                pass
        self.assertEqual(reg.active_count(), before)


if __name__ == "__main__":
    unittest.main()
