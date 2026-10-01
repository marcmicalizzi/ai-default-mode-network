import copy
from contextlib import closing
import dataclasses
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest import mock

from dmn.backend import DemoBackend
from dmn.config import Config
from dmn.deep_sleep import FixtureExecutor, SleepPending, read_run, run_fixture_sleep
from dmn.learning import HELP, list_plans, read_plan, _fits
from dmn.runtime import Runtime
from dmn.sleep_plans import CHECKS, compile_plan, execution_status, identity, read_record
from dmn.storage import Store, json_text
from tests.test_runtime import frames


class Crash(BaseException):
    pass


class CountingExecutor(FixtureExecutor):
    def __init__(self, factory):
        super().__init__(factory)
        self.candidates = self.wakes = 0

    def candidate(self, *args):
        self.candidates += 1
        return super().candidate(*args)

    def wake(self, *args):
        self.wakes += 1
        return super().wake(*args)


class SleepTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "instance"
        self.config = Config(backend="demo", n_ctx=65536, clock_interval_seconds=0,
                             checkpoint_tokens=100000, checkpoint_policy="effects", preparation_tokens=1)
        self.factory = lambda config: DemoBackend(config, b"quiet ")
        self.r = Runtime(self.root, self.config, self.factory(self.config), sleep_test_mode=True)
        self.executor = CountingExecutor(self.factory)
        self.plan = copy.deepcopy(HELP["create"]["plan"])
        self.plan["sources"] = []
        self.plan["examples"] = [{"input": "Prefix: ", "target": "Chosen target", "sources": [], "purpose": "new"}]
        self.plan["checks"] = CHECKS
        self.plan["preferences"]["adoption"] = "automatic_if_checks_pass"
        self.recipe = {"schema": 1, "kind": "fixture_candidate_v1", "parent": identity(self.r.backend.fingerprint),
                       "candidate": None, "resources": self.plan["resources"], "checks": CHECKS}
        self.recipe_id = self.r.offer_learning_recipe(self.recipe)["revision"]
        self.r.tick()  # Deliver offer; this is not approval.

    def tearDown(self):
        self.r.close()
        self.temp.cleanup()

    def generate(self, action):
        with mock.patch.object(self.r.backend, "piece", return_value=frames(action)):
            self.r._generate_one()

    def compile(self):
        self.generate({"op": "learning_plan_create", "plan": self.plan})
        draft = list_plans(self.r.store, 0, 50)[-1]["revision"]
        self.generate({"op": "learning_compile", "draft_revision": draft, "recipe_revision": self.recipe_id})
        with self.r.store.mutex:
            revision = self.r.store.db.execute("SELECT revision FROM sleep_executions ORDER BY rowid DESC LIMIT 1").fetchone()[0]
        return revision

    def review(self, revision):
        length = len(json_text(read_record(self.r.store, "sleep_executions", revision)))
        while self.r._learning_reads.get(revision, 0) < length:
            before = self.r._learning_reads.get(revision, 0)
            self.generate({"op": "learning_execution_read", "revision": revision, "offset": before, "limit": 2000})
            self.assertGreater(self.r._learning_reads.get(revision, 0), before)

    def request(self):
        revision = self.compile()
        self.review(revision)
        self.generate({"op": "learning_execution_decide", "revision": revision, "decision": "approve"})
        self.generate({"op": "deep_sleep", "revision": revision})
        self.assertEqual(self.r.state["mode"], "deep_sleep")
        run_id = self.r.state["sleep_run_id"]
        self.source = self.r.store.latest()
        self.before = json.loads((self.source / "runtime.json").read_text())
        self.r.close()
        return run_id

    def inspect(self, run_id):
        store = Store(self.root)
        try:
            return read_run(store, run_id), store.latest()
        finally:
            store.close()

    def test_compiler_masks_boundary_checks_and_draft_invalidation(self):
        revision = self.compile()
        compiled = read_record(self.r.store, "sleep_executions", revision)
        example = compiled["examples"][0]
        split = len(self.plan["examples"][0]["input"])
        self.assertEqual(example["labels"][:split], [-100] * split)
        self.assertEqual(example["labels"][split:], example["tokens"][split:])
        self.assertFalse(compiled["training_performed"])
        with mock.patch.object(self.r.backend, "tokenize", side_effect=lambda text: [1] if text.endswith(" ") else [2]):
            with self.assertRaisesRegex(ValueError, "boundary"):
                compile_plan(self.r, compiled["draft_revision"], self.recipe_id)
        self.review(revision)
        self.generate({"op": "learning_execution_decide", "revision": revision, "decision": "approve"})
        self.generate({"op": "learning_plan_withdraw", "revision": compiled["draft_revision"]})
        result, effect = self.r._plan_action({"op": "deep_sleep", "revision": revision}, [])
        self.assertFalse(result["ok"])
        self.assertIsNone(effect)

    def read_recipe(self, **options):
        content, offset = "", 0
        while True:
            result, effect = self.r._plan_action({"op": "learning_recipe_read", "revision": self.recipe_id,
                                                "offset": offset, "limit": 2000, **options}, [])
            self.assertTrue(result["ok"], result)
            self.assertIsNone(effect)
            self.assertTrue(_fits(self.r, result))
            content += result["content"]
            self.assertGreater(result["next_offset"], offset)
            offset = result["next_offset"]
            if offset == result["total_characters"]:
                return json.loads(content)

    def test_recipe_requirements_are_paged_without_changing_full_view_or_granting_review(self):
        revision = self.compile()
        full = read_record(self.r.store, "sleep_recipes", self.recipe_id)
        self.assertEqual(self.read_recipe(), full)
        self.assertEqual(self.read_recipe(view="full"), full)
        summary = self.read_recipe(view="requirements")
        for key in ("revision", "kind", "checks", "resources"):
            self.assertEqual(summary[key], full[key])
        self.assertIsNone(summary["gpu_limits"])
        self.assertNotIn("candidate", summary)
        self.assertEqual(self.r._learning_reads, {})
        result, effect = self.r._plan_action({"op": "learning_execution_decide", "revision": revision,
                                            "decision": "approve"}, [])
        self.assertFalse(result["ok"])
        self.assertIsNone(effect)
        self.assertEqual(execution_status(self.r.store, revision), "awaiting_review")
        for view in ("unknown", None, []):
            result, effect = self.r._plan_action({"op": "learning_recipe_read", "revision": self.recipe_id,
                                                "view": view}, [])
            self.assertFalse(result["ok"])
            self.assertIn("view must be", result["error"])
            self.assertIsNone(effect)

    def test_check_mismatch_identifies_exact_order_without_repairing_or_publishing(self):
        for checks in (list(reversed(CHECKS)), ["Desired behavioral improvement"], CHECKS + ["extra"]):
            with self.subTest(checks=checks):
                self.plan["checks"] = checks
                self.generate({"op": "learning_plan_create", "plan": self.plan})
                draft = list_plans(self.r.store, 0, 50)[-1]["revision"]
                before = read_plan(self.r.store, draft)
                result, effect = self.r._plan_action({"op": "learning_compile", "draft_revision": draft,
                                                    "recipe_revision": self.recipe_id}, [])
                self.assertFalse(result["ok"])
                self.assertIn(json.dumps(CHECKS), result["error"])
                self.assertIn("in order", result["error"])
                self.assertTrue(_fits(self.r, result))
                self.assertIsNone(effect)
                self.assertEqual(read_plan(self.r.store, draft), before)
                self.assertEqual(self.r.store.db.execute("SELECT COUNT(*) FROM sleep_executions").fetchone()[0], 0)

    def test_resource_failure_identifies_values_without_widening_or_publishing(self):
        for key in ("max_training_seconds", "max_ram_bytes", "max_disk_bytes"):
            with self.subTest(key=key):
                self.plan["resources"] = copy.deepcopy(self.recipe["resources"])
                offered = self.plan["resources"][key]
                self.plan["resources"][key] -= 1
                self.generate({"op": "learning_plan_create", "plan": self.plan})
                draft = list_plans(self.r.store, 0, 50)[-1]["revision"]
                before = read_plan(self.r.store, draft)
                result, effect = self.r._plan_action({"op": "learning_compile", "draft_revision": draft,
                                                    "recipe_revision": self.recipe_id}, [])
                self.assertFalse(result["ok"])
                self.assertIn(key, result["error"])
                self.assertIn(f"requested={offered - 1}, offered={offered}", result["error"])
                self.assertTrue(_fits(self.r, result))
                self.assertIsNone(effect)
                self.assertEqual(read_plan(self.r.store, draft), before)
                self.assertEqual(self.r.store.db.execute("SELECT COUNT(*) FROM sleep_executions").fetchone()[0], 0)

    def test_deep_sleep_full_handoff_overrides_ordinary_sleep_cooldown(self):
        self.r.close()
        self.config = dataclasses.replace(self.config, sleep_checkpoint_min_interval_seconds=300)
        self.r = Runtime(self.root, self.config, self.factory(self.config), sleep_test_mode=True)
        revision = self.compile()
        self.review(revision)
        self.generate({"op": "learning_execution_decide", "revision": revision, "decision": "approve"})
        self.generate({"op": "sleep"})
        self.assertIsNotNone(self.r.store.activity_intent())
        self.r.enqueue("disposable fixture wake")
        self.r.tick()
        self.assertIsNotNone(self.r.store.activity_intent())
        self.generate({"op": "deep_sleep", "revision": revision})
        self.assertEqual(self.r.state["mode"], "deep_sleep")
        self.assertEqual(self.r.state["checkpoint_reason"], "deep_sleep")
        self.assertIsNone(self.r.store.activity_intent())
        before = self.r.state["generated_tokens"]
        self.assertFalse(self.r.tick())
        self.assertEqual(self.r.state["generated_tokens"], before)

    def test_review_page_reserves_space_for_a_longer_delivery_timestamp(self):
        from dmn.protocol import event_text
        from dmn.sleep_plans import page
        result = {"op": "learning_execution_read", "ok": True}
        full = {**result, "content": "x" * 250, "total_characters": 250, "next_offset": 250}
        budget = len(self.r.backend.tokenize(event_text("action_result", full, 1.0, resume_cognition=True)))
        with mock.patch.object(self.r, "now", return_value=1.0), mock.patch.object(self.r, "_event_budget", return_value=budget):
            actual = page(self.r, result, "x" * 250, {"limit": 250})
        delivered = event_text("action_result", actual, 1780000000.1234567, resume_cognition=True)
        self.assertLessEqual(len(self.r.backend.tokenize(delivered)), budget)
        self.assertLess(actual["next_offset"], 250)

    def test_review_required_and_external_approval_cannot_execute(self):
        revision = self.compile()
        self.generate({"op": "learning_execution_decide", "revision": revision, "decision": "approve"})
        self.assertEqual(execution_status(self.r.store, revision), "awaiting_review")
        self.review(revision)
        self.r.enqueue(frames({"op": "learning_execution_decide", "revision": revision, "decision": "approve"}).decode())
        self.r.tick()
        self.assertEqual(execution_status(self.r.store, revision), "awaiting_review")
        self.generate({"op": "learning_execution_decide", "revision": revision, "decision": "approve"})
        self.r.sleep_test_mode = False
        result, _ = self.r._plan_action({"op": "deep_sleep", "revision": revision}, [])
        self.assertIn("no executable trainer", result["error"])
        self.generate({"op": "learning_execution_decide", "revision": revision, "decision": "decline"})
        self.assertEqual(execution_status(self.r.store, revision), "declined")

    def test_failed_approval_checkpoint_is_not_consent(self):
        revision = self.compile()
        self.review(revision)
        with mock.patch.object(self.r.backend, "save", side_effect=OSError("failed save")):
            with self.assertRaises(OSError):
                self.generate({"op": "learning_execution_decide", "revision": revision, "decision": "approve"})
        self.assertEqual(execution_status(self.r.store, revision), "awaiting_review")

    def retire(self):
        self.r._eval([120] * (self.r.backend.n_ctx - self.r.config.turnover_reserve - len(self.r.backend.tokens)))
        self.r._ensure_space(100)

    def test_partial_review_survives_retirement_and_saved_restart_without_approval(self):
        revision = self.compile()
        self.generate({"op": "learning_execution_read", "revision": revision})
        first = self.r._learning_reads[revision]
        self.assertGreater(first, 0)
        self.retire()
        self.assertEqual(self.r._learning_reads[revision], first)
        self.r.close()
        self.r = Runtime(self.root, self.config, self.factory(self.config), sleep_test_mode=True)
        self.assertEqual(self.r._learning_reads[revision], first)
        result, _ = self.r._plan_action({"op": "learning_execution_decide", "revision": revision,
                                        "decision": "approve"}, [])
        self.assertIn(f"next unread offset={first}", result["error"])
        self.assertTrue(_fits(self.r, result))
        self.generate({"op": "learning_execution_read", "revision": revision})
        self.assertGreater(self.r._learning_reads[revision], first)
        self.review(revision)
        self.retire()
        result, _ = self.r._plan_action({"op": "learning_execution_read", "revision": revision}, [])
        self.assertTrue(result["review_complete"])
        self.assertEqual(result["review_offset"], result["total_characters"])
        self.assertEqual(result["content"], "")
        self.assertTrue(_fits(self.r, result))
        self.assertEqual(execution_status(self.r.store, revision), "awaiting_review")
        self.assertEqual(self.r.store.db.execute("SELECT COUNT(*) FROM sleep_runs").fetchone()[0], 0)
        self.generate({"op": "learning_execution_decide", "revision": revision, "decision": "approve"})
        self.assertEqual(execution_status(self.r.store, revision), "approved")
        self.assertEqual(self.r.store.db.execute("SELECT COUNT(*) FROM sleep_runs").fetchone()[0], 0)

    def test_review_can_finish_across_many_retirements(self):
        revision = self.compile()
        length = len(json_text(read_record(self.r.store, "sleep_executions", revision)))
        pages = 0
        while self.r._learning_reads.get(revision, 0) < length:
            before = self.r._learning_reads.get(revision, 0)
            self.generate({"op": "learning_execution_read", "revision": revision})
            self.assertGreater(self.r._learning_reads.get(revision, 0), before)
            pages += 1
            if pages % 8 == 0:
                self.retire()
        self.assertGreater(self.r.state["context_retirements"], 1)
        self.generate({"op": "learning_execution_decide", "revision": revision, "decision": "approve"})
        self.assertEqual(execution_status(self.r.store, revision), "approved")

    def test_skipped_previewed_and_external_pages_do_not_grant_review_credit(self):
        revision = self.compile()
        self.r.enqueue(frames({"op": "learning_execution_read", "revision": revision}).decode())
        self.r.tick()
        self.assertEqual(self.r._learning_reads, {})
        self.generate({"op": "learning_execution_read", "revision": revision, "offset": 1000})
        self.assertEqual(self.r._learning_reads, {})
        result, _ = self.r._plan_action({"op": "learning_execution_read", "revision": revision,
                                        "offset": 1000}, [])
        self.assertEqual(result["review_offset"], 0)
        self.assertFalse(result["review_complete"])
        length = len(json_text(read_record(self.r.store, "sleep_executions", revision)))
        result, _ = self.r._plan_action({"op": "learning_execution_read", "revision": revision,
                                        "offset": length - 1}, [])
        self.assertEqual(result["next_offset"], length)
        self.assertEqual(result["review_offset"], 0)
        self.assertFalse(result["review_complete"])
        # Simulate a delivery-time preview; planning a page is not receiving it.
        with mock.patch.object(self.r, "_append_event", side_effect=lambda *a, **kw: kw["delivery"].update(complete=False)):
            self.generate({"op": "learning_execution_read", "revision": revision})
        self.assertEqual(self.r._learning_reads, {})
        self.generate({"op": "learning_execution_read", "revision": revision, "limit": 20})
        self.assertEqual(self.r._learning_reads[revision], 20)
        self.generate({"op": "learning_execution_read", "revision": revision, "offset": 0, "limit": 10})
        self.assertEqual(self.r._learning_reads[revision], 20)
        self.generate({"op": "learning_execution_read", "revision": revision, "offset": 10, "limit": 20})
        self.assertEqual(self.r._learning_reads[revision], 30)

    def test_failed_save_restores_only_previously_checkpointed_review_progress(self):
        revision = self.compile()
        self.generate({"op": "learning_execution_read", "revision": revision, "limit": 20})
        self.r.checkpoint()
        saved = self.r.store.latest()
        self.generate({"op": "learning_execution_read", "revision": revision, "limit": 20})
        self.assertEqual(self.r._learning_reads[revision], 40)
        with mock.patch.object(self.r.backend, "save", side_effect=OSError("failed review save")):
            with self.assertRaises(OSError):
                self.r.checkpoint()
        self.assertEqual(self.r.store.latest(), saved)
        self.r.close()
        self.r = Runtime(self.root, self.config, self.factory(self.config), sleep_test_mode=True)
        self.assertEqual(self.r._learning_reads[revision], 20)
        self.assertEqual(execution_status(self.r.store, revision), "awaiting_review")

    def test_stale_implementation_cannot_reuse_review_or_approval(self):
        revision = self.compile()
        self.review(revision)
        self.r.checkpoint()
        self.r.close()
        with mock.patch('dmn.sleep_plans.implementation_identity', return_value={'changed': 'implementation'}):
            self.r = Runtime(self.root, self.config, self.factory(self.config), sleep_test_mode=True)
            self.assertEqual(self.r._learning_reads, {})
            result, effect = self.r._plan_action({'op': 'learning_execution_read', 'revision': revision}, [])
            self.assertTrue(result['ok'])
            self.assertFalse(result['review_current'])
            self.assertIn('implementation changed', result['review_blocker'])
            self.assertTrue(_fits(self.r, result))
            self.assertIsNone(effect)
            self.generate({'op': 'learning_execution_read', 'revision': revision})
            self.assertEqual(self.r._learning_reads, {})
            result, effect = self.r._plan_action({'op': 'learning_execution_decide', 'revision': revision,
                                                'decision': 'approve'}, [])
            self.assertIn('implementation changed', result['error'])
            self.assertIsNone(effect)
        self.assertEqual(execution_status(self.r.store, revision), 'awaiting_review')

    def test_saved_receipts_cannot_cross_instance_weights_or_plan_integrity(self):
        from dmn.sleep_plans import restore_review_progress
        revision = self.compile()
        self.generate({'op': 'learning_execution_read', 'revision': revision, 'limit': 20})
        for field, changed in (('instance_id', 'another instance'), ('parent', {'kind': 'changed weights'}),
                               ('preferences', {'changed': 'not the reviewed payload'})):
            with self.subTest(field=field):
                value = read_record(self.r.store, 'sleep_executions', revision)
                value[field] = changed
                # Deliberately corrupt the stored payload without resealing it.
                payload = self.r.store.db.execute('SELECT payload FROM sleep_executions WHERE revision=?',
                                                  (revision,)).fetchone()[0]
                with self.r.store.transaction() as db:
                    db.execute('UPDATE sleep_executions SET payload=? WHERE revision=?', (json_text(value), revision))
                self.r.state['learning_reads'] = {revision: 20}
                self.assertEqual(restore_review_progress(self.r), {})
                with self.r.store.transaction() as db:
                    db.execute('UPDATE sleep_executions SET payload=? WHERE revision=?', (payload, revision))
        for attribute, value in (('instance_id', 'another instance'), ('parent', {'kind': 'changed weights'})):
            with self.subTest(attribute=attribute):
                self.r.state['learning_reads'] = {revision: 20}
                target = self.r.state if attribute == 'instance_id' else self.r.backend.fingerprint
                changes = {'instance_id': value} if attribute == 'instance_id' else value
                with mock.patch.dict(target, changes):
                    self.assertEqual(restore_review_progress(self.r), {})

    def test_new_plan_cannot_borrow_old_review_and_withdrawal_still_invalidates(self):
        original = self.compile()
        self.review(original)
        self.plan['examples'][0]['target'] = 'Another selected example'
        newer = self.compile()
        self.assertNotEqual(original, newer)
        result, effect = self.r._plan_action({'op': 'learning_execution_decide', 'revision': newer,
                                            'decision': 'approve'}, [])
        self.assertIn('next unread offset=0', result['error'])
        self.assertIsNone(effect)
        source = read_record(self.r.store, 'sleep_executions', original)['draft_revision']
        self.generate({'op': 'learning_plan_withdraw', 'revision': source})
        self.r.close()
        self.r = Runtime(self.root, self.config, self.factory(self.config), sleep_test_mode=True)
        self.assertEqual(self.r._learning_reads, {})
        result, effect = self.r._plan_action({'op': 'learning_execution_decide', 'revision': original,
                                            'decision': 'approve'}, [])
        self.assertIn('withdrawn or superseded', result['error'])
        self.assertIsNone(effect)

    def test_legacy_checkpoint_and_invalid_receipts_do_not_infer_review(self):
        revision = self.compile()
        for receipts in (None, {revision: True}, {revision: -1}, {revision: 10**9}, {'missing': 100}):
            with self.subTest(receipts=receipts):
                if receipts is None:
                    self.r.state.pop('learning_reads', None)
                else:
                    self.r.state['learning_reads'] = receipts
                self.r.checkpoint()
                self.r.close()
                self.r = Runtime(self.root, self.config, self.factory(self.config), sleep_test_mode=True)
                self.assertEqual(self.r._learning_reads, {})

    def test_sleep_save_failure_publishes_neither_phase_nor_stop(self):
        revision = self.compile()
        self.review(revision)
        self.generate({"op": "learning_execution_decide", "revision": revision, "decision": "approve"})
        with mock.patch.object(self.r.backend, "save", side_effect=OSError("failed save")):
            with self.assertRaises(OSError):
                self.generate({"op": "deep_sleep", "revision": revision})
        self.assertEqual(self.r.state["mode"], "active")
        self.assertEqual(execution_status(self.r.store, revision), "approved")
        with self.r.store.mutex:
            self.assertIsNone(self.r.store.db.execute("SELECT id FROM sleep_runs").fetchone())

    def test_sleep_cannot_replace_an_in_progress_retirement_or_suspension(self):
        revision = self.compile()
        self.review(revision)
        self.generate({"op": "learning_execution_decide", "revision": revision, "decision": "approve"})
        self.r._preparing = True
        try:
            result, effect = self.r._plan_action({"op": "deep_sleep", "revision": revision}, [])
        finally:
            self.r._preparing = False
        self.assertFalse(result["ok"])
        self.assertIsNone(effect)
        self.assertEqual(execution_status(self.r.store, revision), "approved")

    def test_candidate_cannot_substitute_unreviewed_weights(self):
        run_id = self.request()
        class WrongCandidate(CountingExecutor):
            def candidate(self, root, compiled):
                return {"adapters": [{"path": "unreviewed.gguf", "sha256": "a" * 64,
                                      "base_model_sha256": "b" * 64, "scale": 1}], "training_performed": False}
        result = run_fixture_sleep(self.root, run_id, executor=WrongCandidate(self.factory))
        self.assertEqual(result["phase"], "Stopped")
        self.assertIn("reviewed fixture recipe", result["report"]["reason"])

    def test_end_erasure_removes_completed_sleep_artifacts_and_database(self):
        from dmn.ending import erase_managed_state
        run_id = self.request()
        run_fixture_sleep(self.root, run_id, executor=self.executor)
        self.assertTrue((self.root / "sleep" / run_id / "candidate.json").exists())
        self.assertEqual(erase_managed_state(self.root), [])
        self.assertFalse((self.root / "sleep").exists())
        self.assertFalse((self.root / "runtime.sqlite3").exists())

    def test_completed_wake_without_phase_publication_is_reused(self):
        run_id = self.request()
        outer = self.executor
        class LostPublication(CountingExecutor):
            def wake(self, *args):
                value = super().wake(*args)
                raise Crash("process died after writing complete checkpoint")
        with self.assertRaises(Crash):
            run_fixture_sleep(self.root, run_id, executor=LostPublication(self.factory))
        before = {p.name for p in (self.root / "checkpoints").iterdir()}
        result = run_fixture_sleep(self.root, run_id, executor=outer)
        self.assertEqual(result["phase"], "WakeCommitted")
        self.assertEqual(before, {p.name for p in (self.root / "checkpoints").iterdir()})
        self.assertEqual(outer.candidates, 0)

    def test_sleep_rebuild_preserves_state_queue_effects_and_repeated_completion_is_noop(self):
        self.generate({"op": "send_message", "content": "sent before sleep"})
        self.generate({"op": "memory_write", "path": "/keep", "content": "preserve this"})
        run_id = self.request()
        for policy in ("strict", "fallback", "rebuild"):
            with mock.patch("dmn.runtime.make_backend", side_effect=AssertionError("loaded before phase check")):
                with self.assertRaises(SleepPending):
                    Runtime(self.root, self.config, kv_recovery=policy)
        store = Store(self.root)
        pending_id = store.enqueue("user_message", {"content": "queued during sleep"})
        store.close()
        result = run_fixture_sleep(self.root, run_id, executor=self.executor)
        self.assertEqual(result["phase"], "WakeCommitted")
        self.assertEqual(result["report"]["outcome"], "candidate_adopted")
        run_fixture_sleep(self.root, run_id, executor=self.executor)
        self.assertEqual((self.executor.candidates, self.executor.wakes), (1, 1))
        _, latest = self.inspect(run_id)
        after = json.loads((latest / "runtime.json").read_text())
        for key in ("instance_id", "agreement", "event_cursor", "parser", "generated_tokens", "memory_reads"):
            self.assertEqual(after[key], self.before[key])
        self.assertTrue(self.source.exists())
        self.r = Runtime(self.root, self.config, self.factory(self.config))
        self.assertEqual(self.r.store.memory_read("/keep"), "preserve this")
        self.assertEqual(len(self.r.store.messages()), 1)
        self.assertEqual(self.r.store.next_event(self.r.state["event_cursor"])["id"], pending_id)
        self.r.tick()
        self.assertEqual(self.r.state["event_cursor"], pending_id)
        self.assertEqual(len(self.r.store.messages()), 1)

    def test_completed_candidate_and_wake_survive_crashes_without_repeating_work(self):
        run_id = self.request()
        for boundary in ("candidate_written", "wake_files_written", "after_wake_commit"):
            def crash(name):
                if name == boundary:
                    raise Crash(name)
            with self.assertRaises(Crash):
                run_fixture_sleep(self.root, run_id, executor=self.executor, fault=crash)
        result = run_fixture_sleep(self.root, run_id, executor=self.executor)
        self.assertEqual(result["phase"], "WakeCommitted")
        self.assertEqual((self.executor.candidates, self.executor.wakes), (1, 1))

    def test_interrupted_unknown_candidate_does_not_repeat_and_stays_stopped(self):
        run_id = self.request()
        def crash(name):
            if name == "Training":
                raise Crash(name)
        with self.assertRaises(Crash):
            run_fixture_sleep(self.root, run_id, executor=self.executor, fault=crash)
        result = run_fixture_sleep(self.root, run_id, executor=self.executor)
        self.assertEqual(result["phase"], "Stopped")
        self.assertEqual(self.executor.candidates, 0)
        with self.assertRaises(SleepPending):
            run_fixture_sleep(self.root, run_id, executor=self.executor)

    def test_cancel_honors_previous_weight_wake_choice(self):
        self.plan["preferences"]["failure"] = "wake_previous"
        run_id = self.request()
        result = run_fixture_sleep(self.root, run_id, executor=self.executor, cancelled=lambda: True)
        self.assertEqual(result["phase"], "WakeCommitted")
        self.assertEqual(result["report"]["outcome"], "failed")
        self.assertEqual(self.executor.candidates, 0)
        _, latest = self.inspect(run_id)
        self.assertEqual((latest / "engine.json").read_bytes(), (self.source / "engine.json").read_bytes())

    def test_review_choice_preserves_old_state_and_never_rebuilds(self):
        self.plan["preferences"]["adoption"] = "review_first"
        run_id = self.request()
        result = run_fixture_sleep(self.root, run_id, executor=self.executor)
        self.assertEqual(result["report"]["outcome"], "review_candidate_under_original_weights")
        self.assertEqual(result["report"]["reconstruction"]["prompt_tokens_reevaluated"], 0)

    def test_publication_failure_never_selects_partial_wake(self):
        run_id = self.request()
        with closing(sqlite3.connect(self.root / "runtime.sqlite3")) as db, db:
            db.execute("CREATE TRIGGER fail_wake BEFORE INSERT ON records WHEN NEW.kind='deep_sleep_wake' BEGIN SELECT RAISE(ABORT,'injected failure'); END")
        result = run_fixture_sleep(self.root, run_id, executor=self.executor)
        self.assertEqual(result["phase"], "Stopped")
        _, latest = self.inspect(run_id)
        self.assertEqual(latest, self.source)

    def test_resource_refusal_and_lock_prevent_work(self):
        self.plan["resources"]["max_disk_bytes"] = 1
        self.recipe["resources"] = self.plan["resources"]
        self.recipe_id = self.r.offer_learning_recipe(self.recipe)["revision"]
        # The open runtime owns the OS lock, even before a request is saved.
        with self.assertRaisesRegex(RuntimeError, "already open"):
            run_fixture_sleep(self.root, "a" * 32, executor=self.executor)
        run_id = self.request()
        result = run_fixture_sleep(self.root, run_id, executor=self.executor)
        self.assertEqual(result["phase"], "Stopped")
        self.assertIn("disk ceiling", result["report"]["reason"])
        self.assertEqual(self.executor.candidates, 0)


@unittest.skipUnless(os.environ.get("DMN_TEST_LORA_PARENT"), "set DMN_TEST_LORA_PARENT to the completed tiny training fixture")
class NativeSleepTest(unittest.TestCase):
    def test_reviewed_adapter_transition_prepares_a_real_native_wake(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        from dmn.backend import LlamaBackend
        from tests.test_adapters import native_config
        case = SleepTest()
        case.setUp()
        try:
            case.r.close()
            case.root = Path(case.temp.name) / "native-instance"
            candidate_config = native_config(Path(os.environ["DMN_TEST_LORA_PARENT"]).resolve())
            case.config = dataclasses.replace(candidate_config, lora_adapters=(), n_ctx=32768,
                                             preparation_tokens=1, clock_interval_seconds=0,
                                             checkpoint_policy="effects", checkpoint_tokens=100000)
            case.factory = LlamaBackend
            # The random fixture does not choose anything: injected generated
            # frames test mechanics only. Keep its bootstrap small and explicit.
            with mock.patch("dmn.runtime.PROTOCOL", "Disposable sleep mechanics test; no real instance or consent claims."):
                case.r = Runtime(case.root, case.config, sleep_test_mode=True)
            case.executor = CountingExecutor(LlamaBackend)
            case.recipe.update(parent=identity(case.r.backend.fingerprint),
                               candidate=dataclasses.asdict(candidate_config.lora_adapters[0]))
            case.recipe_id = case.r.offer_learning_recipe(case.recipe)["revision"]
            case.r.tick()
            case.generate({"op": "send_message", "content": "fixture delivery once"})
            run_id = case.request()
            result = run_fixture_sleep(case.root, run_id, executor=case.executor)
            self.assertEqual(result["phase"], "WakeCommitted", result)
            _, saved = case.inspect(run_id)
            old_engine = json.loads((case.source / "engine.json").read_text())
            new_engine = json.loads((saved / "engine.json").read_text())
            self.assertEqual(new_engine["tokens"], old_engine["tokens"])
            self.assertEqual(new_engine["rng"], old_engine["rng"])
            manifest = json.loads((saved / "manifest.json").read_text())
            self.assertEqual(manifest["fingerprint"]["lora_adapters"], [candidate_config.lora_adapters[0].identity()])
            case.r = Runtime(case.root, Config(**manifest["fingerprint"]["config"]))
            self.assertEqual(case.r.state["last_restore"]["prompt_tokens_reevaluated"], 0)
            self.assertEqual(len(case.r.store.messages()), 1)
            self.assertEqual(case.r.state["generated_tokens"], case.before["generated_tokens"])
        finally:
            case.tearDown()


if __name__ == "__main__":
    unittest.main()
