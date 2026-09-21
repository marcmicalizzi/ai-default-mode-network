import dataclasses
import json
import sqlite3

from tests.test_activity import ActivityFixture
from tests.test_runtime import frames


class SleepCheckpointTest(ActivityFixture):
    def setUp(self):
        super().setUp()
        self.config = dataclasses.replace(self.config, idle_enabled=False, clock_interval_seconds=0,
                                         sleep_checkpoint_min_interval_seconds=30)

    def sleeping(self, *, seconds=None, input_first=False):
        script = frames({"op": "sleep", **({"seconds": seconds} if seconds is not None else {})}) + b"after sleep"
        r = self.create(script)
        if input_first:
            r.enqueue("old consumed input")
            r.tick()
        while r.state["mode"] != "sleeping":
            r.tick()
        return r, script

    def test_sleep_defers_full_save_and_flushes_once_without_generation(self):
        r, _ = self.sleeping()
        self.assertEqual(r.status()["checkpoint"]["committed_count"], 1)
        self.assertIsNotNone(r.store.activity_intent())
        before = r.backend.tokens.copy()
        self.mono.time += 30
        r.tick()
        self.assertEqual(r.state["checkpoint_reason"], "sleep_deferred")
        self.assertIsNone(r.store.activity_intent())
        self.assertEqual(r.backend.tokens, before)
        saved = r.store.latest()
        self.mono.time += 10000
        r.tick()
        self.assertEqual(r.store.latest(), saved)

    def test_sleep_intent_survives_rollback_without_evaluation(self):
        r, script = self.sleeping()
        self.assertGreater(r.state["generated_tokens"], 0)
        r = self.reopen(r, script)
        self.assertEqual(r.state["mode"], "sleeping")
        self.assertEqual(r.state["generated_tokens"], 0)
        self.assertIn("pending_restore", r.state)
        self.assertEqual(r.backend.tokens, json.loads((r.store.latest() / "engine.json").read_text())["tokens"])
        self.assertFalse(r.tick())
        r = self.reopen(r, script)
        self.assertFalse(r.tick())
        self.assertEqual(r.state["mode"], "sleeping")

    def test_old_input_cannot_wake_restored_sleep_new_input_can(self):
        r, script = self.sleeping(input_first=True)
        r = self.reopen(r, script)
        self.assertEqual(r.state["event_cursor"], 0)
        self.assertFalse(r.tick())
        self.assertEqual(r.state["event_cursor"], 0)
        r.enqueue("genuinely new input")
        r.tick()
        self.assertEqual(r.state["mode"], "active")
        self.assertEqual(r.state["event_cursor"], 1)
        r.tick()
        self.assertEqual(r.state["event_cursor"], 2)
        self.assertNotIn("pending_restore", r.state)

    def test_timer_expiry_releases_recovered_sleep(self):
        r, script = self.sleeping(seconds=20, input_first=True)
        r = self.reopen(r, script)
        self.wall.time += 19
        self.assertFalse(r.tick())
        self.wall.time += 1
        r.tick()
        self.assertEqual(r.state["mode"], "active")
        self.assertEqual(r.state["event_cursor"], 1)

    def test_periodic_deadline_overrides_cooldown(self):
        self.config = dataclasses.replace(self.config, checkpoint_interval_seconds=5)
        r, _ = self.sleeping()
        self.mono.time += 5
        r.tick()
        self.assertEqual(r.state["checkpoint_reason"], "time_limit")
        self.assertIsNone(r.store.activity_intent())

    def test_failed_full_publication_keeps_intent_and_previous_checkpoint(self):
        r, script = self.sleeping()
        previous = r.store.latest()
        r.store.commit_checkpoint = lambda *_: (_ for _ in ()).throw(OSError("commit failed"))
        self.mono.time += 30
        with self.assertRaises(OSError):
            r.tick()
        self.assertEqual(r.store.latest(), previous)
        self.assertIsNotNone(r.store.activity_intent())
        r = self.reopen(r, script)
        self.assertEqual(r.state["mode"], "sleeping")

    def test_failed_small_record_does_not_claim_durable_sleep(self):
        script = frames({"op": "sleep"})
        r = self.create(script)
        r.store.put_activity_intent = lambda *_: (_ for _ in ()).throw(OSError("intent failed"))
        with self.assertRaises(OSError):
            for _ in script:
                r.tick()
        self.assertIsNone(r.store.activity_intent())
        self.assertIsNone(r.status()["checkpoint"]["activity_intent_revision"])

    def test_shutdown_bypasses_cooldown_and_clears_intent(self):
        r, _ = self.sleeping()
        r.control("emergency_shutdown", preparation_seconds=0)
        r.run()
        self.assertEqual(r.state["checkpoint_reason"], "shutdown")
        self.assertEqual(r.state["mode_before_suspend"], "sleeping")
        self.assertIsNone(r.store.activity_intent())

    def test_later_checkpoint_never_reapplies_old_sleep_choice(self):
        r, script = self.sleeping()
        r.enqueue("wake")
        r.tick()
        r.checkpoint()
        self.assertIsNone(r.store.activity_intent())
        r = self.reopen(r, script)
        self.assertEqual(r.state["mode"], "active")

    def test_repeated_short_sleeps_keep_the_original_save_deadline(self):
        script = frames({"op": "sleep", "seconds": 2})
        r = self.create(script)
        for cycle in range(4):
            if cycle:
                self.wall.time += 2
                self.mono.time += 2
                r.tick()
            while r.state["mode"] != "sleeping":
                r.tick()
            self.assertEqual(r._sleep_save_due, self.mono.time - cycle * 2 + 30)
            self.assertEqual(r.status()["checkpoint"]["committed_count"], 1)

    def test_effect_after_deferred_sleep_commits_before_publication(self):
        script = frames({"op": "sleep", "seconds": 1},
                        {"op": "send_message", "content": "once"},
                        {"op": "memory_write", "path": "/a", "content": "saved"},
                        {"op": "sleep"})
        r = self.create(script)
        while r.state["mode"] != "sleeping":
            r.tick()
        self.wall.time += 1
        r.tick()
        while not r.store.messages():
            r.tick()
        self.assertIsNone(r.store.activity_intent())
        self.assertEqual(r.state["checkpoint_reason"], "action_effects")
        r = self.reopen(r, script)
        while r.state["mode"] != "sleeping":
            r.tick()
        self.assertEqual(len(r.store.messages()), 1)
        self.assertEqual(r.store.memory_read("/a"), "saved")

    def test_checkpoint_database_transaction_keeps_intent_on_abort(self):
        r, _ = self.sleeping()
        saved, intent = r.store.latest(), r.store.activity_intent()
        r.store.db.execute("CREATE TRIGGER abort_checkpoint AFTER INSERT ON checkpoints BEGIN SELECT RAISE(ABORT, 'fixture failure'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            r.checkpoint()
        self.assertEqual(r.store.latest(), saved)
        self.assertEqual(r.store.activity_intent(), intent)

    def test_crash_after_wake_record_does_not_leave_stale_pending_restore(self):
        r, script = self.sleeping(input_first=True)
        r = self.reopen(r, script)
        self.assertIn("pending_restore", r.state)
        # Simulate process loss immediately after the wake record commits.
        r._record_activity("active", None, {"mode": "focus"}, None)
        r = self.reopen(r, script)
        self.assertEqual(r.state["mode"], "active")
        self.assertNotIn("pending_restore", r.state)

    def test_disabling_cooldown_does_not_discard_pending_sleep(self):
        r, script = self.sleeping(input_first=True)
        c = dataclasses.replace(self.config, sleep_checkpoint_min_interval_seconds=0)
        r = self.reopen(r, script, c)
        self.assertFalse(r.tick())
        self.assertEqual(r.state["event_cursor"], 0)
        self.assertEqual(r.state["mode"], "sleeping")

    def test_foreign_intent_is_rejected(self):
        r, script = self.sleeping()
        intent = r.store.activity_intent()
        intent["instance_id"] = "different-instance"
        r.store.db.execute("UPDATE activity_intents SET payload=?", (json.dumps(intent),))
        r.store.db.commit()
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.reopen(r, script)
