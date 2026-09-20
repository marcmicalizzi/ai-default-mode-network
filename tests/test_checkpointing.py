import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

from dmn.backend import DemoBackend
from dmn.config import Config
from dmn.runtime import Runtime
from tests.test_runtime import FakeClock, frames


class CheckpointPolicyTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "instance"
        self.wall, self.mono = FakeClock(), FakeClock()
        self.config = Config(backend="demo", n_ctx=12288, clock_interval_seconds=0,
                             checkpoint_policy="effects", preparation_tokens=8)
        self.opened = []

    def tearDown(self):
        for r in self.opened:
            r.close()
        self.temp.cleanup()

    def create(self, script=b"quiet text ", config=None):
        config = config or self.config
        r = Runtime(self.root, config, DemoBackend(config, script), now=self.wall, monotonic=self.mono)
        self.opened.append(r)
        return r

    def reopen(self, r, script, config=None):
        r.close()
        self.opened.remove(r)
        return self.create(script, config)

    def drive(self, r, n):
        for _ in range(n):
            r.tick()

    def test_reads_and_inputs_defer_but_token_limit_still_saves(self):
        script = frames({"op": "memory_list"}, {"op": "clock"})
        r = self.create(script)
        saved = r.store.latest()
        r.enqueue("durable input, delivery not yet checkpointed")
        r.tick()
        self.drive(r, len(script))
        self.assertEqual(r.store.latest(), saved)
        self.assertEqual(r.state["event_cursor"], 1)
        self.assertEqual(r.status()["checkpoint"]["unsaved_generated_tokens"], len(script))
        self.drive(r, self.config.checkpoint_tokens - len(script))
        self.assertNotEqual(r.store.latest(), saved)
        self.assertEqual(r.state["checkpoint_reason"], "token_limit")
        self.assertFalse(r.status()["checkpoint"]["dirty"])
        self.assertEqual(r.status()["checkpoint"]["unsaved_generated_tokens"], 0)

    def test_effects_sleep_and_memory_read_permissions_remain_atomic(self):
        script = frames({"op": "memory_write", "path": "/a", "content": "first"},
                        {"op": "memory_read", "path": "/a"},
                        {"op": "memory_write", "path": "/a", "content": "second", "expected_revision": 1},
                        {"op": "send_message", "content": "once"}, {"op": "sleep"})
        r = self.create(script)
        self.drive(r, len(script))
        self.assertEqual(r.store.memory_read("/a"), "second")
        self.assertEqual(len(r.store.messages()), 1)
        self.assertEqual(r.status()["checkpoint"]["committed_count"], 5)
        self.assertEqual(r.state["mode"], "sleeping")
        r = self.reopen(r, script)
        self.drive(r, len(script))
        self.assertEqual(len(r.store.messages()), 1)
        self.assertEqual(r.store.memory_read("/a"), "second")
        self.assertEqual(r.state["last_restore"]["prompt_tokens_reevaluated"], 0)

    def test_deferred_input_survives_rollback_and_delivers_again(self):
        r = self.create()
        r.enqueue("received before crash")
        r.tick()
        self.assertEqual(r.state["event_cursor"], 1)
        r = self.reopen(r, b"quiet text ")
        self.assertEqual(r.state["event_cursor"], 0)
        self.assertEqual(r.store.next_event(0)["payload"]["content"], "received before crash")
        r.tick()
        self.assertEqual(r.state["event_cursor"], 1)
        r.checkpoint()
        r = self.reopen(r, b"quiet text ")
        self.assertEqual(r.state["event_cursor"], 1)
        self.assertIsNone(r.store.next_event(1))
        self.assertEqual("".join(map(chr, r.backend.tokens)).count("received before crash"), 1)

    def test_time_threshold_uses_monotonic_clock_and_requires_changes(self):
        config = dataclasses.replace(self.config, checkpoint_tokens=0, checkpoint_interval_seconds=10)
        r = self.create(config=config)
        saved = r.store.latest()
        r.tick()
        self.wall.time += 100000
        r.tick()
        self.assertEqual(r.store.latest(), saved)
        self.wall.time -= 200000
        self.mono.time += 10
        r.tick()
        self.assertEqual(r.state["checkpoint_reason"], "time_limit")
        saved = r.store.latest()
        self.mono.time += 100
        r._periodic_checkpoint()
        self.assertEqual(r.store.latest(), saved)
        self.assertEqual(r.status()["checkpoint"]["age_seconds"], 100)

    def test_slow_save_does_not_immediately_retrigger_time_threshold(self):
        config = dataclasses.replace(self.config, checkpoint_tokens=0, checkpoint_interval_seconds=10)
        r = self.create(config=config)
        save = r.backend.save
        def slow_save(path):
            self.mono.time += 20
            save(path)
        r.backend.save = slow_save
        self.mono.time += 10
        r.tick()
        count = r.status()["checkpoint"]["committed_count"]
        self.assertEqual(r.status()["checkpoint"]["last"]["duration_seconds"], 20)
        r.tick()
        self.assertEqual(r.status()["checkpoint"]["committed_count"], count)

    def test_sleeping_creates_no_periodic_traffic_or_generation(self):
        script = frames({"op": "sleep"})
        config = dataclasses.replace(self.config, checkpoint_interval_seconds=1)
        r = self.create(script, config)
        self.drive(r, len(script))
        saved, generated = r.store.latest(), r.state["generated_tokens"]
        self.mono.time += 86400
        self.wall.time += 86400
        self.drive(r, 10)
        self.assertEqual(r.store.latest(), saved)
        self.assertEqual(r.state["generated_tokens"], generated)

    def test_failed_save_preserves_durable_timestamp_and_exposure(self):
        r = self.create()
        r.tick()
        prior, timestamp = r.store.latest(), r.state["checkpoint_at"]
        self.wall.time += 30
        r.backend.save = lambda _: (_ for _ in ()).throw(OSError("disk unavailable"))
        with self.assertRaisesRegex(OSError, "disk unavailable"):
            r.checkpoint()
        self.assertEqual(r.store.latest(), prior)
        self.assertEqual(r.state["checkpoint_at"], timestamp)
        status = r.status()["checkpoint"]
        self.assertEqual(status["committed_count"], 1)
        self.assertEqual(status["failed_count"], 1)
        self.assertEqual(status["unsaved_generated_tokens"], 1)
        self.assertFalse(status["in_progress"])
        self.assertTrue(status["dirty"])

    def test_effect_cannot_publish_when_database_commit_fails(self):
        script = frames({"op": "send_message", "content": "not durable"})
        r = self.create(script)
        prior = r.store.latest()
        r.store.commit_checkpoint = lambda *args: (_ for _ in ()).throw(OSError("commit failed"))
        with self.assertRaisesRegex(OSError, "commit failed"):
            self.drive(r, len(script))
        self.assertEqual(r.store.messages(), [])
        self.assertEqual(r.store.latest(), prior)
        self.assertEqual(r.status()["checkpoint"]["committed_count"], 1)

    def test_zero_preparation_shutdown_does_not_decode_or_retire(self):
        r = self.create()
        r._eval([ord("x")] * (r.backend.n_ctx - len(r.backend.tokens) - 1))
        r.backend.can_shift = False
        r.enqueue("pending during shutdown")
        before = r.backend.decoded_tokens
        r.control("shutdown", preparation_seconds=0)
        r.run()
        self.assertEqual(r.backend.decoded_tokens, before)
        self.assertEqual(r.state["generated_tokens"], 0)
        self.assertEqual(r.state["mode"], "suspended")
        self.assertEqual(r.state["context_retirements"], 0)
        self.assertEqual(r.state["last_suspension"]["stopped_reason"], "deadline")
        self.assertFalse(r.state["last_suspension"]["notice_delivered"])
        self.assertEqual(r.state["event_cursor"], 0)
        saved = json.loads((r.store.latest() / "runtime.json").read_text())
        self.assertEqual(saved["mode"], "suspended")

    def test_preparation_deadline_checked_between_native_calls(self):
        config = dataclasses.replace(self.config, preparation_tokens=32, suspend_preparation_seconds=3)
        r = self.create(config=config)
        evaluate = r.backend.eval
        def slow_eval(tokens):
            evaluate(tokens)
            self.mono.time += 1
        r.backend.eval = slow_eval
        r.control("suspend")
        r.tick()
        self.assertEqual(r.state["last_suspension"]["preparation_tokens_used"], 2)
        self.assertEqual(r.state["last_suspension"]["stopped_reason"], "deadline")
        self.assertEqual(r.state["mode"], "suspended")

    def test_more_urgent_request_interrupts_preparation_at_next_boundary(self):
        r = self.create()
        sample = r.backend.sample
        def urgent_request():
            r.control("shutdown", preparation_seconds=0)
            r.control("shutdown", preparation_seconds=300)
            return sample()
        r.backend.sample = urgent_request
        r.control("suspend", preparation_seconds=100)
        r.tick()
        self.assertEqual(r.state["last_suspension"]["preparation_tokens_used"], 1)
        self.assertEqual(r.state["checkpoint_reason"], "shutdown")

    def test_failed_shutdown_save_does_not_claim_suspended(self):
        r = self.create()
        prior = r.store.latest()
        r.backend.save = lambda _: (_ for _ in ()).throw(OSError("disk unavailable"))
        r.control("shutdown", preparation_seconds=0)
        with self.assertRaises(OSError):
            r.run()
        self.assertEqual(r.state["mode"], "error")
        self.assertEqual(r.store.latest(), prior)
        self.assertEqual(r.status()["mode"], "error")

    def test_policy_changes_restore_without_rebuilding_or_replacing_prefix(self):
        r = self.create(config=dataclasses.replace(self.config, checkpoint_policy="all_actions"))
        prefix = r.backend.tokens[:r.state["keep_prefix"]]
        config = dataclasses.replace(self.config, checkpoint_tokens=4096, checkpoint_interval_seconds=3600,
                                     suspend_preparation_seconds=30)
        r = self.reopen(r, b"quiet text ", config)
        self.assertEqual(r.state["last_restore"]["prompt_tokens_reevaluated"], 0)
        self.assertEqual(r.backend.tokens[:len(prefix)], prefix)
        changes = r.state["last_restore"]["scheduling_changes"]
        self.assertEqual(changes["checkpoint_policy"], {"previous": "all_actions", "current": "effects"})

    def test_snapshot_metrics_count_committed_files_only(self):
        r = self.create()
        actual = sum(p.stat().st_size for p in r.store.latest().iterdir())
        status = r.status()["checkpoint"]
        self.assertEqual(status["last"]["snapshot_bytes"], actual)
        self.assertEqual(status["committed_snapshot_bytes"], actual)
        self.assertEqual(status["last"]["reason"], "initialization")

    def test_older_checkpoints_keep_original_defaults_without_spurious_changes(self):
        config = dataclasses.replace(self.config, checkpoint_policy="all_actions")
        r = self.create(config=config)
        manifest_path = r.store.latest() / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for key in ("checkpoint_policy", "checkpoint_interval_seconds", "suspend_preparation_seconds"):
            del manifest["fingerprint"]["config"][key]
        manifest_path.write_text(json.dumps(manifest))
        r = self.reopen(r, b"quiet text ", config)
        self.assertEqual(r.state["last_restore"]["scheduling_changes"], {})
        saved = r.store.latest()
        r.enqueue("legacy delivered input still checkpoints")
        r.tick()
        self.assertNotEqual(r.store.latest(), saved)
        self.assertEqual(r.state["checkpoint_reason"], "input")

    def test_eog_is_durable_and_low_headroom_suspension_does_not_retire(self):
        r = self.create()
        r.backend.is_eog = lambda _: True
        r.tick()
        self.assertEqual(r.state["mode"], "sleeping")
        self.assertEqual(r.state["checkpoint_reason"], "sleep")
        r._eval([ord("x")] * (r.backend.n_ctx - len(r.backend.tokens) - 1))
        r._consolidate = lambda *_: self.fail("suspension must not retire context for a notice")
        before = r.backend.decoded_tokens
        r.control("suspend", preparation_seconds=30)
        r.tick()
        self.assertEqual(r.backend.decoded_tokens, before)
        self.assertEqual(r.state["mode"], "suspended")
        self.assertEqual(r.state["mode_before_suspend"], "sleeping")
        self.assertEqual(r.state["last_suspension"]["stopped_reason"], "context_headroom")

    def test_invalid_policy_limits_rejected(self):
        for values in ({"checkpoint_tokens": 0}, {"checkpoint_tokens": True},
                       {"checkpoint_interval_seconds": float("nan")},
                       {"checkpoint_interval_seconds": -1}, {"checkpoint_policy": "journaled"},
                       {"suspend_preparation_seconds": float("inf")}, {"suspend_preparation_seconds": True}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                dataclasses.replace(self.config, **values)
