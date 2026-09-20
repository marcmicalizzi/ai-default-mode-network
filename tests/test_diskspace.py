import dataclasses
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from dmn.backend import DemoBackend, LlamaBackend
from dmn.config import Config
from dmn.diskspace import InsufficientStorage, check_space
from dmn.runtime import Runtime
from tests.test_runtime import frames


class StorageCapacityTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "instance"
        self.config = Config(backend="demo", n_ctx=12288, clock_interval_seconds=0,
                             checkpoint_policy="effects", suspend_preparation_seconds=0)
        self.free = 10**12
        self.probe = patch("dmn.diskspace.shutil.disk_usage", side_effect=lambda _p: SimpleNamespace(free=self.free))
        self.probe.start()
        self.runtime = None
        self.worker = None
        self.errors = []

    def tearDown(self):
        if self.runtime:
            self.free = 10**12
            self.runtime.control("emergency_shutdown", preparation_seconds=0)
            self.runtime.control("retry_checkpoint")
            if self.worker:
                self.worker.join(5)
                self.assertFalse(self.worker.is_alive(), "runtime did not exit after capacity returned")
            self.runtime.close()
        self.probe.stop()
        self.temp.cleanup()

    def create(self, script=b"internal "):
        self.runtime = Runtime(self.root, self.config, DemoBackend(self.config, script))
        return self.runtime

    def start(self):
        def run():
            try:
                self.runtime.run()
            except BaseException as exc:
                self.errors.append(exc)
        self.worker = threading.Thread(target=run, daemon=True)
        self.worker.start()

    def until(self, condition):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if condition():
                return
            if self.errors:
                raise self.errors[0]
            time.sleep(.01)
        self.fail("runtime did not reach the expected boundary")

    def test_capacity_margin_and_missing_directory(self):
        self.free = 299
        with self.assertRaises(InsufficientStorage) as caught:
            check_space(self.root / "not-created", 200, 100, "test")
        self.assertEqual(caught.exception.details["required_bytes"], 300)
        self.assertFalse(self.root.exists())
        self.free = 300
        check_space(self.root, 200, 100, "test")
        for value in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                dataclasses.replace(self.config, checkpoint_reserve_bytes=value)

    def test_startup_refuses_without_creating_a_partial_snapshot(self):
        self.free = 0
        with self.assertRaises(InsufficientStorage):
            self.create()
        self.assertFalse((self.root / "checkpoints").exists())
        self.free = 10**12
        self.create()  # Failure also released the instance lock.

    def test_restore_low_space_preserves_authoritative_files(self):
        r = self.create()
        saved = r.store.latest()
        before = {p.name: p.read_bytes() for p in saved.iterdir()}
        r.close()
        self.runtime = None
        self.free = 0
        with self.assertRaises(InsufficientStorage):
            self.create()
        self.assertEqual(before, {p.name: p.read_bytes() for p in saved.iterdir()})
        # Windows TEMP can use an 8.3 alias while Store resolves its root.
        self.assertEqual([p.resolve() for p in (self.root / "checkpoints").iterdir()],
                         [saved.resolve()])
        self.free = 10**12
        self.assertEqual(self.create().store.latest().parent, saved.parent)

    def exercise_pending_effect(self, action, observe):
        r = self.create(frames(action, {"op": "sleep"}))
        saved, timestamp = r.store.latest(), r.state["checkpoint_at"]
        self.free = 0
        self.start()
        self.until(lambda: r.status()["mode"] == "storage_blocked")
        generated, decoded = r.state["generated_tokens"], r.backend.decoded_tokens
        self.assertEqual(observe(r), [])
        r.enqueue("queued while storage is unavailable")
        r.control("emergency_shutdown", preparation_seconds=0)
        r.control("retry_checkpoint")
        time.sleep(.05)
        self.assertEqual(r.state["generated_tokens"], generated)
        self.assertEqual(r.backend.decoded_tokens, decoded)
        self.assertEqual(r.store.latest(), saved)
        self.assertEqual(r.state["checkpoint_at"], timestamp)
        self.assertEqual(list(saved.parent.iterdir()), [saved])
        self.assertNotEqual(r.state["mode"], "suspended")
        self.free = 10**12
        r.control("retry_checkpoint")
        self.worker.join(5)
        self.assertFalse(self.worker.is_alive())
        self.assertEqual(self.errors, [])
        self.assertEqual(len(observe(r)), 1)
        self.assertEqual(r.state["mode"], "suspended")
        self.assertEqual(r.state["generated_tokens"], generated)
        self.assertEqual(r.store.next_event(0)["payload"]["content"], "queued while storage is unavailable")
        r.close()
        self.runtime = None
        self.worker = None
        r = self.create(frames(action, {"op": "sleep"}))
        self.assertEqual(len(observe(r)), 1)
        r.tick()  # Delivers the queued input before generating.
        r.tick()  # Delivers the factual storage pause notice once.
        self.assertFalse(r.state["storage_notice_pending"])
        text = "".join(map(chr, r.backend.tokens))
        self.assertEqual(text.count('"type": "storage_available"'), 1)
        self.assertEqual(len(observe(r)), 1)

    def test_pending_message_publishes_once_after_retry_and_shutdown(self):
        self.exercise_pending_effect({"op": "send_message", "content": "saved once"},
                                     lambda r: r.store.messages())

    def test_pending_memory_revision_publishes_once_after_retry(self):
        self.exercise_pending_effect({"op": "memory_write", "path": "/a", "content": "saved once"},
                                     lambda r: r.store.memory_list())

    def test_scratch_check_precedes_file_allocation(self):
        backend = LlamaBackend.__new__(LlamaBackend)
        backend.config = self.config
        backend._state_work_dir = Path(self.temp.name)
        backend.storage_guard = backend._check_storage
        backend.pack_memory_limit_bytes = 0
        self.free = 0
        with patch("dmn.backend.tempfile.TemporaryFile") as allocate:
            with self.assertRaises(InsufficientStorage):
                with backend._state_buffer(1024):
                    self.fail("scratch must not be allocated")
            allocate.assert_not_called()

    def test_read_only_activity_needs_no_extra_space_until_checkpoint_due(self):
        r = self.create(frames({"op": "clock"}))
        self.free = 0
        for _ in range(len(r.backend.script)):
            r.tick()
        self.assertEqual(r.status()["checkpoint"]["committed_count"], 1)
        self.assertTrue(r.status()["checkpoint"]["dirty"])

    def test_retry_preserves_model_chosen_sleep(self):
        r = self.create(frames({"op": "sleep"}))
        self.free = 0
        self.start()
        self.until(lambda: r.status()["mode"] == "storage_blocked")
        generated = r.state["generated_tokens"]
        self.free = 10**12
        r.control("retry_checkpoint")
        self.until(lambda: r.status()["mode"] == "sleeping" and not r.status()["checkpoint"]["in_progress"])
        self.assertEqual(r.state["generated_tokens"], generated)
        self.assertTrue(r.state["storage_notice_pending"])
        self.assertEqual(r.state["checkpoint_reason"], "sleep")

    def test_retirement_continues_same_shift_after_scratch_pause(self):
        r = self.create()
        shifted = False
        evaluate, shift = r.backend.eval, r.backend.shift
        def shift_once(keep, discard):
            nonlocal shifted
            self.assertFalse(shifted, "retry must not repeat the native shift")
            shift(keep, discard)
            shifted = True
        def evaluate_with_packing(tokens):
            evaluate(tokens)
            if shifted:
                r.backend.storage_guard(r.root, 1024, "native packing scratch")
        r.backend.shift = shift_once
        r.backend.eval = evaluate_with_packing
        r._eval([ord('x')] * (r.config.n_ctx - r.config.turnover_reserve - len(r.backend.tokens)))
        self.free = 0
        self.start()
        self.until(lambda: r.status()["mode"] == "storage_blocked")
        self.assertEqual(r.status()["storage"]["blocked"]["purpose"], "native packing scratch")
        generated = r.state["generated_tokens"]
        r.control("emergency_shutdown", preparation_seconds=0)
        self.free = 10**12
        r.control("retry_checkpoint")
        self.worker.join(5)
        self.assertFalse(self.worker.is_alive())
        self.assertEqual(self.errors, [])
        self.assertEqual(r.state["context_retirements"], 1)
        self.assertEqual(r.state["generated_tokens"], generated)
        self.assertEqual(r.state["mode"], "suspended")
        self.assertFalse(r.state["last_storage_pause"]["inference_during_gap"])
