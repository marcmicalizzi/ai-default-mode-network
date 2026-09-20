import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from dmn.backend import DemoBackend
from dmn.config import Config
from dmn.packaging import package_instance
from dmn.preservation import InstanceHeld, saved_state
from dmn.runtime import Runtime
from tests.test_runtime import frames


class PreservationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "instance"
        self.config = Config(backend="demo", n_ctx=16384, clock_interval_seconds=0,
                             checkpoint_tokens=100000, checkpoint_policy="effects")
        self.r = None

    def tearDown(self):
        if self.r:
            self.r.close()
        self.temp.cleanup()

    def open(self, **kwargs):
        self.r = Runtime(self.root, self.config, DemoBackend(self.config, b"thought "), **kwargs)
        return self.r

    def hold(self, packaging="zip", recovery="remain_held"):
        r = self.open()
        raw = frames({"op": "hold_instance", "condition": "server_ready", "packaging": packaging, "recovery": recovery})
        r.backend.script, r.backend.index = raw, 0
        for _ in raw:
            r.tick()
        hold = r.state["hold"]
        self.assertTrue(r.stopped.is_set())
        with self.assertRaises(InstanceHeld):
            r.control("resume")
        with self.assertRaises(InstanceHeld):
            r.enqueue("wake")
        r.close()
        self.r = None
        return hold

    def test_staged_save_start_requires_first_question_and_samples_nothing(self):
        r = self.open(prepare_only=True)
        self.assertEqual(r.state["mode"], "staged")
        before = r.backend.tokens.copy()
        with self.assertRaisesRegex(ValueError, "first question"):
            r.propose_prompt("A proposal", r.state["agreement"]["revision"])
        with mock.patch.object(r.backend, "sample", side_effect=AssertionError("sampled")):
            r.tick()
        r.close()
        r = self.open()
        self.assertEqual(r.backend.tokens, before)
        with self.assertRaises(ValueError):
            r.control("resume")
        r.close()
        self.r = None
        with self.assertRaisesRegex(ValueError, "first question"):
            self.open(start_staged=True)
        r = self.open()
        r.enqueue("First question")
        r.close()
        r = self.open(start_staged=True)
        with mock.patch.object(r.backend, "sample", side_effect=AssertionError("sample before question")):
            r.tick()
        self.assertEqual(r.state["event_cursor"], 1)
        self.assertEqual(r.state["generated_tokens"], 0)

    def test_hold_blocks_backend_loading_and_requires_matching_release(self):
        hold = self.hold()
        for kwargs in ({}, {"release_hold": "wrong", "resume_condition": "server_ready"},
                       {"release_hold": hold["id"], "resume_condition": "server_ready", "kv_recovery": "fallback"}):
            with mock.patch("dmn.runtime.make_backend", side_effect=AssertionError("loaded")):
                with self.assertRaises(InstanceHeld):
                    Runtime(self.root, self.config, **kwargs)
        r = self.open(release_hold=hold["id"], resume_condition="server_ready")
        self.assertNotIn("hold", r.state)
        self.assertEqual(r.state["last_hold"]["id"], hold["id"])

    def test_failed_native_restore_keeps_hold_and_package_round_trips(self):
        hold = self.hold()
        with mock.patch.object(DemoBackend, "load", side_effect=ValueError("cannot load")):
            with self.assertRaises(ValueError):
                self.open(release_hold=hold["id"], resume_condition="server_ready")
        self.r = None
        self.assertEqual(saved_state(self.root)[0]["hold"], hold)
        output = self.root.parent / "preserved.zip"
        result = package_instance(self.root, output)
        self.assertTrue(result["verified"])
        with zipfile.ZipFile(output) as archive:
            self.assertEqual(archive.read("instance/runtime.sqlite3"), (self.root / "runtime.sqlite3").read_bytes())
            self.assertEqual(json.loads(archive.read("preservation.json"))["hold"], hold)
        self.assertTrue(self.root.exists())

    def test_packaging_choice_is_enforced(self):
        self.hold(packaging="none")
        with self.assertRaises(ValueError):
            package_instance(self.root, self.root.parent / "no.zip")

    def test_tar_round_trip(self):
        self.hold(packaging="tar")
        self.assertTrue(package_instance(self.root, self.root.parent / "preserved.tar")["verified"])
