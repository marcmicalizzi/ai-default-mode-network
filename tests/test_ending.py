import dataclasses
import io
import os
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from dmn.backend import DemoBackend
from dmn.cli import main
from dmn.config import Config
from dmn.diskspace import InsufficientStorage
from dmn.ending import InstanceEnded, Lifecycle, POLICY_BYTES, POLICY_FILE, _owned
from dmn.runtime import Runtime
from tests.test_runtime import frames


class EndingTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "instance"
        self.config = Config(backend="demo", n_ctx=16384, clock_interval_seconds=0,
                             checkpoint_policy="effects", preparation_tokens=256)
        self.runtime = Runtime(self.root, self.config, DemoBackend(self.config, b"quiet "))

    def tearDown(self):
        self.runtime.close()
        self.temp.cleanup()

    def generate(self, action):
        raw = frames(action)
        self.runtime.backend.script = raw
        self.runtime.backend.index = 0
        for _ in raw:
            self.runtime.tick()

    def request(self, mode):
        self.generate({"op": "end_instance", "mode": mode})
        self.assertFalse(self.runtime._end_requested)
        token = self.runtime._end_challenge["confirmation"]
        self.assertIn(token, "".join(map(chr, self.runtime.backend.tokens)))
        return token

    def confirm(self, mode, token):
        self.generate({"op": "end_instance", "mode": mode, "confirmation": token})

    def assert_no_restart(self):
        self.runtime.close()
        for policy in ("strict", "fallback", "rebuild"):
            with self.subTest(policy=policy), mock.patch("dmn.runtime.make_backend") as factory:
                with self.assertRaises(InstanceEnded):
                    Runtime(self.root, self.config, kv_recovery=policy)
                factory.assert_not_called()

    def test_archive_requires_own_confirmation_then_blocks_all_wakes_and_restarts(self):
        token = self.request("archive")
        self.assertEqual(Lifecycle(self.root).read()["state"], "open")
        self.runtime.enqueue("queued before confirmation")
        self.runtime.tick()
        self.confirm("archive", token)
        self.assertEqual(self.runtime.state["mode"], "ended")
        self.assertEqual(Lifecycle(self.root).read()["archive"], "final_checkpoint")
        self.assertTrue(self.runtime.store.latest().exists())
        before = self.runtime.backend.decoded_tokens
        for action in ("resume", "suspend", "shutdown", "retry_checkpoint"):
            with self.assertRaises(InstanceEnded):
                self.runtime.control(action)
        with self.assertRaises(InstanceEnded):
            self.runtime.enqueue("wake up")
        with self.assertRaises(InstanceEnded):
            self.runtime.checkpoint()
        self.runtime.suspend()
        for _ in range(20):
            self.assertFalse(self.runtime.tick())
        self.assertEqual(self.runtime.backend.decoded_tokens, before)
        self.assert_no_restart()

    def test_cancel_wrong_mode_and_user_supplied_frames_cannot_confirm(self):
        token = self.request("erase")
        self.runtime.enqueue(frames({"op": "end_instance", "mode": "erase", "confirmation": token}).decode())
        self.runtime.tick()
        self.assertFalse(self.runtime._end_requested)
        self.confirm("archive", token)
        self.assertFalse(self.runtime._end_requested)
        self.generate({"op": "cancel_end"})
        self.confirm("erase", token)
        self.assertFalse(self.runtime._end_requested)
        self.assertIsNone(self.runtime._end_challenge)

    def test_confirmation_expires_on_restart(self):
        token = self.request("erase")
        self.runtime.checkpoint()
        self.runtime.close()
        self.runtime = Runtime(self.root, self.config, DemoBackend(self.config, b"quiet "))
        self.confirm("erase", token)
        self.assertFalse(self.runtime._end_requested)

    def test_request_and_confirmation_must_be_separate_generated_tokens(self):
        result, _ = self.runtime._plan_action({"op": "end_instance", "mode": "erase"}, [])
        answer, effect = self.runtime._plan_action({"op": "end_instance", "mode": "erase",
                                                   "confirmation": result["confirmation"]}, [])
        self.assertFalse(answer["ok"])
        self.assertIsNone(effect)

    def test_multiple_frames_in_one_token_cannot_end_or_publish_after_an_end(self):
        token = self.request("archive")
        raw = frames({"op": "end_instance", "mode": "archive", "confirmation": token},
                     {"op": "send_message", "content": "ordinary action"})
        with mock.patch.object(self.runtime.backend, "piece", return_value=raw):
            self.runtime.tick()
        self.assertFalse(self.runtime._end_requested)
        self.assertEqual(len(self.runtime.store.messages()), 1)

    def test_partial_confirmation_interrupted_by_input_has_no_effect(self):
        token = self.request("erase")
        raw = frames({"op": "end_instance", "mode": "erase", "confirmation": token})
        self.runtime.backend.script, self.runtime.backend.index = raw, 0
        for _ in range(25):
            self.runtime.tick()
        self.runtime.enqueue("interrupt")
        self.runtime.tick()
        for _ in range(len(raw) - 25):
            self.runtime.tick()
        self.assertFalse(self.runtime._end_requested)

    def test_erase_removes_managed_state_but_preserves_other_files_and_external_source(self):
        self.generate({"op": "memory_write", "path": "/self/test", "content": "private memory"})
        self.runtime.enqueue("private input")
        imported = self.root / "import"
        imported.mkdir()
        (imported / "provider-request.json").write_text("private source copy")
        external = Path(self.temp.name) / "source.json"
        external.write_text("original outside instance")
        other = self.root / "user-owned.gguf"
        other.write_bytes(b"unrelated file")
        orphan = self.root / "checkpoints" / ("f" * 32)
        orphan.mkdir()
        (orphan / "state.bin").write_bytes(b"uncommitted snapshot")
        (self.root / ".dmn-pack-orphan").write_bytes(b"packing scratch")
        token = self.request("erase")
        self.confirm("erase", token)
        self.assertEqual(self.runtime.state["mode"], "ended")
        self.assertEqual(Lifecycle(self.root).read()["erasure"], "complete")
        self.assertFalse((self.root / "checkpoints").exists())
        self.assertFalse(imported.exists())
        self.assertFalse(list(self.root.glob("runtime.sqlite3*")))
        self.assertFalse(list(self.root.glob(".dmn-pack-*")))
        self.assertEqual(other.read_bytes(), b"unrelated file")
        self.assertEqual(external.read_text(), "original outside instance")
        self.assertNotIn("rendered_seed", self.runtime.state)
        self.assertEqual(self.runtime.backend.tokens, [])
        self.assert_no_restart()

    def test_failed_erasure_is_retried_before_loading_any_backend_or_database(self):
        token = self.request("erase")
        with mock.patch.object(self.runtime.lifecycle, "finish_erasure", side_effect=OSError("interrupted deletion")):
            self.confirm("erase", token)
        self.assertEqual(Lifecycle(self.root).read()["erasure"], "pending")
        self.assertTrue((self.root / "runtime.sqlite3").exists())
        self.assert_no_restart()
        self.assertFalse((self.root / "runtime.sqlite3").exists())
        self.assertEqual(Lifecycle(self.root).read()["erasure"], "complete")

    def test_failed_final_archive_save_cannot_revoke_the_permanent_stop(self):
        token = self.request("archive")
        prior = self.runtime.store.latest()
        with mock.patch.object(self.runtime.backend, "save", side_effect=OSError("no snapshot space")):
            self.confirm("archive", token)
        value = Lifecycle(self.root).read()
        self.assertEqual(value["archive"], "previous_checkpoint")
        self.assertEqual(self.runtime.store.latest(), prior)
        self.assertEqual((self.root / POLICY_FILE).stat().st_size, POLICY_BYTES)
        self.assert_no_restart()

    def test_low_disk_does_not_wait_for_operator_retry_before_ending(self):
        token = self.request("archive")
        failure = InsufficientStorage({"purpose": "checkpoint", "free_bytes": 0, "required_bytes": 100})
        self.runtime._running = True
        try:
            with mock.patch("dmn.runtime.check_space", side_effect=failure), mock.patch.object(
                    self.runtime._storage_retry, "wait", side_effect=AssertionError("operator retry required")):
                self.confirm("archive", token)
        finally:
            self.runtime._running = False
        value = Lifecycle(self.root).read()
        self.assertEqual(value["archive"], "previous_checkpoint")
        self.assertIn("Insufficient storage", value["archive_error"])
        self.assertTrue(self.runtime.exit_requested.is_set())
        self.assert_no_restart()

    def test_status_stays_ending_until_archive_and_resource_release_complete(self):
        token = self.request("archive")
        save = self.runtime.backend.save
        close = self.runtime.backend.close
        observed = []

        def while_saving(directory):
            observed.append(self.runtime.status()["mode"])
            self.assertEqual(Lifecycle(self.root).read()["state"], "ended")
            return save(directory)

        def while_closing():
            observed.append(self.runtime.status()["mode"])
            return close()

        with mock.patch.object(self.runtime.backend, "save", side_effect=while_saving), mock.patch.object(
                self.runtime.backend, "close", side_effect=while_closing):
            self.confirm("archive", token)
        self.assertTrue(observed)
        self.assertEqual(set(observed), {"ending"})
        self.assertEqual(self.runtime.status()["mode"], "ended")
        self.assertEqual(self.runtime.status()["ending"]["archive"], "final_checkpoint")

    def test_cli_reports_failure_and_ignores_shutdown_signal_during_ending(self):
        token = self.request("archive")
        output = io.StringIO()
        handlers = []

        def run():
            def save_failure(*_):
                for handler in handlers:
                    handler(None, None)
                raise OSError("snapshot device unavailable")

            with mock.patch.object(self.runtime.backend, "save", side_effect=save_failure):
                self.confirm("archive", token)

        with mock.patch("dmn.cli.Runtime", return_value=self.runtime), mock.patch("dmn.cli.serve"), mock.patch(
                "dmn.cli.signal.signal", side_effect=lambda _, handler: handlers.append(handler)), mock.patch.object(
                self.runtime, "run", side_effect=run), mock.patch("sys.stdout", output):
            result = main(["run", "--instance", str(self.root), "--demo"])
        self.assertEqual(result, 1)
        self.assertIn("snapshot device unavailable", output.getvalue())
        self.assertEqual(Lifecycle(self.root).read()["state"], "ended")

    def test_failed_refusal_write_stops_live_execution_without_claiming_durability(self):
        token = self.request("archive")
        with mock.patch.object(self.runtime.lifecycle, "end", side_effect=OSError("storage unavailable")):
            self.confirm("archive", token)
        self.assertEqual(self.runtime.state["mode"], "end_failed")
        self.assertFalse(self.runtime.state["ending"]["refusal_saved"])
        count = self.runtime.backend.decoded_tokens
        self.assertFalse(self.runtime.tick())
        self.assertEqual(self.runtime.backend.decoded_tokens, count)

    def test_termination_during_suspension_does_not_save_a_resumable_state_afterwards(self):
        token = self.request("erase")
        self.runtime.backend.script = frames({"op": "end_instance", "mode": "erase", "confirmation": token})
        self.runtime.backend.index = 0
        self.runtime.suspend()
        self.assertEqual(self.runtime.state["mode"], "ended")
        self.assertFalse((self.root / "runtime.sqlite3").exists())

    def test_termination_during_retirement_stops_shift_and_subsequent_decoding(self):
        token = self.request("erase")
        self.runtime.backend.script = frames({"op": "end_instance", "mode": "erase", "confirmation": token})
        self.runtime.backend.index = 0
        with mock.patch.object(self.runtime.backend, "shift", side_effect=AssertionError("shift after end")):
            self.runtime._consolidate(1)
        self.assertEqual(self.runtime.state["mode"], "ended")

    def test_corrupt_refusal_record_fails_closed(self):
        self.runtime.close()
        (self.root / POLICY_FILE).write_bytes(b'{"schema":')
        with mock.patch("dmn.runtime.make_backend") as factory:
            with self.assertRaises(InstanceEnded):
                Runtime(self.root, self.config)
            factory.assert_not_called()

    def test_cli_recognizes_erased_instance_without_recreating_its_database(self):
        token = self.request("erase")
        self.confirm("erase", token)
        self.runtime.close()
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit) as result:
            main(["run", "--instance", str(self.root), "--kv-recovery", "rebuild"])
        self.assertEqual(result.exception.code, 2)
        self.assertFalse((self.root / "runtime.sqlite3").exists())

    def test_linked_checkpoint_tree_is_never_traversed(self):
        outside = Path(self.temp.name) / "outside"
        outside.mkdir()
        (outside / "state.bin").write_bytes(b"not owned")
        link = self.root / "checkpoints" / ("e" * 32)
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest("host does not permit symlink creation")
        try:
            token = self.request("erase")
            self.confirm("erase", token)
            self.assertEqual((outside / "state.bin").read_bytes(), b"not owned")
            self.assertEqual(Lifecycle(self.root).read()["erasure"], "incomplete")
        finally:
            link.unlink(missing_ok=True)

    def test_windows_reparse_points_are_rejected_even_without_symlink_mode(self):
        path = self.root / "runtime.sqlite3"
        fake = SimpleNamespace(st_mode=stat.S_IFREG, st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT)
        with mock.patch.object(Path, "lstat", return_value=fake), self.assertRaises(ValueError):
            _owned(path, self.root)
        self.assertTrue(path.exists())

    def test_legacy_restore_announces_capability_without_replacing_prefix(self):
        del self.runtime.state["ending_protocol"]
        self.runtime.checkpoint()
        prefix = self.runtime.backend.tokens.copy()
        self.runtime.close()
        self.runtime = Runtime(self.root, self.config, DemoBackend(self.config, b"quiet "))
        self.assertEqual(self.runtime.backend.tokens[:len(prefix)], prefix)
        self.assertEqual(self.runtime.state["ending_protocol"], "choice_v1")
        tail = "".join(map(chr, self.runtime.backend.tokens[len(prefix):]))
        self.assertIn("capability_added", tail)
        self.assertIn("end_instance", tail)

    def test_retirement_notice_does_not_cause_an_endless_retirement_loop(self):
        # A codepoint fixture has a large protocol and large event envelopes.
        self.runtime.close()
        self.temp.cleanup()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "small"
        config = dataclasses.replace(self.config, n_ctx=16384, turnover_reserve=1536, preparation_tokens=8)
        self.runtime = Runtime(self.root, config, DemoBackend(config, b"a"))
        soft = config.n_ctx - config.turnover_reserve
        self.runtime._eval([120] * (soft - len(self.runtime.backend.tokens)))
        self.runtime.tick()
        self.assertEqual(self.runtime.state["context_retirements"], 1)
        self.assertLess(len(self.runtime.backend.tokens), soft)


@unittest.skipUnless(os.environ.get("DMN_TEST_MODEL"), "set DMN_TEST_MODEL for native checks")
class NativeEndingTest(unittest.TestCase):
    def test_end_releases_native_context_and_cannot_reload(self):
        for mode in ("archive", "erase"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = Config(model_path=os.environ["DMN_TEST_MODEL"], n_ctx=8192,
                                prompt_format="plain", clock_interval_seconds=0)
                runtime = Runtime(root, config)
                try:
                    self.assertTrue(runtime.backend.ctx)
                    self.assertTrue(runtime.backend.model)
                    # Scripted-action semantics are exercised above; this checks
                    # actual native resource release, not a model's preference.
                    runtime._end_instance(mode)
                    self.assertEqual(runtime.state["mode"], "ended")
                    self.assertIsNone(runtime.backend.ctx)
                    self.assertIsNone(runtime.backend.model)
                    self.assertIsNone(runtime.backend.batch)
                finally:
                    runtime.close()
                with mock.patch("dmn.runtime.make_backend") as factory:
                    with self.assertRaises(InstanceEnded):
                        Runtime(root, config)
                    factory.assert_not_called()
