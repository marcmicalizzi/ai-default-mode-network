import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

from dmn.backend import DemoBackend, sha256_file
from dmn.config import Config
from dmn.migration import active_context_projection, prepare_bundle
from dmn.runtime import Runtime


class RecoveryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = Config(backend="demo", clock_interval_seconds=0, preparation_tokens=8)
        self.script = (b'\n<dmn_action>{"op":"memory_write","path":"/keep","content":"retained"}</dmn_action>'
                       b'\n<dmn_action>{"op":"send_message","content":"published once"}</dmn_action>'
                       b'\n<dmn_action>{"op":"sleep"}</dmn_action>\n')
        r = self.open()
        for _ in range(len(self.script)):
            r.tick()
        r.enqueue("pending during recovery")
        self.saved = r.store.latest()
        self.expected_tokens = r.backend.tokens.copy()
        self.identity = r.state["instance_id"]
        r.close()

    def tearDown(self):
        self.temp.cleanup()

    def open(self, policy="strict", config=None, backend=None):
        config = config or self.config
        return Runtime(self.root, config, backend or DemoBackend(config, self.script), kv_recovery=policy)

    def mark_native_cache_missing(self):
        manifest_path = self.saved / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        missing = self.saved / "state.bin"
        missing.write_bytes(b"test cache")
        manifest["files"]["state.bin"] = sha256_file(missing)
        manifest_path.write_text(json.dumps(manifest))
        missing.unlink()

    def test_strict_rejects_missing_cache_but_explicit_fallback_retains_effects(self):
        self.mark_native_cache_missing()
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.open()
        r = self.open("fallback")
        try:
            self.assertEqual(r.state["last_restore"]["method"], "retained_token_reconstruction")
            self.assertEqual(r.state["last_restore"]["prompt_tokens_reevaluated"], len(self.expected_tokens))
            self.assertEqual(r.state["instance_id"], self.identity)
            self.assertEqual(r.backend.tokens[:len(self.expected_tokens)], self.expected_tokens)
            self.assertEqual(r.store.memory_read("/keep"), "retained")
            self.assertEqual(len(r.store.messages()), 1)
            self.assertEqual(r.state["event_cursor"], 0)
            r.tick()
            self.assertEqual(r.state["event_cursor"], 1)
            self.assertEqual(len(r.store.messages()), 1)
        finally:
            r.close()

    def test_rebuild_does_not_even_attempt_native_loader(self):
        backend = DemoBackend(self.config, self.script)
        backend.load = lambda _: (_ for _ in ()).throw(AssertionError("loader must not be called"))
        r = self.open("rebuild", backend=backend)
        try:
            self.assertEqual(r.state["reconstructions"], 1)
        finally:
            r.close()

    def test_fallback_uses_native_path_if_available(self):
        backend = DemoBackend(self.config, self.script)
        backend.rebuild = lambda _: (_ for _ in ()).throw(AssertionError("healthy state must be loaded"))
        r = self.open("fallback", backend=backend)
        try:
            self.assertEqual(r.state["last_restore"]["method"], "demo_restore")
            self.assertEqual(r.state["last_restore"]["prompt_tokens_reevaluated"], 0)
        finally:
            r.close()

    def test_fallback_after_loader_error_rebuilds(self):
        backend = DemoBackend(self.config, self.script)
        backend.load = lambda _: (_ for _ in ()).throw(RuntimeError("native error"))
        r = self.open("fallback", backend=backend)
        try:
            self.assertIn("native error", r.state["last_restore"]["reason"])
        finally:
            r.close()

    def test_corrupt_recovery_sidecar_is_never_used(self):
        (self.saved / "engine.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.open("rebuild")

    def test_reconstruction_rejects_different_model_or_sampler(self):
        with self.assertRaisesRegex(ValueError, "identity"):
            self.open("rebuild", backend=DemoBackend(self.config, b"different fixture"))
        with self.assertRaisesRegex(ValueError, "sampler"):
            self.open("rebuild", config=dataclasses.replace(self.config, temperature=0.7))

    def test_hardware_placement_can_change_only_with_reconstruction(self):
        config = dataclasses.replace(self.config, n_ctx=10000, n_threads=2)
        with self.assertRaisesRegex(ValueError, "environment differs"):
            self.open(config=config)
        r = self.open("fallback", config=config)
        try:
            self.assertEqual(r.state["last_restore"]["method"], "retained_token_reconstruction")
        finally:
            r.close()

    def test_pacing_changes_preserve_native_restore_path(self):
        config = dataclasses.replace(self.config, token_delay_seconds=0.5, checkpoint_tokens=1024)
        backend = DemoBackend(config, self.script)
        backend.rebuild = lambda _: (_ for _ in ()).throw(AssertionError("pacing must not rebuild context"))
        r = self.open(config=config, backend=backend)
        try:
            self.assertEqual(r.state["last_restore"]["method"], "demo_restore")
            self.assertEqual(r.state["last_restore"]["scheduling_changes"]["token_delay_seconds"]["current"], 0.5)
            self.assertEqual(r.state["last_restore"]["prompt_tokens_reevaluated"], 0)
        finally:
            r.close()


class CompactionProjectionTest(unittest.TestCase):
    def test_latest_summary_keeps_boundary_and_does_not_change_archive(self):
        branch = [{"id": "s", "role": "system", "content": "identity"},
                  {"id": "a", "role": "user", "content": "first"},
                  {"id": "b", "role": "assistant", "contextSummary": "old summary"},
                  {"id": "c", "role": "user", "context_summary": "latest summary"},
                  {"id": "d", "role": "assistant", "content": "recent"}]
        original = json.dumps(branch)
        result = active_context_projection(branch)
        self.assertEqual(result["summary"], "latest summary")
        self.assertEqual(result["retained_message_ids"], ["c", "d"])
        self.assertEqual(result["archived_message_ids"], ["a", "b"])
        self.assertEqual(result["system_messages"][0]["content"], "identity")
        self.assertFalse(result["is_final_provider_request"])
        self.assertEqual(json.dumps(branch), original)

    def test_disabled_compaction_uses_full_history_even_with_saved_summary(self):
        branch = [{"id": "a"}, {"id": "b", "contextSummary": "saved"}, {"id": "c"}]
        result = active_context_projection(branch, False)
        self.assertEqual(result["retained_message_ids"], ["a", "b", "c"])
        self.assertIsNone(result["summary"])

    def test_bundle_preserves_final_provider_request_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            export = root / "export.json"
            export.write_text('{"messages":[{"id":"a","role":"user","content":"old"},{"id":"b","role":"assistant","content":"recent","contextSummary":"saved summary"}]}')
            request = root / "request.json"
            request.write_text('{"messages": [{"role":"system","content":"final injected system"}], "model":"test"}')
            manifest = prepare_bundle(export, root / "bundle", request_path=request)
            self.assertTrue(manifest["provider_request_preserved"])
            self.assertEqual((root / "bundle/provider-request.json").read_bytes(), request.read_bytes())
            active = json.loads((root / "bundle/active-context.json").read_text())
            self.assertEqual(active["summary"], "saved summary")
