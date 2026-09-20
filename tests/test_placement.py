import dataclasses
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from dmn.backend import sha256_file
from dmn.config import Config
from dmn.recovery import restore_checkpoint
from dmn.runtime import Runtime


class PlacementPolicyTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.saved = Path(self.temp.name)
        self.config = Config(n_threads=6, n_gpu_layers=24)
        self.fingerprint = {"kind": "native_llama_kv", "model_sha256": "model",
                            "native_binaries": {"llama": "pinned"}, "config": self.config.to_dict()}
        payloads = {"runtime.json": '{"schema":1,"instance_id":"unchanged"}', "engine.json": '{}',
                    "state.bin": 'opaque native bytes', "logits.npy": 'opaque logits'}
        for name, data in payloads.items():
            (self.saved / name).write_text(data)
        self.manifest = {"fingerprint": self.fingerprint,
                         "files": {name: sha256_file(self.saved / name) for name in payloads}}
        (self.saved / "manifest.json").write_text(json.dumps(self.manifest))

    def backend(self, **changes):
        config = dataclasses.replace(self.config, **changes)
        return SimpleNamespace(kind="native_llama_kv", fingerprint={**self.fingerprint, "config": config.to_dict()},
            load=Mock(return_value={"prompt_tokens_reevaluated": 0, "decode_calls_during_load": 0}),
            rebuild=Mock(side_effect=AssertionError("must never reconstruct")),
            verify_loaded_snapshot=Mock(return_value={"serialized_native_state_bytes_equal": True}))

    def test_explicit_placement_change_verifies_state_and_records_changes(self):
        backend = self.backend(n_threads=12, n_gpu_layers=30)
        with self.assertRaisesRegex(ValueError, "environment differs"):
            restore_checkpoint(backend, self.saved)
        backend.load.assert_not_called()
        state, evidence = restore_checkpoint(backend, self.saved, allow_placement_change=True)
        self.assertEqual(state["instance_id"], "unchanged")
        self.assertEqual(evidence["method"], "native_restore")
        self.assertEqual(evidence["placement_changes"], {
            "n_threads": {"previous": 6, "current": 12}, "n_gpu_layers": {"previous": 24, "current": 30}})
        backend.verify_loaded_snapshot.assert_called_once_with(self.saved, self.manifest["files"]["state.bin"])
        backend.rebuild.assert_not_called()

    def test_unrelated_changes_are_rejected_before_loading(self):
        for changes in ({"n_ctx": 16384}, {"n_batch": 128}, {"swa_full": False},
                        {"temperature": .4}, {"type_k": "q8_0"}, {"system_prompt": "replacement"},
                        {"offload_kqv": False}, {"flash_attn": True}, {"pack_checkpoints": True}):
            with self.subTest(changes=changes):
                backend = self.backend(n_threads=12, **changes)
                with self.assertRaisesRegex(ValueError, "only n_threads and n_gpu_layers"):
                    restore_checkpoint(backend, self.saved, allow_placement_change=True)
                backend.load.assert_not_called()
        backend = self.backend(n_threads=12)
        backend.fingerprint["native_binaries"] = {"llama": "different"}
        with self.assertRaisesRegex(ValueError, "other inference settings"):
            restore_checkpoint(backend, self.saved, allow_placement_change=True)
        backend.load.assert_not_called()

    def test_corruption_and_round_trip_failure_cannot_fall_back(self):
        backend = self.backend(n_threads=12)
        for policy in ("fallback", "rebuild"):
            with self.assertRaisesRegex(ValueError, "strict recovery"):
                restore_checkpoint(backend, self.saved, policy, allow_placement_change=True)
        backend.verify_loaded_snapshot.side_effect = RuntimeError("bytes differ")
        with self.assertRaisesRegex(RuntimeError, "bytes differ"):
            restore_checkpoint(backend, self.saved, allow_placement_change=True)
        backend.rebuild.assert_not_called()
        backend.load.reset_mock()
        (self.saved / "state.bin").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "integrity"):
            restore_checkpoint(backend, self.saved, allow_placement_change=True)
        backend.load.assert_not_called()

    def test_cannot_claim_original_environment_or_reconstruct_with_opt_in(self):
        for kwargs in ({"resume_condition": "original_environment"}, {"kv_recovery": "fallback"}):
            with self.assertRaisesRegex(ValueError, "placement changes require strict"):
                Runtime(self.saved, self.config, allow_placement_change=True, **kwargs)

    def test_replay_is_rejected_before_verification(self):
        backend = self.backend(n_threads=12)
        backend.load.return_value["decode_calls_during_load"] = 1
        with self.assertRaisesRegex(RuntimeError, "prompt evaluation"):
            restore_checkpoint(backend, self.saved, allow_placement_change=True)
        backend.verify_loaded_snapshot.assert_not_called()


@unittest.skipUnless(os.environ.get("DMN_TEST_MODEL"), "set DMN_TEST_MODEL for native placement verification")
class NativePlacementTest(unittest.TestCase):
    def test_thread_change_preserves_native_bytes_without_replay_and_restart_after_retirement(self):
        import numpy as np
        from dmn.backend import LlamaBackend
        config = Config(model_path=str(Path(os.environ["DMN_TEST_MODEL"]).resolve()),
                        n_ctx=2048, n_threads=1, n_gpu_layers=0, pack_checkpoints=True)
        backend = LlamaBackend(config)
        try:
            with tempfile.TemporaryDirectory() as folder:
                saved = Path(folder)
                backend.eval(backend.tokenize("A disposable placement check. " * 12, initial=True))
                backend.save(saved)
                (saved / "runtime.json").write_text('{"schema":1}')
                files = {name: sha256_file(saved / name) for name in
                         ("state.bin", "engine.json", "logits.npy", "runtime.json")}
                (saved / "manifest.json").write_text(json.dumps({"fingerprint": backend.fingerprint, "files": files}))
                original_tokens, original_rng = backend.tokens.copy(), backend.rng.getstate()
                backend.close()
                backend = LlamaBackend(dataclasses.replace(config, n_threads=2))
                evaluate = backend.eval
                backend.eval = Mock(side_effect=AssertionError("restore must never evaluate tokens"))
                _, evidence = restore_checkpoint(backend, saved, allow_placement_change=True)
                self.assertTrue(evidence["serialized_native_state_bytes_equal"])
                self.assertFalse(evidence["future_continuation_bit_identical_guaranteed"])
                self.assertEqual(backend.tokens, original_tokens)
                self.assertEqual(backend.rng.getstate(), original_rng)
                self.assertEqual(backend.decode_calls, 0)
                backend.eval.assert_not_called()
                with self.assertRaisesRegex(RuntimeError, "byte for byte"):
                    backend.verify_loaded_snapshot(saved, "0" * 64)
                backend.eval = evaluate
                backend.shift(4, 16)
                backend.eval(backend.tokenize("\nOlder context retired.\n"))
                backend.save(saved)
                expected = []
                for _ in range(8):
                    token = backend.sample()
                    backend.eval([token])
                    expected.append((token, backend.logits.copy()))
                backend.close()
                backend = LlamaBackend(dataclasses.replace(config, n_threads=2))
                backend.load(saved)
                for token, logits in expected:
                    self.assertEqual(backend.sample(), token)
                    backend.eval([token])
                    np.testing.assert_array_equal(backend.logits, logits)
                self.assertFalse(list(saved.parent.glob(".dmn-placement-*")))
        finally:
            backend.close()
