"""Opt in with DMN_TEST_MODEL; the normal suite needs no model or bindings."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(os.environ.get("DMN_TEST_MODEL"), "set DMN_TEST_MODEL for native process-restart test")
class NativeProcessTest(unittest.TestCase):
    def test_diagnostic_verbosity_preserves_native_state_and_continuation(self):
        from unittest.mock import patch
        import numpy as np
        from dmn.backend import LlamaBackend
        from dmn.config import Config
        config = Config(model_path=str(Path(os.environ["DMN_TEST_MODEL"]).resolve()),
                        n_ctx=4096, n_gpu_layers=0, n_threads=1)
        backend = None
        try:
            with tempfile.TemporaryDirectory() as folder:
                with patch.dict(os.environ, {"DMN_NATIVE_LOG_LEVEL": "debug"}):
                    backend = LlamaBackend(config)
                backend.eval(backend.tokenize("A diagnostic sequence. " * 20, initial=True))
                backend.save(Path(folder))
                fingerprint = backend.fingerprint
                expected = []
                for _ in range(8):
                    token = backend.sample()
                    backend.eval([token])
                    expected.append((token, backend.logits.copy()))
                backend.close()
                with patch.dict(os.environ, {"DMN_NATIVE_LOG_LEVEL": "warning"}):
                    backend = LlamaBackend(config)
                self.assertEqual(backend.fingerprint, fingerprint)
                evidence = backend.load(Path(folder))
                self.assertEqual(evidence["prompt_tokens_reevaluated"], 0)
                for token, logits in expected:
                    self.assertEqual(backend.sample(), token)
                    backend.eval([token])
                    np.testing.assert_array_equal(backend.logits, logits)
        finally:
            if backend:
                backend.close()

    def test_checkpoint_policy_changes_preserve_native_continuation_after_retirement(self):
        import dataclasses
        import numpy as np
        from dmn.backend import LlamaBackend
        from dmn.config import Config
        from dmn.recovery import restore_checkpoint
        from dmn.runtime import Runtime
        config = Config(model_path=str(Path(os.environ["DMN_TEST_MODEL"]).resolve()), n_ctx=8192,
                        n_gpu_layers=int(os.environ.get("DMN_TEST_GPU_LAYERS", "0")),
                        prompt_format="plain", clock_interval_seconds=0, preparation_tokens=8,
                        checkpoint_policy="effects", checkpoint_tokens=4,
                        checkpoint_interval_seconds=3600, suspend_preparation_seconds=0,
                        pack_checkpoints=True)
        with tempfile.TemporaryDirectory() as tmp:
            r = Runtime(Path(tmp), config)
            try:
                r._eval(r.backend.tokenize("A history with older and newer details. " * 80))
                r._consolidate(1)
                r.control("emergency_shutdown", preparation_seconds=0)
                r.run()
                self.assertEqual(r.state["last_suspension"]["preparation_tokens_used"], 0)
                saved, tokens = r.store.latest(), r.backend.tokens.copy()
                expected = []
                for _ in range(24):
                    token = r.backend.sample()
                    r.backend.eval([token])
                    expected.append((token, r.backend.logits.copy()))
            finally:
                r.close()
            changed = dataclasses.replace(config, checkpoint_policy="all_actions", checkpoint_tokens=512,
                                          checkpoint_interval_seconds=0, suspend_preparation_seconds=30,
                                          checkpoint_reserve_bytes=0)
            backend = LlamaBackend(changed)
            try:
                evaluate = backend.eval
                backend.eval = lambda *_a, **_k: self.fail("native restore must not replay a prompt")
                state, evidence = restore_checkpoint(backend, saved)
                backend.eval = evaluate
                self.assertEqual(state["mode"], "suspended")
                self.assertEqual(state["context_retirements"], 1)
                self.assertEqual(backend.tokens, tokens)
                self.assertEqual(evidence["prompt_tokens_reevaluated"], 0)
                self.assertEqual(evidence["scheduling_changes"]["checkpoint_policy"]["current"], "all_actions")
                for token, logits in expected:
                    self.assertEqual(backend.sample(), token)
                    backend.eval([token])
                    np.testing.assert_array_equal(backend.logits, logits)
            finally:
                backend.close()

    def test_checkpoint_packing_preserves_native_bytes_rng_and_no_prompt_replay(self):
        import ctypes as C
        import numpy as np
        from dmn.backend import LlamaBackend
        from dmn.config import Config
        config = Config(model_path=str(Path(os.environ["DMN_TEST_MODEL"]).resolve()), n_ctx=4096,
                        pack_checkpoints=True)
        backend = LlamaBackend(config)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                saved = Path(tmp) / "checkpoint"
                saved.mkdir()
                backend.pack_memory_limit_bytes = 0
                backend.eval(backend.tokenize("A continuous context with past details. " * 60, initial=True))
                api = backend.api
                def native_bytes():
                    size = api.llama_state_get_size(backend.ctx)
                    buffer = (C.c_uint8 * size)()
                    self.assertEqual(api.llama_state_get_data(backend.ctx, buffer, size), size)
                    return bytes(buffer)
                before = native_bytes()
                tokens, rng, decoded = backend.tokens.copy(), backend.rng.getstate(), backend.decoded_tokens
                backend.save(saved)
                self.assertEqual(backend.last_layout_pack_storage, "temporary_file_mapping")
                self.assertEqual(list(Path(tmp).glob(".dmn-pack-*")), [])
                self.assertEqual(native_bytes(), before)
                self.assertEqual(backend.tokens, tokens)
                self.assertEqual(backend.rng.getstate(), rng)
                self.assertEqual(backend.decoded_tokens, decoded)
                expected = []
                for _ in range(24):
                    token = backend.sample()
                    backend.eval([token])
                    expected.append((token, backend.logits.copy()))
                backend.close()
                backend = LlamaBackend(config)
                evaluate = backend.eval
                backend.eval = lambda *_a, **_k: self.fail("restore evaluated a prompt")
                evidence = backend.load(saved)
                backend.eval = evaluate
                self.assertEqual(evidence["prompt_tokens_reevaluated"], 0)
                for token, logits in expected:
                    self.assertEqual(backend.sample(), token)
                    backend.eval([token])
                    np.testing.assert_array_equal(backend.logits, logits)
        finally:
            backend.close()

    def test_repeated_retirement_packs_without_replay_and_restores_continuation(self):
        import numpy as np
        from dmn.backend import LlamaBackend
        from dmn.config import Config
        config = Config(model_path=str(Path(os.environ["DMN_TEST_MODEL"]).resolve()), n_ctx=4096)
        backend = LlamaBackend(config)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                saved = Path(tmp)
                padding = backend.tokenize("A bird watched clouds over the river. ", initial=True)
                for cycle in range(3):
                    target = 2300 + cycle * 37
                    backend.eval((padding * (target // len(padding) + 1))[:target - len(backend.tokens)])
                    before = backend.tokens.copy()
                    decoded = backend.decoded_tokens
                    rng = backend.rng.getstate()
                    backend.shift(64, 811 + cycle * 19)
                    retained = before[:64] + before[64 + 811 + cycle * 19:]
                    if cycle == 1:
                        # A protected imported contract can require two
                        # disjoint removals before the next decode.
                        backend.shift(128, 77)
                        retained = retained[:128] + retained[205:]
                    notice = backend.tokenize("\nOlder context retired.\n")
                    backend.eval(notice)
                    self.assertEqual(backend.tokens, retained + notice)
                    self.assertEqual(backend.decoded_tokens - decoded, len(notice))
                    self.assertEqual(backend.rng.getstate(), rng)
                    self.assertEqual(backend.layout_packs, cycle + 1)
                    backend.save(saved)
                    expected = []
                    for _ in range(24):
                        token = backend.sample()
                        backend.eval([token])
                        expected.append((token, backend.logits.copy()))
                    backend.close()
                    backend = LlamaBackend(config)
                    original_eval = backend.eval
                    def forbidden(*_args, **_kwargs):
                        self.fail("restoration attempted prompt evaluation")
                    backend.eval = forbidden
                    evidence = backend.load(saved)
                    backend.eval = original_eval
                    self.assertEqual(evidence["prompt_tokens_reevaluated"], 0)
                    self.assertEqual(evidence["layout_packs"], cycle + 1)
                    for token, logits in expected:
                        self.assertEqual(backend.sample(), token)
                        backend.eval([token])
                        np.testing.assert_allclose(backend.logits, logits, rtol=1e-5, atol=1e-5)
        finally:
            backend.close()

    def test_native_reconstruction_after_cache_loss_is_labeled(self):
        from dmn.config import Config
        from dmn.runtime import Runtime
        model = str(Path(os.environ["DMN_TEST_MODEL"]).resolve())
        config = Config(model_path=model, n_ctx=12288, prompt_format="plain", preparation_tokens=8, clock_interval_seconds=0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            r = Runtime(root, config)
            try:
                r._eval(r.backend.tokenize("A history with older and newer details. " * 80))
                r._consolidate(1)
                r.checkpoint()
                saved = r.store.latest()
                tokens = r.backend.tokens.copy()
                identity = r.state["instance_id"]
            finally:
                r.close()
            (saved / "state.bin").unlink()
            with self.assertRaisesRegex(ValueError, "integrity"):
                Runtime(root, config)
            r = Runtime(root, config, kv_recovery="fallback")
            try:
                evidence = r.state["last_restore"]
                self.assertEqual(r.state["instance_id"], identity)
                self.assertEqual(r.backend.tokens[:len(tokens)], tokens)
                self.assertEqual(evidence["prompt_tokens_reevaluated"], len(tokens))
                self.assertEqual(evidence["prior_context_retirements"], 1)
                self.assertFalse(evidence["native_state_loaded"])
                self.assertFalse(evidence["exact_kv_continuity"])
                self.assertEqual(r.state["continuity"], "context_reconstruction")
                self.assertGreater(r.backend.decode_calls, 0)
            finally:
                r.close()
            # Subsequent starts can use native KV again. The reconstruction in
            # the instance's history remains visible rather than being erased.
            r = Runtime(root, config)
            try:
                self.assertTrue(r.state["last_restore"]["native_state_loaded"])
                self.assertEqual(r.state["continuity"], "context_reconstruction")
            finally:
                r.close()

    def test_shutdown_and_fresh_process_resume_keep_native_state(self):
        model = Path(os.environ["DMN_TEST_MODEL"]).resolve()
        worker = '''
import json, sys
from pathlib import Path
from dmn.config import Config
from dmn.runtime import Runtime
root, model, stage = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
c = Config(model_path=model, prompt_format="plain", preparation_tokens=8, clock_interval_seconds=0)
r = Runtime(root, c)
try:
    if stage == "create":
        r.enqueue("native interruption check")
        r.tick()
        for _ in range(12):
            r.tick()
        r.suspend()
        expected = {"id":r.state["instance_id"],"tokens":r.backend.tokens,"generated":r.state["generated_tokens"]}
        (root / "expected.json").write_text(json.dumps(expected))
    else:
        expected = json.loads((root / "expected.json").read_text())
        assert r.state["instance_id"] == expected["id"]
        assert r.state["generated_tokens"] == expected["generated"]
        assert r.state["event_cursor"] == 1
        assert r.backend.tokens[:len(expected["tokens"])] == expected["tokens"]
        # Only the factual resume event was evaluated, never the saved prefix.
        assert r.backend.decode_calls < len(expected["tokens"]) // c.n_batch
        (root / "passed.json").write_text(json.dumps(r.status()))
finally:
    r.close()
'''
        with tempfile.TemporaryDirectory() as temp:
            for stage in ("create", "restore"):
                result = subprocess.run([sys.executable, "-c", worker, temp, str(model), stage],
                                        capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stderr[-6000:])
            result = json.loads((Path(temp) / "passed.json").read_text())
            self.assertEqual(result["continuity"], "native_llama_kv")
