"""Opt in using the random fixture from scripts/generate_swa_fixture.py."""
import os
import dataclasses
from pathlib import Path
import shutil
import tempfile
import unittest

from dmn.config import Config


@unittest.skipUnless(os.environ.get("DMN_TEST_SWA_MODEL"), "set DMN_TEST_SWA_MODEL to the random Gemma4 fixture")
class CompactRuntimeNativeTest(unittest.TestCase):
    def test_shifted_retained_keys_match_full_cache_reference_from_identical_state(self):
        from dmn.backend import LlamaBackend
        from scripts.compact_cache_state import compact_state, inspect_state, row_hashes
        model = Path(os.environ["DMN_TEST_SWA_MODEL"]).resolve()
        self.assertLess(model.stat().st_size, 2 * 1024 * 1024)
        for kind in ("f16", "q8_0"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = Config(model_path=str(model), n_ctx=2048, n_batch=64, n_threads=1,
                                n_gpu_layers=0, offload_kqv=False, flash_attn=True,
                                type_k=kind, type_v=kind, pack_checkpoints=True)
                full = compact = None
                try:
                    full = LlamaBackend(config)
                    full.eval([50 + i % 160 for i in range(1024)])
                    source, target = root / "full", root / "compact"
                    source.mkdir()
                    target.mkdir()
                    full.save(source)
                    compact_state(source / "state.bin", target / "state.bin", 64)
                    for name in ("engine.json", "logits.npy"):
                        shutil.copyfile(source / name, target / name)
                    compact = LlamaBackend(dataclasses.replace(config, swa_full=False, experimental_compact_swa=True))
                    compact.load(target)
                    # Track rows from the shared initial state. New-token rows
                    # can differ through attention arithmetic and aren't exact
                    # references for testing the position-shift operation itself.
                    old_rows = [True] * 1024
                    for cycle in range(3):
                        for backend in (full, compact):
                            backend.shift(16, 128)
                            backend.eval([80 + cycle])
                        del old_rows[16:144]
                        old_rows.append(False)
                        for backend, name in ((full, "full"), (compact, "compact")):
                            path = root / f"{name}-{cycle}"
                            path.mkdir()
                            backend.save(path)
                        a = inspect_state(root / f"full-{cycle}/state.bin")
                        b = inspect_state(root / f"compact-{cycle}/state.bin")
                        for cache_name in ("global_cache", "local_cache"):
                            ah = row_hashes(a, getattr(a, cache_name))
                            bh = row_hashes(b, getattr(b, cache_name))
                            compared = 0
                            for key, digest in bh.items():
                                if old_rows[key[1]]:
                                    self.assertEqual(ah[key], digest, (kind, cycle, cache_name, key))
                                    compared += 1
                            self.assertGreater(compared, 0)
                finally:
                    if full:
                        full.close()
                    if compact:
                        compact.close()

    def test_runtime_retirement_preserves_protocol_suffix_and_native_restart(self):
        import numpy as np
        from dmn.backend import LlamaBackend
        from dmn.recovery import restore_checkpoint
        from dmn.runtime import Runtime
        model = Path(os.environ["DMN_TEST_SWA_MODEL"]).resolve()
        self.assertLess(model.stat().st_size, 2 * 1024 * 1024, "use only the tiny random fixture")
        config = Config(model_path=str(model), n_ctx=16384, n_batch=192, n_threads=1, n_gpu_layers=0,
                        offload_kqv=False, flash_attn=True, type_k="q8_0", type_v="q8_0",
                        swa_full=False, experimental_compact_swa=True, pack_checkpoints=True,
                        prompt_format="plain", preparation_tokens=1, turnover_reserve=2048,
                        clock_interval_seconds=0)
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Runtime(Path(tmp), config)
            try:
                self.assertTrue(runtime.backend.can_shift)
                self.assertFalse(runtime.backend.api.llama_memory_can_shift(runtime.backend.memory))
                prefix = runtime.backend.tokens.copy()
                runtime._eval([100] * 1500)
                start = len(runtime.backend.tokens)
                runtime._eval([101] * 64)
                runtime.state["protected_agreement"] = {"start": start, "end": start + 64}
                runtime._eval([102] * 1500)
                for _ in range(3):
                    runtime._consolidate(1)
                    self.assertEqual(runtime.backend.tokens[:len(prefix)], prefix)
                    span = runtime.state["protected_agreement"]
                    self.assertEqual(runtime.backend.tokens[span["start"]:span["end"]], [101] * 64)
                    self.assertEqual(runtime.state["mode"], "active")
                saved = runtime.store.latest()
                expected = []
                for _ in range(8):
                    token = runtime.backend.sample()
                    runtime.backend.eval([token])
                    expected.append((token, runtime.backend.logits.copy()))
            finally:
                runtime.close()
            backend = LlamaBackend(config)
            try:
                state, evidence = restore_checkpoint(backend, saved)
                self.assertEqual(evidence["prompt_tokens_reevaluated"], 0)
                self.assertEqual(state["context_retirements"], 3)
                for token, logits in expected:
                    self.assertEqual(backend.sample(), token)
                    backend.eval([token])
                    np.testing.assert_array_equal(backend.logits, logits)
            finally:
                backend.close()


if __name__ == "__main__":
    unittest.main()
