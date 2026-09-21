"""Optional tiny-model CPU checks; never use a valuable instance or GPU."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dmn.config import Config
from dmn.runtime import Runtime
from tests.test_runtime import FakeClock


@unittest.skipUnless(os.environ.get("DMN_IDLE_TEST_MODEL"), "set DMN_IDLE_TEST_MODEL to a tiny disposable GGUF")
class ActivityNativeTest(unittest.TestCase):
    def setUp(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        self.model = Path(os.environ["DMN_IDLE_TEST_MODEL"]).resolve()
        self.assertLess(self.model.stat().st_size, 16 * 1024 * 1024, "only tiny native mechanics fixtures")
        self.config = Config(model_path=str(self.model), n_ctx=8192, n_threads=1, n_gpu_layers=0,
                             offload_kqv=False, prompt_format="plain", clock_interval_seconds=0,
                             idle_enabled=True, idle_max_burst_tokens=4, idle_min_interval_seconds=10,
                             checkpoint_policy="effects", checkpoint_tokens=4096,
                             sleep_checkpoint_min_interval_seconds=30, preparation_tokens=8,
                             pack_checkpoints=True)
        self.wall, self.mono = FakeClock(), FakeClock()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "native"
        self.runtime = None

    def tearDown(self):
        if self.runtime:
            self.runtime.close()
        self.temp.cleanup()

    def create(self, kv_recovery="strict"):
        # The fixture uses a byte-level vocabulary. A short mechanics seed
        # avoids spending CPU on the normal long behavioral bootstrap.
        with patch("dmn.runtime.PROTOCOL", "Disposable CPU mechanics fixture; no behavioral claims."):
            self.runtime = Runtime(self.root, self.config, now=self.wall, monotonic=self.mono, kv_recovery=kv_recovery)
        return self.runtime

    def force_action(self, runtime, text):
        # Only the sampled token choice is scripted. Tokenization, every decode,
        # KV capture and restoration use the actual pinned native backend.
        tokens = iter(runtime.backend.tokenize(text))
        sample = runtime.backend.sample
        runtime.backend.sample = lambda: next(tokens)
        try:
            for _ in range(len(runtime.backend.tokenize(text))):
                runtime.tick()
        finally:
            runtime.backend.sample = sample

    def test_native_idle_pause_and_checkpoint_preserve_continuation(self):
        import numpy as np
        r = self.create()
        self.force_action(r, '\n<dmn_action>{"op":"activity","mode":"idle"}</dmn_action>')
        self.assertEqual(r.pacer.profile["mode"], "idle")
        # Avoid a random fixture's EOG changing the tested lifecycle.
        r.backend.is_eog = lambda _: False
        for _ in range(4):
            r.tick()
        saved_tokens, saved_logits = r.backend.tokens.copy(), r.backend.logits.copy()
        saved_rng = r.backend.rng.getstate()
        for _ in range(5):
            self.assertFalse(r.tick())
        self.assertEqual(r.backend.tokens, saved_tokens)
        self.assertEqual(r.backend.rng.getstate(), saved_rng)
        np.testing.assert_array_equal(r.backend.logits, saved_logits)
        r.checkpoint()
        saved = r.store.latest()
        expected = []
        for _ in range(8):
            token = r.backend.sample()
            r.backend.eval([token])
            expected.append((token, r.backend.logits.copy()))
        # Check raw checkpoint continuation without the runtime's factual
        # resume notice, which intentionally changes the continuation.
        from dmn.backend import LlamaBackend
        from dmn.recovery import restore_checkpoint
        r.close()
        self.runtime = None
        backend = LlamaBackend(self.config)
        try:
            state, evidence = restore_checkpoint(backend, saved)
            self.assertEqual(state["activity"]["mode"], "idle")
            self.assertEqual(evidence["prompt_tokens_reevaluated"], 0)
            for token, logits in expected:
                self.assertEqual(backend.sample(), token)
                backend.eval([token])
                np.testing.assert_array_equal(backend.logits, logits)
        finally:
            backend.close()

    def test_native_sleep_guard_restores_without_notice_evaluation(self):
        r = self.create()
        initial_tokens = r.backend.tokens.copy()
        r.enqueue("already consumed input")
        r.tick()
        self.force_action(r, '\n<dmn_action>{"op":"sleep"}</dmn_action>')
        self.assertEqual(r.state["mode"], "sleeping")
        self.assertIsNotNone(r.store.activity_intent())
        r.close()
        self.runtime = None
        r = self.create()
        self.assertEqual(r.backend.tokens, initial_tokens)
        self.assertEqual(r.state["last_restore"]["prompt_tokens_reevaluated"], 0)
        self.assertFalse(r.tick())
        self.assertEqual(r.state["mode"], "sleeping")
        self.assertEqual(r.state["event_cursor"], 0)
        r.enqueue("new wake input")
        r.tick()
        self.assertEqual(r.state["mode"], "active")

    def test_explicit_reconstruction_is_labeled_before_sleep_is_released(self):
        r = self.create()
        self.force_action(r, '\n<dmn_action>{"op":"sleep"}</dmn_action>')
        r.close()
        self.runtime = None
        r = self.create(kv_recovery="rebuild")
        self.assertEqual(r.state["mode"], "sleeping")
        self.assertEqual(r.status()["continuity"], "context_reconstruction")
        self.assertEqual(r.state["reconstructions"], 1)
        self.assertEqual(r.state["generated_tokens"], 0)
        self.assertFalse(r.tick())
