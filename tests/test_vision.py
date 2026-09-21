"""Native adapter mechanics; real CPU inference is explicitly opt-in."""
import ctypes as C
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from dmn.backend import LlamaBackend
from dmn.config import Config
from dmn.recovery import same_native_environment, reconstruction_compatible
from dmn.vision import PreparedImages


class ProjectorIdentityTest(unittest.TestCase):
    def test_input_projector_can_change_without_relaxing_language_model_identity(self):
        saved = {"kind": "native_llama_kv", "model_sha256": "language-model", "config": Config().to_dict()}
        current = {**saved, "vision": {"projector_sha256": "projector"},
                   "config": Config(vision_projector_path="test.gguf").to_dict()}
        self.assertTrue(same_native_environment(saved, current))
        reconstruction_compatible(saved, current)
        changed = {**current, "model_sha256": "different-model"}
        self.assertFalse(same_native_environment(saved, changed))
        with self.assertRaisesRegex(ValueError, "same model"):
            reconstruction_compatible(saved, changed)


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is optional")
class VisionAdapterTest(unittest.TestCase):
    def test_visual_positions_are_counted_but_not_sampled_as_text(self):
        import numpy as np
        backend = LlamaBackend.__new__(LlamaBackend)
        logits = (C.c_float * 3)(1, 2, 4)
        backend.api = SimpleNamespace(llama_get_logits_ith=lambda *_: logits)
        backend.config = Config(n_ctx=4096, temperature=0, repeat_last_n=64, repeat_penalty=100)
        backend.tokens, backend.decoded_tokens, backend.decode_calls = [0], 1, 1
        backend.np, backend.n_vocab, backend.n_ctx, backend.ctx = np, 3, 4096, 100
        def evaluate(ctx, lctx, chunks, past, seq, batch, last, end):
            self.assertEqual((ctx, lctx, chunks, past, seq, last), (101, 100, 102, 1, 0, True))
            C.cast(end, C.POINTER(C.c_int32))[0] = 4
            return 0
        vision = SimpleNamespace(backend=backend, ctx=101,
                                 api=SimpleNamespace(mtmd_helper_eval_chunks=evaluate))
        PreparedImages(vision, 102, [1, -1, -1]).evaluate()
        self.assertEqual(backend.tokens, [0, 1, -1, -1])
        self.assertEqual(backend.decoded_tokens, 4)
        self.assertEqual(backend.sample(), 2)  # -1 must not penalize the final vocabulary token.
        with self.assertRaisesRegex(ValueError, "visual positions"):
            backend.eval([-1])

    def test_partial_native_decode_is_fatal_and_not_acknowledged(self):
        backend = SimpleNamespace(tokens=[1], n_ctx=4096, ctx=100, config=Config())
        vision = SimpleNamespace(backend=backend, ctx=101,
            api=SimpleNamespace(mtmd_helper_eval_chunks=Mock(return_value=1)))
        with self.assertRaisesRegex(RuntimeError, "restore the last committed checkpoint"):
            PreparedImages(vision, 102, [-1]).evaluate()
        self.assertEqual(backend.tokens, [1])


@unittest.skipUnless(os.environ.get("DMN_TEST_VISION_MODEL") and os.environ.get("DMN_TEST_VISION_PROJECTOR"),
                     "set dedicated vision model/projector paths for disposable CPU validation")
class NativeVisionTest(unittest.TestCase):
    def test_native_save_restore_retirement_and_no_pixel_archive(self):
        import numpy as np
        from dmn.attachments import decode_uploads
        from tests.test_attachments import upload
        config = Config(model_path=os.environ["DMN_TEST_VISION_MODEL"],
                        vision_projector_path=os.environ["DMN_TEST_VISION_PROJECTOR"],
                        n_ctx=4096, n_batch=512, n_threads=1, n_gpu_layers=0,
                        offload_kqv=False, flash_attn=False, pack_checkpoints=True)
        backend = None
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            try:
                backend = LlamaBackend(config)
                backend.eval(backend.tokenize("A synthetic visual input follows. ", initial=True))
                keep = len(backend.tokens)
                with backend.vision.prepare(decode_uploads([upload()])) as prepared:
                    prepared.evaluate()
                backend.eval(backend.tokenize("\nDescribe the colors. "))
                self.assertIn(-1, backend.tokens)
                backend.save(path)
                tokens, logits = backend.tokens.copy(), backend.logits.copy()
                chosen = backend.sample()
                backend.close()
                backend = LlamaBackend(config)
                evidence = backend.load(path)
                self.assertEqual(evidence["decode_calls_during_load"], 0)
                self.assertEqual(backend.tokens, tokens)
                np.testing.assert_array_equal(backend.logits, logits)
                self.assertEqual(backend.sample(), chosen)
                with self.assertRaisesRegex(ValueError, "visual positions"):
                    backend.rebuild(path)
                if backend.can_shift:
                    last_image = max(i for i, token in enumerate(tokens) if token == -1)
                    backend.shift(keep, last_image - keep + 1)
                    backend.eval(backend.tokenize("\nThe image was retired. "))
                    self.assertNotIn(-1, backend.tokens)
                    backend.save(path)
                    backend.rebuild(path)
                self.assertEqual(set(json.loads((path / "engine.json").read_text())) & {"images", "pixels", "embeddings"}, set())
            finally:
                if backend:
                    backend.close()
