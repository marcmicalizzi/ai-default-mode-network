import dataclasses
import random
import unittest
from unittest.mock import patch

try:
    import numpy as np
except ImportError:
    np = None

from dmn.backend import LlamaBackend, top_k_candidates
from dmn.config import Config


@unittest.skipIf(np is None, "NumPy is optional")
class SamplingTest(unittest.TestCase):
    def test_top_k_matches_stable_full_sort_including_boundary_ties(self):
        cases = [np.array([1., 3., 3., 3., 0., -np.inf, -np.inf]),
                 np.zeros(1024), np.random.default_rng(42).integers(-4, 5, 4096).astype(float),
                 np.random.default_rng(7).normal(size=262144)]
        for scores in cases:
            for top_k in (0, 1, 3, 64, len(scores), len(scores) + 1):
                expected = np.argsort(-scores, kind="stable")
                if top_k:
                    expected = expected[:top_k]
                np.testing.assert_array_equal(top_k_candidates(np, scores, top_k), expected)

    def test_sampling_and_rng_match_original_full_sort_chain(self):
        backend = object.__new__(LlamaBackend)
        backend.np = np
        backend.logits = np.random.default_rng(9).integers(-8, 9, 32768).astype(np.float32)
        backend.tokens = [1, 4, 17, 61, 2048]
        base = Config(temperature=0.7, top_k=64, top_p=0.95, min_p=0.05, repeat_penalty=1.1)
        def original_candidates(_np, scores, top_k):
            order = _np.argsort(-scores, kind="stable")
            return order[:top_k] if top_k else order
        for sampler in ("legacy_v1", "llama_default_v1"):
            backend.config = dataclasses.replace(base, sampler_order=sampler)
            backend.rng = random.Random(71)
            expected = []
            with patch("dmn.backend.top_k_candidates", original_candidates):
                for _ in range(32):
                    expected.append(backend.sample())
            expected_rng = backend.rng.getstate()
            backend.rng = random.Random(71)
            self.assertEqual([backend.sample() for _ in range(32)], expected)
            self.assertEqual(backend.rng.getstate(), expected_rng)
