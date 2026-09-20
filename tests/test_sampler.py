import unittest
import importlib.util
from types import SimpleNamespace


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is part of the optional native dependencies")
class SamplerOrderTest(unittest.TestCase):
    def test_source_filter_order_and_legacy_order_have_distinct_expected_support(self):
        import numpy as np
        from dmn.backend import LlamaBackend
        from dmn.config import Config
        backend = LlamaBackend.__new__(LlamaBackend)
        backend.np, backend.tokens = np, []
        backend.logits = np.log([.6, .3, .1])
        backend.rng = SimpleNamespace(random=lambda: .9)
        # Before temperature, p=.7 retains two candidates (.6 + .3).
        # Temperature=.5 first would concentrate >.7 on the first candidate.
        for order, expected in (("llama_default_v1", 1), ("legacy_v1", 0)):
            with self.subTest(order=order):
                backend.config = Config(temperature=.5, top_k=0, top_p=.7, min_p=0,
                                        repeat_penalty=1, sampler_order=order)
                self.assertEqual(backend.sample(), expected)
