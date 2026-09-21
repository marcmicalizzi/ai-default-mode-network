import dataclasses
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from dmn.backend import DemoBackend, LlamaBackend
from dmn.compact_cache import gemma4_retirement_window, validate_retirements
from dmn.config import Config
from dmn.prompts import retirement_ranges, shift_protected
from dmn.recovery import same_native_environment, native_placement_changes
from dmn.runtime import Runtime, ContextFull


class CompactRetirementTest(unittest.TestCase):
    def config(self, **changes):
        return dataclasses.replace(Config(swa_full=False, experimental_compact_swa=True,
            flash_attn=True, pack_checkpoints=True, type_k="q8_0", type_v="q8_0"), **changes)

    def test_opt_in_requires_reviewed_geometry_and_configuration(self):
        metadata = {"general.architecture": "gemma4", "gemma4.attention.key_length": "512",
                    "gemma4.attention.key_length_swa": "256"}
        self.assertEqual(gemma4_retirement_window(self.config(), lambda key: metadata.get(key, ""), 1024, "0.3.35"), 1024)
        for changes in ({"swa_full": True}, {"pack_checkpoints": False}, {"flash_attn": False},
                        {"type_k": "q4_0"}, {"backend": "demo"}, {"experimental_compact_swa": 1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.config(**changes)
        for change in ({"general.architecture": "gemma3"}, {"gemma4.attention.shared_kv_layers": "6"},
                       {"gemma4.attention.key_length_swa": "32"}, {"gemma4.attention.key_length": ""}):
            values = {**metadata, **change}
            with self.subTest(change=change), self.assertRaises(ValueError):
                gemma4_retirement_window(self.config(), lambda key: values.get(key, ""), 1024, "0.3.35")
        with self.assertRaises(ValueError):
            gemma4_retirement_window(self.config(), lambda key: metadata.get(key, ""), 1024, "0.3.36")

    def test_planner_protects_spans_and_entire_suffix_across_disjoint_removals(self):
        state = {"keep_prefix": 200, "protected_protocol": {"start": 400, "end": 500},
                 "protected_agreement": {"start": 1800, "end": 1900}, "context_capacity": 2048}
        original = list(range(2048))
        tokens = original.copy()
        ranges = retirement_ranges(state, len(tokens), 400, 256, 128, minimum_suffix=256)
        self.assertGreater(len(ranges), 1)
        validate_retirements(len(tokens), ranges, 256)
        for start, count in ranges:
            del tokens[start:start + count]
            shift_protected(state, start, count)
        self.assertEqual(tokens[-256:], original[-256:])
        self.assertEqual(tokens[:200], original[:200])
        self.assertEqual(tokens[200:300], original[400:500])  # protocol folds into prefix
        span = state["protected_agreement"]
        self.assertEqual(tokens[span["start"]:span["end"]], original[1800:1900])

    def test_insufficient_room_fails_without_relaxing_suffix(self):
        with self.assertRaisesRegex(ValueError, "no usable context"):
            retirement_ranges({"keep_prefix": 1950, "context_capacity": 2048}, 2048, 1, 256, 0, minimum_suffix=256)

    def test_direct_unsafe_shift_never_calls_native_memory(self):
        backend = LlamaBackend.__new__(LlamaBackend)
        backend.can_shift, backend.retirement_window = True, 64
        backend.tokens, backend.api = list(range(1024)), Mock()
        with self.assertRaises(ValueError):
            backend.shift(16, 945)
        self.assertEqual(backend.tokens, list(range(1024)))
        self.assertEqual(backend.api.mock_calls, [])

    def test_runtime_preflights_entire_plan_and_saves_before_any_removal(self):
        config = Config(backend="demo", n_ctx=24576, preparation_tokens=1, clock_interval_seconds=0)
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Runtime(Path(tmp), config, DemoBackend(config, b"thought "))
            try:
                runtime.backend.retirement_window = 64
                runtime._eval([120] * 1024)
                def unsafe(*args, **kwargs):
                    return [(runtime.state["keep_prefix"], 16), (len(runtime.backend.tokens) - 64, 1)]
                with patch("dmn.runtime.retirement_ranges", side_effect=unsafe), patch.object(runtime.backend, "shift") as shift:
                    with self.assertRaises(ContextFull):
                        runtime._consolidate(1)
                    shift.assert_not_called()
                self.assertEqual(runtime.state["mode"], "context_full")
                self.assertTrue(runtime.store.latest())
            finally:
                runtime.close()

    def test_existing_full_checkpoint_cannot_silently_switch_to_compact(self):
        old = {"kind": "native_llama_kv", "config": Config().to_dict()}
        changed = {**old, "config": self.config().to_dict()}
        self.assertFalse(same_native_environment(old, changed))
        with self.assertRaises(ValueError):
            native_placement_changes(old, changed)
        old["config"].pop("experimental_compact_swa")
        self.assertTrue(same_native_environment(old, {**old, "config": Config().to_dict()}))


if __name__ == "__main__":
    unittest.main()
