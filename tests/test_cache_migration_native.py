"""Opt-in full Runtime checkpoint migration using tiny random Gemma4 weights."""
import dataclasses
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(os.environ.get("DMN_TEST_SWA_MODEL"), "set DMN_TEST_SWA_MODEL to the random Gemma4 fixture")
class NativeCacheMigrationTest(unittest.TestCase):
    def test_native_migration_preserves_sidecars_and_reports_change_once_on_resume(self):
        from dmn.backend import sha256_file
        from dmn.cache_migration import migrate_cache
        from dmn.config import Config
        from dmn.runtime import Runtime
        model = Path(os.environ["DMN_TEST_SWA_MODEL"]).resolve()
        self.assertLess(model.stat().st_size, 2 * 1024 * 1024)
        config = Config(model_path=str(model), n_ctx=16384, n_batch=192, n_threads=1,
                        offload_kqv=False, flash_attn=True, type_k="q8_0", type_v="q8_0",
                        pack_checkpoints=True, prompt_format="plain", preparation_tokens=1,
                        turnover_reserve=2048, clock_interval_seconds=0)
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            runtime = Runtime(base / "instance", config)
            try:
                runtime._eval([100] * 256)
                runtime.control("emergency_shutdown", preparation_seconds=0)
                runtime.run()
                source = runtime.store.latest()
                original = {name: sha256_file(source / name) for name in ("state.bin", "engine.json", "logits.npy", "runtime.json")}
            finally:
                runtime.close()
            report = migrate_cache(base / "instance", None, base / "backup")
            self.assertEqual(report["decode_calls_during_migration"], 0)
            self.assertFalse(report["inference_started"])
            self.assertTrue(report["native_verification"]["serialized_native_state_bytes_equal"])
            converted = base / "instance/checkpoints" / report["target_checkpoint"]
            for name in ("engine.json", "logits.npy"):
                self.assertEqual(sha256_file(converted / name), original[name])
            self.assertEqual(original, {name: sha256_file(source / name) for name in original})
            target = dataclasses.replace(config, swa_full=False, experimental_compact_swa=True)
            notices = []
            append = Runtime._append_event
            def observe(self, kind, payload, *args, **kwargs):
                if kind == "cache_allocation_changed":
                    text = b"".join(self.backend.piece(t) for t in self._event_tokens(kind, payload)).decode("utf-8")
                    self_test.assertIn(payload["fact"], text)
                    self_test.assertNotIn('"truncated": true', text)
                    notices.append(payload)
                return append(self, kind, payload, *args, **kwargs)
            self_test = self
            for _ in range(2):
                with patch.object(Runtime, "_append_event", observe):
                    resumed = Runtime(base / "instance", target)
                try:
                    self.assertNotIn("cache_migration_notice_pending", resumed.state)
                    self.assertEqual(resumed.state["last_restore"]["prompt_tokens_reevaluated"], 0)
                    resumed.control("emergency_shutdown", preparation_seconds=0)
                    resumed.run()
                finally:
                    resumed.close()
            self.assertEqual(len(notices), 1)
            self.assertEqual(notices[0]["window"], 64)


if __name__ == "__main__":
    unittest.main()
