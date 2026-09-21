"""Research-only LoRA wake mechanics; no trainer, production instance or GPU."""
import os
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(os.environ.get("DMN_TEST_SWA_MODEL"), "set DMN_TEST_SWA_MODEL to the tiny generated Gemma4 fixture")
class LoraWakeTest(unittest.TestCase):
    def test_changed_weights_rebuild_and_unchanged_weights_restore(self):
        from scripts.probe_lora_wake import run
        with tempfile.TemporaryDirectory() as folder:
            report = run(Path(os.environ["DMN_TEST_SWA_MODEL"]), Path(folder) / "probe")
            self.assertTrue(report["completed"])
            self.assertFalse(report["training_performed"])
            self.assertFalse(report["runtime_constructed"])
            self.assertLessEqual(report["zero_scale_max_logit_difference"], 1e-6)
            self.assertEqual(len(report["updates"]), 2)
            for update in report["updates"]:
                self.assertTrue(update["tokens_and_rng_preserved"])
                self.assertTrue(update["fresh_logits_equal"])
                self.assertTrue(update["weight_change_altered_kv"])
                self.assertGreater(update["wake"]["prompt_tokens_reevaluated"], 0)
                self.assertEqual(update["restore"]["prompt_tokens_reevaluated"], 0)
            self.assertTrue(report["restart"]["fresh_process"])
            self.assertTrue(report["restart"]["tokens_equal"])
            self.assertTrue(report["restart"]["logits_equal"])


if __name__ == "__main__":
    unittest.main()
