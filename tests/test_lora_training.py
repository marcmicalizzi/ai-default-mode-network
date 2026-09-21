"""Opt-in end-to-end training/conversion test; standard CI needs no trainer."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


@unittest.skipUnless(all(os.environ.get(key) for key in (
    "DMN_TEST_TRAIN_PYTHON", "DMN_TEST_LORA_CONVERTER", "DMN_TEST_NATIVE_PYTHON")),
    "set DMN_TEST_TRAIN_PYTHON, DMN_TEST_LORA_CONVERTER and DMN_TEST_NATIVE_PYTHON for CPU LoRA research")
class LoraTrainingTest(unittest.TestCase):
    def test_learned_adapter_conversion_and_wake(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "probe"
            result = subprocess.run([
                os.environ["DMN_TEST_TRAIN_PYTHON"], str(root / "scripts/probe_lora_training.py"),
                "--output", str(output), "--converter", os.environ["DMN_TEST_LORA_CONVERTER"],
                "--native-python", os.environ["DMN_TEST_NATIVE_PYTHON"],
            ], cwd=root, capture_output=True, text=True, timeout=300)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr +
                             "\n".join(p.read_text(errors="replace")[-5000:]
                                       for p in output.glob("*.log")))
            report = json.loads((output / "report.json").read_text())
            self.assertTrue(report["completed"])
            self.assertFalse(report["runtime_constructed"])
            self.assertFalse(report["actions_executed"])
            self.assertFalse(report["gpu_used"])
            training = report["training"]
            self.assertIsNone(training["cuda_build"])
            self.assertTrue(training["frozen_base_unchanged"])
            self.assertTrue(training["peft_reload_logits_equal"])
            self.assertTrue(training["confirmation_check_passed"])
            self.assertTrue(report["conversion"]["all_24_factors_bit_equal"])
            native = report["native"]
            self.assertTrue(native["completed"])
            for comparison in native["comparisons"].values():
                self.assertLessEqual(comparison["max_logit_difference"], .005)
            self.assertTrue(native["retained_tokens_and_rng_equal"])
            self.assertTrue(native["rebuilt_logits_equal_fresh"])
            self.assertTrue(native["strict_old_weights_rejected"])
            self.assertGreater(native["wake"]["prompt_tokens_reevaluated"], 0)
            self.assertEqual(native["restart"]["restore"]["prompt_tokens_reevaluated"], 0)
            self.assertTrue(native["restart"]["eight_tokens_and_logits_equal"])


if __name__ == "__main__":
    unittest.main()
