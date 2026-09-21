"""Opt-in CPU repeated-learning/quantized-inference experiment."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


@unittest.skipUnless(all(os.environ.get(key) for key in (
    "DMN_TEST_TRAIN_PYTHON", "DMN_TEST_LORA_CONVERTER", "DMN_TEST_NATIVE_PYTHON", "DMN_TEST_LORA_PARENT")),
    "set training/native interpreters, converter and DMN_TEST_LORA_PARENT for this CPU experiment")
class LoraRepeatTest(unittest.TestCase):
    def test_second_learning_cycle_and_quantized_base_transfer(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "repeat"
            result = subprocess.run([
                os.environ["DMN_TEST_TRAIN_PYTHON"], str(root / "scripts/probe_lora_repeat.py"),
                "--parent", os.environ["DMN_TEST_LORA_PARENT"], "--output", str(output),
                "--converter", os.environ["DMN_TEST_LORA_CONVERTER"],
                "--native-python", os.environ["DMN_TEST_NATIVE_PYTHON"],
            ], cwd=root, capture_output=True, text=True, timeout=300)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr +
                "\n".join(p.read_text(errors="replace")[-4000:] for p in output.rglob("*.log")))
            report = json.loads((output / "report.json").read_text())
            self.assertTrue(report["completed"])
            self.assertTrue(report["parent_unchanged"])
            self.assertFalse(report["runtime_constructed"])
            self.assertFalse(report["actions_executed"])
            self.assertFalse(report["gpu_used"])
            for name in ("new_only", "with_replay"):
                training = report["training"][name]
                self.assertTrue(training["base_unchanged"])
                self.assertTrue(training["peft_reload_exact"])
                self.assertNotEqual(training["parent_adapter_sha256"], training["candidate_adapter_sha256"])
                self.assertLess(training["scores"]["new_test"]["cross_entropy"],
                                report["training"]["parent_scores"]["new_test"]["cross_entropy"])
                self.assertTrue(report["conversion"][name]["all_24_factors_bit_equal"])
            for precision in ("q8_0", "q4_0"):
                self.assertGreater(report["quantized_bases"][precision]["tensor_types"][precision.upper()], 0)
            native = report["native"]
            self.assertEqual(len(native["evaluations"]), 9)
            self.assertTrue(native["completed"])
            self.assertTrue(native["wake"]["retained_tokens_and_rng_equal"])
            self.assertTrue(native["wake"]["fresh_rebuild_logits_equal"])
            restart = native["wake"]["restart"]
            self.assertTrue(restart["fresh_process"])
            self.assertTrue(restart["eight_tokens_and_logits_equal"])
            self.assertEqual(restart["restore"]["prompt_tokens_reevaluated"], 0)


if __name__ == "__main__":
    unittest.main()
