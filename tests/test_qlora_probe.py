import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from dmn.storage import write_durable
from scripts.probe_qlora import inspect_fixture, execute


class QloraInspectionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.fixture = self.root / "fixture"
        (self.fixture / "base").mkdir(parents=True)
        write_durable(self.fixture / "fixture.json", {"synthetic_only": True})
        write_durable(self.fixture / "base/config.json", {"architectures": ["Gemma4ForConditionalGeneration"],
            "text_config": {"hidden_size": 256, "num_hidden_layers": 2, "vocab_size": 263, "max_position_embeddings": 2048}})
        (self.fixture / "base/model.safetensors").write_bytes(b"contract-only, never loaded")

    def tearDown(self):
        self.temp.cleanup()

    def test_default_inspection_needs_no_torch_or_gpu_and_creates_no_outputs(self):
        with patch.dict(sys.modules, {"torch": None, "bitsandbytes": None}):
            plan = inspect_fixture(self.fixture)
        self.assertFalse(plan["gpu_execution"])
        self.assertEqual(plan["training"]["quantization"], "NF4")
        result = subprocess.run([sys.executable, "scripts/launch_qlora_probe.py", "--fixture", str(self.fixture)],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(json.loads(result.stdout)["gpu_execution"])
        self.assertFalse(json.loads(result.stdout)["requested_limits"]["total_vram_quota"])
        self.assertEqual({p.name for p in self.root.iterdir()}, {"fixture"})

    def test_execution_refuses_before_importing_cuda_without_contained_opt_in(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "-1", "DMN_GPU_PROBE_CONTAINED": ""}), patch.dict(sys.modules, {"torch": None}):
            with self.assertRaisesRegex(ValueError, "contained research launcher"):
                execute(self.fixture, self.root / "output", 1024)
        self.assertFalse((self.root / "output").exists())

    def test_large_or_nonfixture_models_cannot_be_substituted(self):
        write_durable(self.fixture / "fixture.json", {"synthetic_only": False})
        with self.assertRaisesRegex(ValueError, "generated tiny"):
            inspect_fixture(self.fixture)
        write_durable(self.fixture / "fixture.json", {"synthetic_only": True})
        with (self.fixture / "base/model.safetensors").open("wb") as stream:
            stream.seek(4 * 1024**2)
            stream.write(b"x")
        with self.assertRaisesRegex(ValueError, "generated tiny"):
            inspect_fixture(self.fixture)


if __name__ == "__main__":
    unittest.main()
