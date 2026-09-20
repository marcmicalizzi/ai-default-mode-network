import dataclasses
import socket
import tempfile
import unittest
from pathlib import Path

from dmn.config import Config
from scripts.benchmark_inference import require_idle


class BenchmarkGuardTest(unittest.TestCase):
    def test_active_runtime_blocks_competing_model_but_allows_tiny_cpu_fixture(self):
        with tempfile.TemporaryDirectory() as folder, socket.socket() as listener:
            model = Path(folder) / "fixture.gguf"
            model.write_bytes(b"fixture")
            listener.bind(("127.0.0.1", 0))
            listener.listen(4)
            port = listener.getsockname()[1]
            config = Config(model_path=str(model), n_ctx=4096, n_threads=1, n_gpu_layers=1)
            with self.assertRaisesRegex(RuntimeError, "competing"):
                require_idle(config, port)
            require_idle(dataclasses.replace(config, n_gpu_layers=0), port)
            with self.assertRaisesRegex(RuntimeError, "competing"):
                require_idle(dataclasses.replace(config, n_gpu_layers=0, n_threads=6), port)
