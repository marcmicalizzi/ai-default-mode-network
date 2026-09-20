import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dmn.config import Config
from scripts.benchmark_matrix import execute_plan, main, make_plan


class MatrixTest(unittest.TestCase):
    def test_plan_only_never_loads_or_executes_a_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config.json"
            config.write_text(json.dumps(Config(n_ctx=60000, n_threads=6, n_gpu_layers=24).to_dict()))
            with patch("scripts.benchmark_matrix.subprocess.run", side_effect=AssertionError("must not execute")):
                main(["--config", str(config), "--output", str(root / "planned")])
            plan = json.loads((root / "planned/plan.json").read_text())
            self.assertEqual(plan["threads"], [6, 8, 12])
            self.assertEqual(plan["gpu_layers"], [24, 27, 30])
            self.assertFalse(plan["completed"])
            self.assertEqual(plan["runs"], [])

    def test_sequential_trials_select_threads_then_layers_and_recheck_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = Config(n_ctx=60000, n_threads=6, n_gpu_layers=24)
            plan = make_plan(config, [8, 12], [27, 30], 25000, 8, 64, 2)
            calls = []
            def child(command, **kwargs):
                candidate = json.loads(Path(command[command.index("--config") + 1]).read_text())
                output = Path(command[command.index("--output") + 1])
                calls.append((candidate["n_threads"], candidate["n_gpu_layers"]))
                output.mkdir()
                # More threads can be slower: the runner must use the measurements.
                speed = {6: 1., 8: 1.5, 12: 1.2}[candidate["n_threads"]] + (candidate["n_gpu_layers"] - 24) / 6
                (output / "report.json").write_text(json.dumps({"completed": True, "steady_tokens_per_second": speed}))
                return SimpleNamespace(returncode=0)
            with patch("scripts.benchmark_matrix.require_idle") as idle, patch("scripts.benchmark_matrix.subprocess.run", side_effect=child):
                report = execute_plan(plan, root, "python", 8765)
            self.assertEqual(calls, [(6, 24)] * 2 + [(8, 24)] * 2 + [(12, 24)] * 2 +
                             [(8, 27)] * 2 + [(8, 30)] * 2 + [(6, 24)] * 2)
            self.assertEqual(idle.call_count, len(calls))
            self.assertEqual(report["best_measured"]["n_threads"], 8)
            self.assertEqual(report["best_measured"]["n_gpu_layers"], 30)
            self.assertEqual(report["baseline_recheck_ratio"], 1)
            self.assertFalse(report["promotion_performed"])

    def test_failed_trial_stops_remaining_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = make_plan(Config(n_ctx=60000), [8], [24], 25000, 8, 64, 1)
            with patch("scripts.benchmark_matrix.require_idle"), patch("scripts.benchmark_matrix.subprocess.run", return_value=SimpleNamespace(returncode=1)) as child:
                with self.assertRaisesRegex(RuntimeError, "remaining trials were not started"):
                    execute_plan(plan, Path(tmp), "python", 8765)
            self.assertEqual(child.call_count, 1)
            self.assertFalse(json.loads((Path(tmp) / "report.json").read_text())["completed"])

    def test_invalid_config_or_context_is_rejected(self):
        for config in (Config(backend="demo"), Config(system_prompt="private prompt")):
            with self.assertRaises(ValueError):
                make_plan(config, [8], [30], 256, 8, 64, 1)
        with self.assertRaises(ValueError):
            make_plan(Config(), [8], [30], 8190, 8, 64, 1)
