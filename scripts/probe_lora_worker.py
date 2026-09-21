"""Run the existing synthetic CPU training/conversion probe under Windows limits.

This creates only a new disposable output directory. It cannot open an instance,
approve a learning plan or install an adapter. No GPU or downloads are required.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dmn.worker_limits import WorkerLimits, run_cpu_worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-python", type=Path, required=True)
    parser.add_argument("--native-python", type=Path, required=True)
    parser.add_argument("--converter", type=Path, required=True)
    parser.add_argument("--max-ram-mib", type=int, default=1536, help="aggregate job committed-memory limit")
    parser.add_argument("--max-seconds", type=float, default=180)
    args = parser.parse_args()
    limits = WorkerLimits(max_committed_bytes=args.max_ram_mib * 1024**2, max_seconds=args.max_seconds)
    if sys.platform != "win32":
        parser.error("this worker experiment requires Windows; Linux containment is still pending")
    output = args.output.resolve()
    # Never reuse, inspect or overwrite an existing instance/probe directory.
    output.mkdir(parents=True, exist_ok=False)
    result = run_cpu_worker(args.training_python, [str(ROOT / "scripts/probe_lora_training.py"),
        "--output", str(output / "probe"), "--converter", str(args.converter.resolve()),
        "--native-python", str(args.native_python.resolve())], cwd=ROOT, log=output / "worker.log", limits=limits)
    result.update(synthetic_only=True, runtime_constructed=False, production_sleep_enabled=False)
    (output / "worker.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["succeeded"] else 1)


if __name__ == "__main__":
    main()
