"""Inspect by default; explicitly opt into a contained tiny GPU experiment."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dmn.storage import write_durable
from dmn.worker_limits import WorkerLimits, run_gpu_research_worker
from scripts.probe_qlora import inspect_fixture
from scripts.qlora_prepare import ALLOCATOR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--training-python", type=Path)
    parser.add_argument("--torch-vram-mib", type=int, default=1024)
    parser.add_argument("--max-ram-mib", type=int, default=4096)
    parser.add_argument("--max-seconds", type=int, default=180)
    parser.add_argument("--execute", action="store_true", help="actually use GPU 0 after maintenance consent; default only inspects")
    parser.add_argument("--stream-source", action="store_true")
    parser.add_argument("--vision-cpu", action="store_true")
    args = parser.parse_args()
    plan = inspect_fixture(args.fixture)
    limits = WorkerLimits(args.max_ram_mib * 1024**2, args.max_seconds)
    if not 256 <= args.torch_vram_mib <= 2048:
        parser.error("tiny GPU probe permits only 256..2048 MiB of PyTorch allocator memory")
    plan["requested_limits"] = {"max_ram_mib": args.max_ram_mib, "max_seconds": args.max_seconds,
                                "torch_allocator_mib": args.torch_vram_mib, "total_vram_quota": False}
    plan["source_loader"] = "tensor_stream_v1" if args.stream_source else "safetensors_default"
    plan["vision_cpu"] = args.vision_cpu
    plan["torch_allocator_configuration"] = ALLOCATOR
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    if args.output is None or args.training_python is None:
        parser.error("execution requires a fresh --output and isolated --training-python")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_durable(output / "plan.json", plan)
    result = run_gpu_research_worker(args.training_python, [str(ROOT / "scripts/probe_qlora.py"),
        "--execute", "--fixture", str(args.fixture.resolve()), "--output", str(output / "probe"),
        "--torch-vram-mib", str(args.torch_vram_mib), *(["--stream-source"] if args.stream_source else []),
        *(["--vision-cpu"] if args.vision_cpu else [])],
        cwd=ROOT, log=output / "worker.log", limits=limits, allow_gpu=True)
    write_durable(output / "process.json", result)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["succeeded"] else 1)


if __name__ == "__main__":
    main()
