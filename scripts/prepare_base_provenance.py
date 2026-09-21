"""Prepare reusable conversion/quantization evidence on tiny local CPU fixtures.

No instance is opened and no learning is authorized. The generated record can be
offered in a v2 training recipe, which still requires model review and approval.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dmn.base_provenance import prepare
from dmn.worker_limits import WorkerLimits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conversion", type=Path, required=True, help="JSON with pinned source/converter/interpreter metadata")
    parser.add_argument("--quantizer", type=Path, help="JSON emitted by the native interpreter's dmn.provenance_native identity")
    parser.add_argument("--quantization", choices=["F32", "Q8_0", "Q4_0", "Q4_K_M"], default="F32")
    parser.add_argument("--max-ram-mib", type=int, default=1536)
    parser.add_argument("--max-seconds", type=int, default=180)
    args = parser.parse_args()
    request = {"schema": 1, "kind": "tiny_cpu_base_provenance_v1", "conversion": json.loads(args.conversion.read_text()),
        "quantization": args.quantization, "quantizer": json.loads(args.quantizer.read_text()) if args.quantizer else None}
    result = prepare(args.output, request, WorkerLimits(args.max_ram_mib * 1024**2, args.max_seconds))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
