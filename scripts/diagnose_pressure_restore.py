"""Read-only, action-free diagnostic of a pressure-test native snapshot."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dmn.backend import LlamaBackend, sha256_file
from dmn.config import Config


def main():
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=ROOT / "data/context-pressure-01")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = Config.read(args.experiment / "config.json")
    backend = LlamaBackend(config)
    try:
        saved = args.experiment / "comparison-state"
        reference = np.load(args.experiment / "continuation-reference.npz", allow_pickle=False)
        evidence = backend.load(saved)
        backend.save(args.output)
        results = {"actions_executed": False, "source_modified": False, "restore": evidence,
                   "native_state_roundtrip_byte_identical": sha256_file(saved / "state.bin") == sha256_file(args.output / "state.bin"),
                   "original_native_sha256": sha256_file(saved / "state.bin"),
                   "roundtrip_native_sha256": sha256_file(args.output / "state.bin"), "branches": []}
        restored_reference = []
        for branch in range(2):
            backend.load(saved)
            errors, sample_differences = [], []
            for i, (token, logits) in enumerate(zip(reference["tokens"], reference["logits"])):
                sample = backend.sample()
                if sample != int(token):
                    sample_differences.append(i)
                # Force the same token to isolate cache arithmetic from sampling.
                backend.eval([int(token)])
                expected = logits if branch == 0 else restored_reference[i]
                errors.append(float(np.max(np.abs(expected - backend.logits))))
                if branch == 0:
                    restored_reference.append(backend.logits.copy())
            results["branches"].append({"comparison": "uninterrupted_vs_restored" if branch == 0 else "restored_vs_restored",
                                        "teacher_forced_tokens": len(errors), "logit_maximum_absolute_errors": errors,
                                        "sample_difference_steps_vs_original": sample_differences})
        (args.output / "report.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(results, indent=2), flush=True)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
