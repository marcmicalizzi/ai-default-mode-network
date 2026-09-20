"""Discarded backend-only experiment: fragmented versus packed KV restoration."""
import argparse
import ctypes as C
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dmn.backend import LlamaBackend, sha256_file
from dmn.config import Config


def pack(backend):
    api = backend.api
    size = api.llama_state_get_size(backend.ctx)
    buffer = (C.c_uint8 * size)()
    assert api.llama_state_get_data(backend.ctx, buffer, size) == size
    assert api.llama_state_set_data(backend.ctx, buffer, size) == size


def main():
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=ROOT / "data/context-pressure-01")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = Config.read(args.experiment / "config.json")
    backend = LlamaBackend(config)
    # Explicitly reproduce the old, fragmented layout for the control. Current
    # production eval packs after shift; this discarded diagnostic controls
    # packing manually so the control never accidentally tests the fix.
    backend._pack_native_layout = lambda: setattr(backend, "_layout_pending", False)
    results = []
    try:
        backend.load(args.experiment / "comparison-state")
        for cycle, normalized in enumerate([False, True, True, True]):
            padding = backend.tokenize(" Diagnostic load: cloud stone river meadow.")
            target = 3200 + cycle * 37
            backend.eval((padding * (target // len(padding) + 1))[:target - len(backend.tokens)])
            backend.shift(955, 800 + cycle * 41)
            backend.eval(backend.tokenize("\nDiagnostic retirement completed.\n"))
            before = args.output / f"cycle-{cycle}-before"
            before.mkdir()
            backend.save(before)
            decoded = backend.decoded_tokens
            if normalized:
                pack(backend)
            saved = args.output / f"cycle-{cycle}-saved"
            saved.mkdir()
            backend.save(saved)
            assert backend.decoded_tokens == decoded
            native_equal = sha256_file(before / "state.bin") == sha256_file(saved / "state.bin")
            expected = []
            for _ in range(24):
                token = backend.sample()
                backend.eval([token])
                expected.append((token, backend.logits.copy()))
            backend.load(saved)
            errors, sample_differences = [], []
            for i, (token, logits) in enumerate(expected):
                if backend.sample() != token:
                    sample_differences.append(i)
                backend.eval([token])
                errors.append(float(np.max(np.abs(logits - backend.logits))))
            row = {"cycle": cycle, "packed_after_retirement": normalized,
                   "packing_preserved_serialized_native_bytes": native_equal,
                   "packing_prompt_tokens_reevaluated": backend.decoded_tokens - decoded - 24,
                   "teacher_forced_tokens": 24, "maximum_logit_absolute_error": max(errors),
                   "sample_difference_steps": sample_differences}
            results.append(row)
            print(json.dumps(row), flush=True)
            (args.output / "report.json").write_text(json.dumps({"actions_executed": False, "cycles": results}, indent=2))
    finally:
        backend.close()


if __name__ == "__main__":
    main()
