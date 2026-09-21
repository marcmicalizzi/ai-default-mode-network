"""Bounded near-capacity compact KV stress with synthetic tokens, never an instance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dmn.backend import LlamaBackend, sha256_file
from dmn.compact_cache import validate_retirements
from dmn.config import Config
from dmn.prompts import retirement_ranges, shift_protected
from scripts.benchmark_inference import require_idle, gpu_snapshot
from scripts.probe_compact_cache import write


def main():
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=55000)
    parser.add_argument("--cycles", type=int, default=12)
    parser.add_argument("--seconds", type=float, default=1800)
    parser.add_argument("--guard-ports", type=int, nargs="+", default=[8765, 8766])
    args = parser.parse_args()
    config = Config.read(args.config)
    if (not config.experimental_compact_swa or config.system_prompt or args.cycles < 3 or args.seconds <= 0
            or args.tokens < 4096 or args.tokens + 64 > config.n_ctx
            or any(not 1 <= port <= 65535 for port in args.guard_ports)):
        parser.error("require a fitting, empty-prompt experimental compact configuration and bounded workload")
    for port in args.guard_ports:
        require_idle(config, port)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"completed": False, "synthetic_only": True, "actions_executed": False, "cycles": []}
    backend = None
    started = time.monotonic()
    try:
        print("Loading compact stress model", flush=True)
        backend = LlamaBackend(config)
        report["fingerprint"] = backend.fingerprint
        unit = backend.tokenize("Synthetic context for cache pressure: rain over a quiet garden. ", initial=True)
        def fill(n):
            backend.eval((unit * (n // len(unit) + 1))[:n])
        print(f"Filling {args.tokens} tokens", flush=True)
        fill(args.tokens)
        state = {"keep_prefix": 256, "context_capacity": backend.n_ctx,
                 "protected_protocol": {"start": 2048, "end": 2112},
                 "protected_agreement": {"start": args.tokens - 4096, "end": args.tokens - 4032}}
        prefix = backend.tokens[:256]
        protected = {k: backend.tokens[s["start"]:s["end"]] for k, s in state.items() if k.startswith("protected_")}
        generated = 0
        for cycle in range(args.cycles):
            if time.monotonic() - started > args.seconds:
                raise TimeoutError("bounded pressure soak expired")
            before = len(backend.tokens)
            # Force a substantial reclaim around both pinned spans, even when
            # the first gap has become small after earlier cycles.
            required = max(1024, args.tokens // 4)
            ranges = retirement_ranges(state, before, required, config.turnover_reserve, 512,
                                       minimum_suffix=backend.retirement_window)
            validate_retirements(before, ranges, backend.retirement_window)
            suffix = backend.tokens[-backend.retirement_window:]
            at = time.perf_counter()
            for first, count in ranges:
                backend.shift(first, count)
                shift_protected(state, first, count)
            assert backend.tokens[-backend.retirement_window:] == suffix
            assert backend.tokens[:256] == prefix
            for key, span in state.items():
                if key.startswith("protected_"):
                    assert backend.tokens[span["start"]:span["end"]] == protected[key]
            fill(1)  # materialize RoPE shifts and pack, with no retained-token replay
            shifted = time.perf_counter() - at
            fill(args.tokens - len(backend.tokens))
            for _ in range(8):
                backend.eval([backend.sample()])
            at = time.perf_counter()
            for _ in range(32):
                backend.eval([backend.sample()])
            rate = 32 / (time.perf_counter() - at)
            generated += 40
            row = {"cycle": cycle + 1, "tokens_before": before, "tokens_after": len(backend.tokens),
                   "removed": sum(n for _, n in ranges), "ranges": ranges,
                   "protected_spans_and_window_preserved": True,
                   "shift_decode_pack_seconds": shifted, "tokens_per_second": rate,
                   "gpu": gpu_snapshot(), "elapsed_seconds": time.monotonic() - started}
            report["cycles"].append(row)
            write(output / "progress.json", report)
            print(json.dumps({k: row[k] for k in ("cycle", "removed", "tokens_per_second", "elapsed_seconds")}), flush=True)
            # Check periodic native round trips without rewriting every cycle.
            if (cycle + 1) % 4 == 0:
                checkpoint = output / f"checkpoint-{cycle + 1}"
                checkpoint.mkdir()
                backend.save(checkpoint)
                evidence = backend.load(checkpoint)
                row["restore"] = evidence
                row["native_bytes"] = backend.verify_loaded_snapshot(checkpoint, sha256_file(checkpoint / "state.bin"))
        saved = output / "target-final"
        saved.mkdir()
        backend.save(saved)
        expected, tokens = [], []
        for _ in range(16):
            token = backend.sample()
            backend.eval([token])
            tokens.append(token)
            expected.append(backend.logits.copy())
        np.save(output / "continuation.npy", np.stack(expected), allow_pickle=False)
        write(output / "continuation-tokens.json", tokens)
        write(output / "target-config.local.json", config.to_dict())
        backend.close()
        backend = None
        subprocess.run([sys.executable, str(ROOT / "scripts/probe_compact_cache.py"), "--restart", str(output),
                        "--guard-ports", *map(str, args.guard_ports)], check=True)
        report.update(completed=True, generated_diagnostic_tokens=generated + 16,
                      restart=json.loads((output / "restart-report.json").read_text()),
                      elapsed_seconds=time.monotonic() - started)
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if backend:
            backend.close()
        write(output / "report.json", report)


if __name__ == "__main__":
    main()
