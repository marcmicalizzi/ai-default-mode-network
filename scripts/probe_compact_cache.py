"""Disposable Gemma4 full-to-compact conversion, retirement and restart research.

Never accepts an instance or existing checkpoint. The compact target uses the
explicit experimental runtime policy. Ordinary recovery still rejects changing
cache allocation; this harness performs the conversion only on its own fixtures.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dmn.backend import LlamaBackend, sha256_file
from dmn.config import Config
from scripts.benchmark_inference import require_idle, gpu_snapshot
from scripts.compact_cache_state import compact_state, inspect_state, row_hashes, validate_retirements
from scripts.inspect_gguf import inspect


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def save(backend, path):
    path.mkdir()
    started = time.perf_counter()
    backend.save(path)
    return time.perf_counter() - started


def idle(config, ports):
    for port in ports:
        require_idle(config, port)


def compare_conversion(source, converted):
    left, right = inspect_state(source), inspect_state(converted)
    if left.tokens != right.tokens:
        raise AssertionError("conversion changed token IDs")
    for name in ("global_cache", "local_cache"):
        a, b = getattr(left, name), getattr(right, name)
        ah, bh = row_hashes(left, a), row_hashes(right, b)
        if any(ah.get(key) != digest for key, digest in bh.items()):
            raise AssertionError("conversion changed retained native rows")
        if name == "global_cache" and ah != bh:
            raise AssertionError("conversion lost global KV")
    return {"global_kv_rows_byte_equal": True, "retained_local_kv_rows_byte_equal": True,
            "tokens_equal": True}


def check_retired_values(before, after, ranges, window):
    """V never receives RoPE: every surviving V row must remain byte-identical."""
    left, right = inspect_state(before), inspect_state(after)
    checked = 0
    for name in ("global_cache", "local_cache"):
        a, b = getattr(left, name), getattr(right, name)
        ah, bh = row_hashes(left, a), row_hashes(right, b)
        for (layer, pos), digest in ah.items():
            if layer < a.layers:
                continue  # K is deliberately transformed by native RoPE shifting.
            for start, count in ranges:
                if start <= pos < start + count:
                    pos = None
                    break
                if pos >= start + count:
                    pos -= count
            if pos is not None:
                needed = name == "global_cache" or pos >= len(right.tokens) - window
                if needed and (layer, pos) not in bh:
                    raise AssertionError("retirement lost an applicable native V row")
                if (layer, pos) in bh and bh[layer, pos] != digest:
                    raise AssertionError("retirement changed surviving V bytes")
                checked += int((layer, pos) in bh)
    return checked


def restart(output, ports):
    import numpy as np
    config = Config.read(output / "target-config.local.json")
    idle(config, ports)
    backend = LlamaBackend(config)
    try:
        snapshot = output / "target-final"
        restored = backend.load(snapshot)
        verification = backend.verify_loaded_snapshot(snapshot, sha256_file(snapshot / "state.bin"))
        expected = np.load(output / "continuation.npy", allow_pickle=False)
        tokens = json.loads((output / "continuation-tokens.json").read_text())
        errors, differences = [], []
        for i, (token, logits) in enumerate(zip(tokens, expected)):
            if backend.sample() != token:
                differences.append(i)
            backend.eval([token])
            errors.append(float(np.max(np.abs(backend.logits - logits))))
        result = {"fresh_process": True, "restore": restored, "verification": verification,
                  "sample_difference_indices": differences, "max_logit_error": max(errors)}
        write(output / "restart-report.json", result)
        if differences or max(errors) != 0:
            raise AssertionError("compact fresh-process continuation was not bit-identical")
    finally:
        backend.close()


def verify(config, target, output, tokens, cycles, ports):
    import numpy as np
    metadata = inspect(Path(config.model_path))["metadata"]
    if metadata.get("general.architecture") != "gemma4":
        raise ValueError("this research harness is only for Gemma4 STANDARD SWA")
    window = metadata["gemma4.attention.sliding_window"]
    if tokens < 2 * window + 256:
        raise ValueError("synthetic context must hold two local windows plus protected test spans")
    report = {"completed": False, "synthetic_only": True, "instance_used": False,
              "experimental_compact_policy": True, "window": window, "cycles": []}
    backend = None
    try:
        idle(config, ports)
        print("Loading full-cache synthetic source", flush=True)
        backend = LlamaBackend(config)
        if backend.fingerprint["binding_version"] != "0.3.35":
            raise ValueError("research requires the pinned binding and native source")
        report["source_fingerprint"] = backend.fingerprint
        unit = backend.tokenize("A synthetic cache test about clouds above a garden. ", initial=True)
        print(f"Evaluating {tokens} synthetic source tokens", flush=True)
        started = time.perf_counter()
        backend.eval((unit * (tokens // len(unit) + 1))[:tokens])
        report["source_prefill_seconds"] = time.perf_counter() - started
        print("Saving the full-cache source", flush=True)
        report["source_save_seconds"] = save(backend, output / "source")
        backend.close()
        backend = None

        print("Converting only masked local rows; verifying every retained row", flush=True)
        converted = output / "converted"
        converted.mkdir()
        started = time.perf_counter()
        report["conversion"] = compact_state(output / "source/state.bin", converted / "state.bin", window)
        for name in ("engine.json", "logits.npy"):
            shutil.copyfile(output / "source" / name, converted / name)
        report["conversion_seconds"] = time.perf_counter() - started
        report["conversion_verification"] = compare_conversion(output / "source/state.bin", converted / "state.bin")
        idle(target, ports)
        print("Loading compact target with zero prompt replay", flush=True)
        backend = LlamaBackend(target)
        report["target_fingerprint"] = backend.fingerprint
        report["native_compact_can_shift"] = bool(backend.api.llama_memory_can_shift(backend.memory))
        report["policy_compact_can_shift"] = backend.can_shift
        if backend.api.llama_model_n_swa(backend.model) != window:
            raise AssertionError("native SWA window disagrees with GGUF")
        started = time.perf_counter()
        report["conversion_restore"] = backend.load(converted)
        report["conversion_native_verification"] = backend.verify_loaded_snapshot(converted, sha256_file(converted / "state.bin"))
        report["conversion_load_and_verify_seconds"] = time.perf_counter() - started
        report["gpu_after_load"] = gpu_snapshot()
        for cycle in range(cycles):
            print(f"Compact retirement cycle {cycle + 1}/{cycles}", flush=True)
            before, after = output / f"before-{cycle}", output / f"after-{cycle}"
            save(backend, before)
            count = max(64, window // 4)
            ranges = [(16, count), (128, count)]
            validate_retirements(len(backend.tokens), ranges, window)
            protected = backend.tokens[count + 32:count + 64]
            prefix = backend.tokens[:16]
            started = time.perf_counter()
            for start, amount in ranges:
                backend.shift(start, amount)
            backend.eval([unit[cycle % len(unit)]])
            if backend.tokens[:16] != prefix or backend.tokens[32:64] != protected:
                raise AssertionError("retirement changed a protected synthetic span")
            shift_seconds = time.perf_counter() - started
            save(backend, after)
            checked = check_retired_values(before / "state.bin", after / "state.bin", ranges, window)
            # Refill across the local ring boundary repeatedly, not just shift it.
            fill = 2 * count - 1
            backend.eval((unit * (fill // len(unit) + 1))[:fill])
            report["cycles"].append({"cycle": cycle, "ranges": ranges,
                                     "protected_prefix_and_middle_span_unchanged": True,
                                     "surviving_v_rows_checked": checked, "shift_and_decode_seconds": shift_seconds,
                                     "tokens_after_refill": len(backend.tokens)})
        report["final_save_seconds"] = save(backend, output / "target-final")
        expected, sampled = [], []
        started = time.perf_counter()
        for _ in range(16):
            token = backend.sample()
            backend.eval([token])
            sampled.append(token)
            expected.append(backend.logits.copy())
        report["post_retirement_tokens_per_second"] = 16 / (time.perf_counter() - started)
        np.save(output / "continuation.npy", np.stack(expected), allow_pickle=False)
        write(output / "continuation-tokens.json", sampled)
        backend.close()
        backend = None
        write(output / "target-config.local.json", target.to_dict())
        print("Checking native restart and continuation in a fresh process", flush=True)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--restart", str(output),
                        "--guard-ports", *map(str, ports)], check=True)
        report["restart"] = json.loads((output / "restart-report.json").read_text())
        report["completed"] = True
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if backend:
            backend.close()
        write(output / "report.json", report)
    print(json.dumps({key: report[key] for key in ("completed", "conversion", "post_retirement_tokens_per_second", "restart")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--target-gpu-layers", type=int)
    parser.add_argument("--guard-ports", nargs="+", type=int, default=[8765, 8766])
    parser.add_argument("--restart", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if any(not 1 <= p <= 65535 for p in args.guard_ports):
        parser.error("invalid guard port")
    if args.restart:
        restart(args.restart.resolve(), args.guard_ports)
        return
    if not args.config or not args.output:
        parser.error("--config and --output are required")
    config = Config.read(args.config)
    if (config.backend != "llama" or config.system_prompt or not config.swa_full or
            (args.target_gpu_layers is not None and args.target_gpu_layers < -1) or
            config.type_k not in ("f16", "q8_0") or config.type_v not in ("f16", "q8_0") or
            not config.flash_attn or not config.pack_checkpoints or args.cycles < 2 or
            not 1 <= args.tokens <= config.n_ctx - 16):
        parser.error("require a disposable empty-prompt full-SWA packed F16/Q8 flash-attention config and fitting context")
    target = dataclasses.replace(config, swa_full=False, experimental_compact_swa=True, n_gpu_layers=(config.n_gpu_layers
        if args.target_gpu_layers is None else args.target_gpu_layers))
    idle(config, args.guard_ports)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    verify(config, target, output, args.tokens, args.cycles, args.guard_ports)


if __name__ == "__main__":
    main()
