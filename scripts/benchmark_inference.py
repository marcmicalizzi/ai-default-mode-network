"""Measure a fresh synthetic context, never an existing DMN instance.

Heavy runs refuse to start while the local DMN port is open. Tiny, single-thread
CPU fixtures can exercise the harness while a real instance remains running.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dmn.config import Config


def require_idle(config, port=8765):
    small_cpu_fixture = (config.n_gpu_layers == 0 and config.n_threads == 1
                         and config.n_ctx <= 4096
                         and Path(config.model_path).stat().st_size <= 16 * 1024 * 1024)
    if small_cpu_fixture:
        return
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            pass
    except ConnectionRefusedError:
        return
    except OSError as exc:
        raise RuntimeError("Cannot establish that the runtime is stopped; benchmark refused") from exc
    raise RuntimeError("Local DMN port is open; refusing a competing model benchmark")


def summary(values):
    ordered = sorted(values)
    return {"total_seconds": sum(values), "mean_ms": 1000 * sum(values) / len(values),
            "p50_ms": 1000 * ordered[len(ordered) // 2],
            "p95_ms": 1000 * ordered[math.ceil(len(ordered) * 0.95) - 1]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new directory for diagnostic evidence")
    parser.add_argument("--tokens", type=int, default=20000, help="occupied synthetic context, separate from capacity")
    parser.add_argument("--warmup", type=int, default=8, help="unmeasured decode steps for shape/kernel warmup")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--guard-port", type=int, default=8765)
    parser.add_argument("--native-log-level", choices=["debug", "info", "warning", "error"], default="warning")
    args = parser.parse_args(argv)
    config = Config.read(args.config)
    if (config.backend != "llama" or args.tokens < 1 or args.warmup < 0 or args.steps < 8
            or args.tokens + args.warmup + args.steps > config.n_ctx):
        parser.error("require a llama config and a fitting context with at least eight measured steps")
    if config.system_prompt:
        parser.error("use a disposable config with an empty system prompt")
    if not 1 <= args.guard_port <= 65535:
        parser.error("guard-port must be between 1 and 65535")
    require_idle(config, args.guard_port)
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["DMN_NATIVE_LOG_LEVEL"] = args.native_log_level
    # This import happens only after the resource guard. No runtime is created.
    from dmn.backend import LlamaBackend
    backend = None
    report = {"completed": False, "synthetic_only": True, "runtime_actions_executed": False,
              "capacity_requested": config.n_ctx, "occupied_tokens_requested": args.tokens,
              "warmup_steps": args.warmup, "measured_steps": args.steps,
              "native_log_level": args.native_log_level, "config": config.to_dict(),
              "timing_scope": "Load, prefill, warmup and steady decoding separately; no snapshots or actions."}
    try:
        print("Loading diagnostic model", flush=True)
        started = time.perf_counter()
        backend = LlamaBackend(config)
        report["load_seconds"] = time.perf_counter() - started
        report["fingerprint"] = backend.fingerprint
        report["native_retirement_supported"] = backend.can_shift
        unit = backend.tokenize("A synthetic benchmark passage about clouds above a garden. ", initial=True)
        tokens = (unit * (args.tokens // len(unit) + 1))[:args.tokens]
        print(f"Evaluating {len(tokens)} synthetic tokens", flush=True)
        started = time.perf_counter()
        backend.eval(tokens)
        report["prefill_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        for _ in range(args.warmup):
            backend.eval([backend.sample()])
        report["warmup_seconds"] = time.perf_counter() - started
        sampling, decoding = [], []
        started = time.perf_counter()
        for _ in range(args.steps):
            before = time.perf_counter()
            token = backend.sample()
            sampled = time.perf_counter()
            backend.eval([token])
            after = time.perf_counter()
            sampling.append(sampled - before)
            decoding.append(after - sampled)
        report.update(completed=True, steady_seconds=time.perf_counter() - started,
                      sampling=summary(sampling), native_decode_and_logit_copy=summary(decoding))
        report["steady_tokens_per_second"] = args.steps / report["steady_seconds"]
        print(json.dumps({k: report[k] for k in ("load_seconds", "prefill_seconds", "warmup_seconds",
              "steady_tokens_per_second", "sampling", "native_decode_and_logit_copy", "native_retirement_supported")}, indent=2))
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if backend:
            backend.close()


if __name__ == "__main__":
    main()
