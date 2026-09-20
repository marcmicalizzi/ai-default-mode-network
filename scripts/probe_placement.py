"""Validate thread/layer changes using fresh synthetic KV, never an instance."""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dmn.backend import LlamaBackend, sha256_file
from dmn.config import Config
from dmn.recovery import restore_checkpoint
from scripts.benchmark_inference import require_idle


def save_probe(backend, directory):
    directory.mkdir()
    backend.save(directory)
    (directory / "runtime.json").write_text('{"schema":1,"synthetic_diagnostic":true}', encoding="utf-8")
    files = {name: sha256_file(directory / name) for name in
             ("state.bin", "engine.json", "logits.npy", "runtime.json")}
    (directory / "manifest.json").write_text(json.dumps({"fingerprint": backend.fingerprint, "files": files}), encoding="utf-8")


def verify(config, target, output, tokens, steps, guard_port):
    import numpy as np
    backend = None
    report = {"completed": False, "synthetic_only": True, "instance_used": False,
              "source_config": config.to_dict(), "target_config": target.to_dict(),
              "tokens": tokens, "steps": steps, "retirement_verified": False,
              "limits": "Evidence for this build/model/configuration; future CPU/GPU arithmetic may differ."}
    try:
        require_idle(config, guard_port)
        print("Loading source diagnostic placement", flush=True)
        backend = LlamaBackend(config)
        unit = backend.tokenize("A synthetic placement check about clouds and a garden. ", initial=True)
        print(f"Evaluating {tokens} synthetic tokens", flush=True)
        backend.eval((unit * (tokens // len(unit) + 1))[:tokens])
        source = output / "source"
        print("Saving source native state and reference continuation", flush=True)
        save_probe(backend, source)
        expected = []
        for _ in range(steps):
            token = backend.sample()
            backend.eval([token])
            expected.append((token, backend.logits.copy()))
        backend.close()
        backend = None
        require_idle(target, guard_port)
        print("Loading target placement and verifying native state", flush=True)
        backend = LlamaBackend(target)
        started = time.perf_counter()
        _, report["placement_restore"] = restore_checkpoint(backend, source, allow_placement_change=True)
        report["placement_restore_seconds"] = time.perf_counter() - started
        differences, max_error = [], 0.0
        for index, (token, logits) in enumerate(expected):
            sampled = backend.sample()
            if sampled != token:
                differences.append(index)
            # Identical token inputs isolate arithmetic differences across placements.
            backend.eval([token])
            max_error = max(max_error, float(np.max(np.abs(backend.logits - logits))))
        report["cross_placement_forced_token_comparison"] = {
            "sample_difference_indices": differences, "maximum_logit_absolute_error": max_error,
            "not_an_uninterrupted_target_continuation": True}
        if not backend.can_shift:
            raise RuntimeError("target placement cannot retire context")
        print("Checking repeated retirement under target placement", flush=True)
        for cycle in range(3):
            backend.shift(16, min(64, len(backend.tokens) - 17))
            backend.eval(backend.tokenize("\nOlder synthetic context retired.\n"))
        report["retirement_verified"] = True
        saved = output / "target-after-retirement"
        save_probe(backend, saved)
        expected = []
        for _ in range(steps):
            token = backend.sample()
            backend.eval([token])
            expected.append((token, backend.logits.copy()))
        backend.close()
        backend = None
        require_idle(target, guard_port)
        print("Restarting target placement and checking continuation", flush=True)
        backend = LlamaBackend(target)
        _, report["target_restart"] = restore_checkpoint(backend, saved)
        max_error = 0.0
        for token, logits in expected:
            if backend.sample() != token:
                raise AssertionError("target placement changed sampled tokens after restart")
            backend.eval([token])
            max_error = max(max_error, float(np.max(np.abs(backend.logits - logits))))
            np.testing.assert_allclose(backend.logits, logits, rtol=1e-5, atol=1e-5)
        report.update(completed=True, target_restart_maximum_logit_absolute_error=max_error,
                      target_restart_sampled_tokens_equal=True)
        return report
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        finally:
            if backend:
                backend.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--gpu-layers", type=int, required=True)
    parser.add_argument("--tokens", type=int, default=25000)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--guard-port", type=int, default=8765)
    args = parser.parse_args(argv)
    config = Config.read(args.config)
    if (config.backend != "llama" or config.system_prompt or args.tokens < 256 or args.steps < 8
            or args.tokens + 2 * args.steps + 128 > config.n_ctx or not 1 <= args.guard_port <= 65535
            or args.gpu_layers < -1):
        parser.error("require an empty-prompt llama diagnostic config, fitting context and valid placement/port")
    target = dataclasses.replace(config, n_threads=args.threads, n_gpu_layers=args.gpu_layers)
    require_idle(config, args.guard_port)
    require_idle(target, args.guard_port)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = verify(config, target, output, args.tokens, args.steps, args.guard_port)
    print(json.dumps({k: report[k] for k in ("completed", "placement_restore", "retirement_verified",
          "cross_placement_forced_token_comparison", "target_restart_maximum_logit_absolute_error")}, indent=2))


if __name__ == "__main__":
    main()
