"""Fresh-process native comparisons at a specified occupied context size.

Diagnostic branches only: generation never executes runtime actions. Every
checkpoint and reference is kept under a new experiment directory.
"""
from __future__ import annotations
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dmn.backend import LlamaBackend, sha256_file
from dmn.config import Config
from dmn.storage import write_durable


def resources():
    data = {}
    if os.name == "nt":
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
                (name, ctypes.c_size_t) for name in ("peak_working_set", "working_set", "peak_paged_pool", "paged_pool",
                    "peak_nonpaged_pool", "nonpaged_pool", "pagefile", "peak_pagefile", "private_bytes")]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        get_handle = ctypes.windll.kernel32.GetCurrentProcess
        get_handle.restype = ctypes.c_void_p
        get_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_info.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        if get_info(get_handle(), ctypes.byref(counters), counters.cb):
            data.update({"process_" + name: getattr(counters, name) for name in
                         ("working_set", "peak_working_set", "private_bytes", "peak_pagefile")})
    try:
        result = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.free,power.draw",
            "--format=csv,noheader,nounits"], text=True, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        memory, free, power = map(float, result.splitlines()[0].split(","))
        data["whole_device"] = {"memory_used_mib": memory, "memory_free_mib": free, "power_w": power}
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        data["gpu_unavailable"] = str(exc)
    return data


def worker(args):
    import numpy as np
    folder = args.output.resolve()
    config = Config.read(folder / "config.json")
    stage = args.stage
    started = time.monotonic()
    backend = None
    report = {"stage": stage, "started_at": time.time(), "completed": False}
    try:
        backend = LlamaBackend(config)
        report.update(load_seconds=time.monotonic() - started, after_load=resources(),
                      native_can_shift=backend.can_shift, fingerprint=backend.fingerprint)
        checkpoint = folder / ("shifted" if stage == "restore-shifted" else "unshifted")
        if stage == "create":
            checkpoint.mkdir()
            unit = backend.tokenize("A disposable diagnostic passage about clouds crossing a garden. ", initial=True)
            tokens = (unit * (args.tokens // len(unit) + 1))[:args.tokens]
            before = time.monotonic()
            # Batch-by-batch progress survives a native abort or interrupted run.
            for offset in range(0, len(tokens), config.n_batch):
                backend.eval(tokens[offset:offset + config.n_batch])
                write_durable(folder / "progress.json", {"stage": stage, "evaluated_tokens": len(backend.tokens),
                    "target_tokens": args.tokens, "prefill_seconds": time.monotonic() - before})
            report["prefill_seconds"] = time.monotonic() - before
        else:
            original_eval = backend.eval
            backend.eval = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("restore attempted prompt eval"))
            before = time.monotonic()
            report["restore"] = backend.load(checkpoint)
            report["restore_seconds"] = time.monotonic() - before
            backend.eval = original_eval
            reference = np.load(folder / (checkpoint.name + "-reference.npz"), allow_pickle=False)
            errors, same = [], []
            for expected_token, expected_logits in zip(reference["tokens"], reference["logits"]):
                token = backend.sample()
                same.append(token == int(expected_token))
                # Teacher force the reference to compare the same causal branch
                # even if a sampled token first differs; never hide mismatches.
                backend.eval([int(expected_token)])
                errors.append(float(np.max(np.abs(backend.logits - expected_logits))))
            report.update(sampled_tokens_equal=all(same), maximum_logit_absolute_error=max(errors),
                per_step_maximum_errors=errors, compared_tokens=len(errors),
                continuation_verified=all(same) and max(errors) <= 1e-5)
            if stage == "restore-shifted":
                report["completed"] = True
                return
            if not report["continuation_verified"]:
                raise AssertionError("unshifted fresh-process continuation differs")
            if not backend.can_shift:
                raise RuntimeError("native context does not support retirement")
            checkpoint = folder / "shifted"
            checkpoint.mkdir()
            before = time.monotonic()
            decoded = backend.decoded_tokens
            backend.shift(64, len(backend.tokens) // 2)
            notice = backend.tokenize("\nOlder diagnostic context has been retired; continue.\n")
            backend.eval(notice)
            report["shift_seconds_including_pack"] = time.monotonic() - before
            report["tokens_evaluated_for_shift_notice"] = backend.decoded_tokens - decoded
            assert backend.decoded_tokens - decoded == len(notice)
        report["before_checkpoint"] = resources()
        before = time.monotonic()
        backend.save(checkpoint)
        report.update(save_seconds=time.monotonic() - before, checkpoint_tokens=len(backend.tokens),
                      checkpoint_bytes=sum(p.stat().st_size for p in checkpoint.iterdir()),
                      checkpoint_sha256=sha256_file(checkpoint / "state.bin"))
        generated, logits = [], []
        before = time.monotonic()
        for _ in range(args.steps):
            token = backend.sample()
            backend.eval([token])
            generated.append(token)
            logits.append(backend.logits.copy())
        report["continuation_seconds"] = time.monotonic() - before
        np.savez(folder / (checkpoint.name + "-reference.npz"), tokens=generated, logits=np.stack(logits))
        report["completed"] = True
    except BaseException as exc:
        report["error"] = repr(exc)
        raise
    finally:
        report["final_resources"] = resources()
        report["wall_seconds"] = time.monotonic() - started
        write_durable(folder / (stage + ".json"), report)
        if backend:
            backend.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=55000)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--stage", choices=["create", "restore", "restore-shifted"])
    args = parser.parse_args()
    if args.stage:
        worker(args)
        return
    config = Config.read(args.config)
    if not 256 <= args.tokens <= config.n_ctx - args.steps - 128:
        parser.error("occupied tokens must leave room for continuation")
    args.output.mkdir(parents=True, exist_ok=False)
    write_durable(args.output / "config.json", config.to_dict())
    write_durable(args.output / "baseline-resources.json", resources())
    reports = []
    try:
        for stage in ("create", "restore", "restore-shifted"):
            with (args.output / (stage + ".log")).open("w", encoding="utf-8") as log:
                proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--output", str(args.output.resolve()),
                    "--tokens", str(args.tokens), "--steps", str(args.steps), "--stage", stage],
                    stdout=log, stderr=log, timeout=5400,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            path = args.output / (stage + ".json")
            if path.exists():
                reports.append(json.loads(path.read_text()))
            if proc.returncode:
                raise RuntimeError(f"{stage} exited with {proc.returncode}; inspect its log")
        report = {"completed": True, "verified": all(r.get("continuation_verified", True) for r in reports),
            "occupied_context_tokens": args.tokens, "configured_context": config.n_ctx,
            "fresh_processes": 3, "primary_conversation_used": False, "stages": reports}
    except BaseException as exc:
        report = {"completed": False, "verified": False, "error": repr(exc), "stages": reports}
        raise
    finally:
        write_durable(args.output / "report.json", report)


if __name__ == "__main__":
    main()
