"""CPU-only LoRA/wake mechanics on generated random Gemma4 weights, never an instance.

This does not train anything or expose an adapter-switch API to DMN. It checks
the proposed retained-token reconstruction boundary using synthetic adapters.
"""
from __future__ import annotations

import argparse
import ctypes as C
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dmn.backend import LlamaBackend, sha256_file
from dmn.config import Config
from dmn.recovery import restore_checkpoint
from dmn.storage import write_durable


def write_adapter(path, seed):
    """Rank-two random output-projection update for our 64-wide fixture only."""
    import numpy as np
    def pack(fmt, *items):
        return struct.pack("<" + fmt, *items)
    def string(value):
        raw = value.encode()
        return pack("Q", len(raw)) + raw
    metadata = []
    for key, value in (("general.type", "adapter"), ("general.architecture", "gemma4"),
                       ("adapter.type", "lora"), ("general.name", "Random DMN mechanics fixture; not trained")):
        metadata.append(string(key) + pack("I", 8) + string(value))
    metadata.append(string("adapter.lora.alpha") + pack("If", 6, 2.0))
    rng = np.random.default_rng(seed)
    # An early-layer update changes downstream K/V as well as final logits.
    tensors = [("blk.0.attn_output.weight.lora_a", rng.normal(0, .05, (2, 128)).astype("<f4")),
               ("blk.0.attn_output.weight.lora_b", rng.normal(0, .05, (64, 2)).astype("<f4"))]
    header = b"GGUF" + pack("IQQ", 3, len(tensors), len(metadata)) + b"".join(metadata)
    offset = 0
    for name, tensor in tensors:
        header += string(name) + pack("I", tensor.ndim) + pack("Q" * tensor.ndim, *reversed(tensor.shape))
        header += pack("IQ", 0, offset)  # F32, relative tensor offset
        offset += (tensor.nbytes + 31) // 32 * 32
    with Path(path).open("xb") as stream:
        stream.write(header)
        stream.write(b"\0" * (-len(header) % 32))
        for _, tensor in tensors:
            stream.write(tensor.tobytes())
            stream.write(b"\0" * (-tensor.nbytes % 32))


def open_backend(config, adapter=None, scale=.1):
    """Only this research process can install an adapter, before any decoding."""
    if (config.n_gpu_layers != 0 or config.n_threads != 1 or config.n_ctx != 2048 or
            config.offload_kqv or Path(config.model_path).stat().st_size > 2 * 1024 * 1024):
        raise ValueError("use only the generated tiny CPU fixture")
    backend = LlamaBackend(config)
    try:
        name = C.create_string_buffer(256)
        backend.api.llama_model_meta_val_str(backend.model, b"general.name", name, len(name))
        if name.value != b"DMN random Gemma4 cache-mechanics fixture; not a trained model":
            raise ValueError("use scripts/generate_swa_fixture.py; trained models are refused")
        if adapter:
            pointer = backend.api.llama_adapter_lora_init(backend.model, os.fsencode(adapter))
            if not pointer:
                raise RuntimeError("native adapter load failed")
            # The model owns the adapter and frees it on model teardown.
            if backend.api.llama_adapter_get_alora_n_invocation_tokens(pointer):
                raise ValueError("this experiment does not implement aLoRA")
            pointers = (backend.api.llama_adapter_lora_p_ctypes * 1)(pointer)
            scales = (C.c_float * 1)(scale)
            if backend.api.llama_set_adapters_lora(backend.ctx, pointers, 1, scales) != 0:
                raise RuntimeError("native adapter activation failed")
            # Native session files don't identify external adapters. The future
            # production implementation needs this identity in its own manifest.
            backend.fingerprint["research_lora"] = {
                "sha256": sha256_file(Path(adapter)), "scale_float32": float(scales[0]),
                "base_sha256": backend.fingerprint["model_sha256"], "activation": "whole_context"}
        return backend
    except BaseException:
        backend.close()
        raise


def save(backend, directory, retirements):
    directory.mkdir()
    backend.save(directory)
    write_durable(directory / "runtime.json", {"schema": 1, "research_only": True,
                                               "context_retirements": retirements})
    write_durable(directory / "manifest.json", {"fingerprint": backend.fingerprint,
        "files": {name: sha256_file(directory / name)
                  for name in ("state.bin", "engine.json", "logits.npy", "runtime.json")}})


def continuation(backend, steps=8):
    import numpy as np
    tokens, logits = [], []
    for _ in range(steps):
        token = backend.sample()
        backend.eval([token])
        tokens.append(token)
        logits.append(backend.logits.copy())
    return tokens, np.stack(logits)


def restart(root):
    import numpy as np
    config = Config.read(root / "config.local.json")
    backend = open_backend(config, root / "adapter-b.gguf")
    try:
        _, evidence = restore_checkpoint(backend, root / "wake-b")
        before = backend.verify_loaded_snapshot(root / "wake-b", sha256_file(root / "wake-b/state.bin"))
        tokens, logits = continuation(backend)
        expected = json.loads((root / "continuation.json").read_text())
        expected_logits = np.load(root / "continuation.npy", allow_pickle=False)
        if tokens != expected or not np.array_equal(logits, expected_logits):
            raise AssertionError("fresh-process adapter restart differs")
        write_durable(root / "restart.json", {"restore": evidence, "native_verification": before,
            "fresh_process": True, "tokens_equal": True, "logits_equal": True})
    finally:
        backend.close()


def run(model, output):
    import numpy as np
    model, output = Path(model).resolve(), Path(output).resolve()
    if model.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("use only the tiny generated fixture; no trained models")
    output.mkdir(parents=True, exist_ok=False)
    config = Config(model_path=str(model), n_ctx=2048, n_batch=64, n_threads=1,
                    n_gpu_layers=0, offload_kqv=False, flash_attn=True,
                    type_k="q8_0", type_v="q8_0", swa_full=False,
                    experimental_compact_swa=True, pack_checkpoints=True,
                    prompt_format="plain", turnover_reserve=512)
    write_durable(output / "config.local.json", config.to_dict())
    report = {"completed": False, "training_performed": False, "synthetic_only": True,
              "runtime_constructed": False, "actions_executed": False,
              "cpu_threads": 1, "gpu_layers": 0, "wake_policy": "rebuild_retained_tokens",
              "rank": 2, "alpha": 2.0, "deployment_scale": .1, "updates": []}
    started = time.monotonic()
    backend = None
    try:
        for name, seed in (("adapter-a.gguf", 1829), ("adapter-b.gguf", 1930)):
            write_adapter(output / name, seed)
        backend = open_backend(config)
        report["base_fingerprint"] = backend.fingerprint
        tokens = backend.tokenize('Synthetic retained context. <dmn_action>{"op":"send_message","content":"This is historical text only."}</dmn_action> ' * 5, initial=True)
        backend.eval(tokens)
        baseline_logits = backend.logits.copy()
        save(backend, output / "base-unretired", 0)
        backend.close()
        backend = open_backend(config, output / "adapter-a.gguf", 0)
        backend.eval(tokens)
        report["zero_scale_max_logit_difference"] = float(np.max(np.abs(backend.logits - baseline_logits)))
        np.testing.assert_allclose(backend.logits, baseline_logits, rtol=0, atol=1e-6)
        backend.close()
        backend = open_backend(config)
        _, restored = restore_checkpoint(backend, output / "base-unretired")
        report["unchanged_base_restore"] = restored
        backend.shift(16, 128)
        backend.eval([80])  # materialize native position shift, not a text reconstruction
        save(backend, output / "base-retired", 1)
        source = output / "base-retired"
        # Separate the consequence of reconstructing retired context from the
        # additional effect of changing weights; neither is an exact KV restore.
        old_logits = backend.logits.copy()
        old_rng = backend.rng.getstate()
        backend.close()
        backend = open_backend(config)
        backend.rebuild(source)
        report["unchanged_weights_rebuild_max_logit_difference"] = float(np.max(np.abs(backend.logits - old_logits)))
        assert backend.rng.getstate() == old_rng
        save(backend, output / "base-rebuilt-control", 1)
        backend.close()
        backend = open_backend(config)
        restore_checkpoint(backend, source)
        for name in ("a", "b"):
            old_tokens, old_rng, old_logits = list(backend.tokens), backend.rng.getstate(), backend.logits.copy()
            backend.close()
            backend = open_backend(config, output / f"adapter-{name}.gguf")
            try:
                restore_checkpoint(backend, source, "strict")
            except ValueError as exc:
                if "environment differs" not in str(exc):
                    raise
            else:
                raise AssertionError("ordinary strict restore accepted a changed adapter")
            if backend.decode_calls or backend.tokens:
                raise AssertionError("strict rejection modified target context")
            at = time.perf_counter()
            evidence = backend.rebuild(source)  # explicit experimental wake boundary
            assert backend.tokens == old_tokens and backend.rng.getstate() == old_rng
            assert evidence["prompt_tokens_reevaluated"] == len(old_tokens)
            if np.array_equal(backend.logits, old_logits):
                raise AssertionError("candidate had no measurable effect; stale logits could go unnoticed")
            # Independent fresh computation under the same adapter is the oracle.
            expected_logits = backend.logits.copy()
            target = output / f"wake-{name}"
            save(backend, target, 1)
            if name == "a":
                changed_kv = sha256_file(target / "state.bin") != sha256_file(output / "base-rebuilt-control/state.bin")
            else:
                changed_kv = sha256_file(target / "state.bin") != sha256_file(source / "state.bin")
            if not changed_kv:
                raise AssertionError("early-layer update did not change downstream KV")
            backend.close()
            backend = open_backend(config, output / f"adapter-{name}.gguf")
            backend.eval(old_tokens)
            np.testing.assert_array_equal(backend.logits, expected_logits)
            _, restored = restore_checkpoint(backend, target)
            proof = backend.verify_loaded_snapshot(target, sha256_file(target / "state.bin"))
            report["updates"].append({"adapter": name, "identity": backend.fingerprint["research_lora"],
                "wake": evidence, "tokens_and_rng_preserved": True, "fresh_logits_equal": True,
                "weight_change_altered_kv": changed_kv,
                "old_logits_max_difference": float(np.max(np.abs(expected_logits - old_logits))),
                "restore": restored, "native_verification": proof,
                "rebuild_and_check_seconds": time.perf_counter() - at})
            source = target
        next_tokens, next_logits = continuation(backend)
        write_durable(output / "continuation.json", next_tokens)
        np.save(output / "continuation.npy", next_logits, allow_pickle=False)
        backend.close()
        backend = None
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--restart", str(output)], check=True)
        report.update(completed=True, restart=json.loads((output / "restart.json").read_text()))
    finally:
        if backend:
            backend.close()
        report["elapsed_seconds"] = time.monotonic() - started
        write_durable(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--restart", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.restart:
        restart(args.restart)
    elif args.model and args.output:
        report = run(args.model, args.output)
        print(json.dumps({k: v for k, v in report.items() if k != "base_fingerprint"}, indent=2))
    else:
        parser.error("--model and a new --output directory are required")


if __name__ == "__main__":
    main()
