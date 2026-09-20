"""Explicitly distinguish native restoration from reevaluating retained tokens."""
from __future__ import annotations

import json
from pathlib import Path

from .backend import sha256_file
from .config import Config


PLACEMENT_SETTINGS = {"model_path", "n_ctx", "n_batch", "n_gpu_layers", "n_threads",
                      "offload_kqv", "flash_attn", "type_k", "type_v", "swa_full"}
SCHEDULING_SETTINGS = {"token_delay_seconds", "checkpoint_tokens", "checkpoint_interval_seconds",
                       "checkpoint_policy", "suspend_preparation_seconds", "checkpoint_reserve_bytes"}


def same_native_environment(saved, current):
    # Pacing/checkpoint cadence never define model or KV compatibility. All
    # native binary, model, sampler and cache-layout checks remain strict.
    def normalized(value):
        return {**value, "config": {key: item for key, item in Config(**value["config"]).to_dict().items()
                                   if key not in SCHEDULING_SETTINGS}}
    return normalized(saved) == normalized(current)


def reconstruction_compatible(saved: dict, current: dict):
    if saved.get("kind") != current.get("kind"):
        raise ValueError("context reconstruction cannot change backend type")
    identity = "model_sha256" if current["kind"] == "native_llama_kv" else "script_sha256"
    if not saved.get(identity) or saved[identity] != current.get(identity):
        raise ValueError("context reconstruction requires the same model/tokenizer identity")
    left, right = Config(**saved["config"]).to_dict(), Config(**current["config"]).to_dict()
    differences = {key for key in left.keys() | right.keys() if left.get(key) != right.get(key)}
    if differences - PLACEMENT_SETTINGS - SCHEDULING_SETTINGS:
        raise ValueError("context reconstruction cannot silently change protocol or sampler settings: "
                         + ", ".join(sorted(differences - PLACEMENT_SETTINGS - SCHEDULING_SETTINGS)))


def restore_checkpoint(backend, directory: Path, policy="strict"):
    if policy not in {"strict", "fallback", "rebuild"}:
        raise ValueError("unknown KV recovery policy")
    manifest = json.loads((directory / "manifest.json").read_text())
    files = manifest["files"]
    if not {"runtime.json", "engine.json"} <= files.keys():
        raise ValueError("checkpoint integrity metadata is incomplete")
    corrupt = []
    for name, digest in files.items():
        if Path(name).name != name:
            raise ValueError("invalid checkpoint file path")
        path = directory / name
        if not path.is_file() or sha256_file(path) != digest:
            corrupt.append(name)
    # The token sequence, RNG and runtime state are the authoritative textual
    # recovery sidecar. Never infer them from a corrupt binary or UI transcript.
    if any(name not in {"state.bin", "logits.npy"} for name in corrupt):
        raise ValueError("checkpoint file integrity check failed: " + ", ".join(corrupt))
    state = json.loads((directory / "runtime.json").read_text())
    if state.get("schema") != 1:
        raise ValueError("unsupported checkpoint schema")
    reasons = []
    if not same_native_environment(manifest["fingerprint"], backend.fingerprint):
        reasons.append("inference environment differs from checkpoint")
    if corrupt:
        reasons.append("checkpoint file integrity check failed: " + ", ".join(corrupt))
    if backend.kind == "native_llama_kv" and not {"state.bin", "logits.npy"} <= files.keys():
        reasons.append("native snapshot files missing from manifest")
    if policy == "strict" and reasons:
        raise ValueError("; ".join(reasons) + "; refusing silent reconstruction")
    if policy != "rebuild" and not reasons:
        try:
            evidence = backend.load(directory)
            saved_config = Config(**manifest["fingerprint"]["config"]).to_dict()
            current_config = Config(**backend.fingerprint["config"]).to_dict()
            evidence["scheduling_changes"] = {
                key: {"previous": saved_config[key], "current": current_config[key]}
                for key in SCHEDULING_SETTINGS
                if saved_config[key] != current_config[key]
            }
            return state, {**evidence, "method": "native_restore" if backend.kind == "native_llama_kv" else "demo_restore"}
        except (RuntimeError, ValueError, OSError) as exc:
            if policy == "strict":
                raise
            reasons.append("native loader failed: " + str(exc))
    reconstruction_compatible(manifest["fingerprint"], backend.fingerprint)
    evidence = backend.rebuild(directory)
    return state, {**evidence, "method": "retained_token_reconstruction",
                   "reason": "; ".join(reasons) or "explicit rebuild requested",
                   "prior_context_retirements": state.get("context_retirements", 0),
                   "exact_kv_continuity": False,
                   "limitation": "Retained tokens are reevaluated. Past attention to evicted tokens is not recreated."}
