"""Explicitly distinguish native restoration from reevaluating retained tokens."""
from __future__ import annotations

import json
from pathlib import Path

from .backend import sha256_file
from .config import Config
from .adapters import saved_adapter_identity


PLACEMENT_SETTINGS = {"model_path", "n_ctx", "n_batch", "n_gpu_layers", "n_threads",
                      "offload_kqv", "flash_attn", "type_k", "type_v", "swa_full"}
SCHEDULING_SETTINGS = {"token_delay_seconds", "checkpoint_tokens", "checkpoint_interval_seconds",
                       "checkpoint_policy", "suspend_preparation_seconds", "checkpoint_reserve_bytes",
                       "idle_enabled", "idle_max_burst_tokens", "idle_min_interval_seconds",
                       "sleep_checkpoint_min_interval_seconds"}
NATIVE_PLACEMENT_SETTINGS = {"n_threads", "n_gpu_layers"}


def same_native_environment(saved, current):
    # Pacing/checkpoint cadence never define model or KV compatibility. All
    # native binary, model, sampler and cache-layout checks remain strict.
    def normalized(value):
        config = Config(**value["config"])
        adapters = saved_adapter_identity(value, config)
        return {**value, "lora_adapters": adapters,
                "config": {**{key: item for key, item in config.to_dict().items()
                               if key not in SCHEDULING_SETTINGS}, "lora_adapters": adapters}}
    return normalized(saved) == normalized(current)


def reconstruction_compatible(saved: dict, current: dict):
    if saved.get("kind") != current.get("kind"):
        raise ValueError("context reconstruction cannot change backend type")
    identity = "model_sha256" if current["kind"] == "native_llama_kv" else "script_sha256"
    if not saved.get(identity) or saved[identity] != current.get(identity):
        raise ValueError("context reconstruction requires the same model/tokenizer identity")
    left_config, right_config = Config(**saved["config"]), Config(**current["config"])
    left_adapters = saved_adapter_identity(saved, left_config)
    right_adapters = saved_adapter_identity(current, right_config)
    if left_adapters != right_adapters or saved.get("research_lora") != current.get("research_lora"):
        raise ValueError("context reconstruction cannot change adapter identity; an explicit weight transition is required")
    left = {**left_config.to_dict(), "lora_adapters": left_adapters}
    right = {**right_config.to_dict(), "lora_adapters": right_adapters}
    differences = {key for key in left.keys() | right.keys() if left.get(key) != right.get(key)}
    if differences - PLACEMENT_SETTINGS - SCHEDULING_SETTINGS:
        raise ValueError("context reconstruction cannot silently change protocol or sampler settings: "
                         + ", ".join(sorted(differences - PLACEMENT_SETTINGS - SCHEDULING_SETTINGS)))


def native_placement_changes(saved, current):
    """A narrow opt-in; cache layout, native build, model and sampler stay strict."""
    if saved.get("kind") != "native_llama_kv" or current.get("kind") != "native_llama_kv":
        raise ValueError("placement changes require a native llama checkpoint")
    left, right = Config(**saved["config"]).to_dict(), Config(**current["config"]).to_dict()
    adjusted = {**current, "config": {**right, **{key: left[key] for key in NATIVE_PLACEMENT_SETTINGS}}}
    if not same_native_environment(saved, adjusted):
        raise ValueError("placement restore permits only n_threads and n_gpu_layers; other inference settings differ")
    return {key: {"previous": left[key], "current": right[key]}
            for key in sorted(NATIVE_PLACEMENT_SETTINGS) if left[key] != right[key]}


def restore_checkpoint(backend, directory: Path, policy="strict", allow_placement_change=False):
    if policy not in {"strict", "fallback", "rebuild"}:
        raise ValueError("unknown KV recovery policy")
    if allow_placement_change and policy != "strict":
        raise ValueError("placement changes require strict recovery; reconstruction is not permitted")
    manifest = json.loads((directory / "manifest.json").read_text())
    saved_config = manifest["fingerprint"]["config"]
    if saved_config.get("multi_user") and "require_contact_consent" not in saved_config:
        raise ValueError("this earlier multi-user prototype has no saved consent contract; "
                         "use a fresh instance until deliberate migration is implemented")
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
    placement = (native_placement_changes(manifest["fingerprint"], backend.fingerprint)
                 if allow_placement_change else None)
    if placement is None and not same_native_environment(manifest["fingerprint"], backend.fingerprint):
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
            if allow_placement_change:
                if evidence.get("prompt_tokens_reevaluated") != 0 or evidence.get("decode_calls_during_load") != 0:
                    raise RuntimeError("placement restore attempted prompt evaluation")
                verification = backend.verify_loaded_snapshot(directory, files["state.bin"])
                evidence.update(verification, placement_changes=placement)
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
