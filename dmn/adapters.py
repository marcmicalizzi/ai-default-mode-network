"""Explicit whole-context LoRA identities; no training or live weight switching."""
from __future__ import annotations

import ctypes as C
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AdapterSpec:
    path: str
    sha256: str
    base_model_sha256: str
    scale: float = 1.0

    def __post_init__(self):
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("adapter path must be nonempty")
        for value in (self.sha256, self.base_model_sha256):
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError("adapter and base identities must be lowercase SHA-256 hashes")
        if (type(self.scale) not in (int, float) or not math.isfinite(self.scale) or
                not math.isfinite(C.c_float(self.scale).value) or
                (self.scale != 0 and C.c_float(self.scale).value == 0)):
            raise ValueError("adapter scale must be finite and representable as nonzero float32 unless explicitly zero")

    def identity(self):
        return {"sha256": self.sha256, "base_model_sha256": self.base_model_sha256,
                "scale": C.c_float(self.scale).value, "activation": "whole_context"}


def attach_adapters(backend):
    """Apply in declared order before any eval/load. The native model owns handles."""
    from .backend import sha256_file

    specs = backend.config.lora_adapters
    if not specs:
        return
    api = backend.api
    required = ("llama_adapter_lora_init", "llama_set_adapters_lora",
                "llama_adapter_get_alora_n_invocation_tokens", "llama_adapter_lora_p_ctypes")
    if any(not hasattr(api, name) for name in required):
        raise ValueError("this native binding lacks the required whole-context LoRA API")
    handles = []
    for spec in specs:
        path = Path(spec.path)
        if spec.base_model_sha256 != backend.fingerprint["model_sha256"]:
            raise ValueError("adapter declared base model identity differs from the loaded GGUF")
        if sha256_file(path) != spec.sha256:
            raise ValueError("adapter file differs from its declared SHA-256 identity")
        handle = api.llama_adapter_lora_init(backend.model, os.fsencode(path))
        if not handle:
            raise ValueError("llama.cpp could not load the declared adapter")
        if api.llama_adapter_get_alora_n_invocation_tokens(handle):
            raise ValueError("invocation-gated aLoRA is not supported; whole-context adapters are required")
        # Detect ordinary replacement during loading, before any tokens are used.
        if sha256_file(path) != spec.sha256:
            raise ValueError("adapter changed while loading")
        handles.append(handle)
    pointers = (api.llama_adapter_lora_p_ctypes * len(handles))(*handles)
    scales = (C.c_float * len(specs))(*(spec.scale for spec in specs))
    if api.llama_set_adapters_lora(backend.ctx, pointers, len(handles), scales):
        raise ValueError("llama.cpp refused adapter activation")
    backend.fingerprint["lora_adapters"] = [spec.identity() for spec in specs]


def saved_adapter_identity(fingerprint, config):
    expected = [spec.identity() for spec in config.lora_adapters]
    if fingerprint.get("lora_adapters", []) != expected:
        raise ValueError("adapter fingerprint and declared configuration disagree")
    return expected
