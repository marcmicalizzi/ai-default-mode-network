from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class Config:
    backend: str = "llama"
    model_path: str = ""
    n_ctx: int = 8192
    n_batch: int = 256
    n_gpu_layers: int = 0
    n_threads: int = 4
    offload_kqv: bool = True
    flash_attn: bool = False
    type_k: str = "f16"
    type_v: str = "f16"
    swa_full: bool = True  # Preserve the pinned binding's original default.
    pack_checkpoints: bool = False
    prompt_format: str = "model"
    jinja_thinking: bool = False
    system_prompt: str = ""
    seed: int = 42
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    min_p: float = 0.05
    repeat_penalty: float = 1.05
    repeat_last_n: int = 64
    sampler_order: str = "legacy_v1"
    checkpoint_tokens: int = 512
    checkpoint_interval_seconds: float = 0.0  # 0 disables the time threshold.
    checkpoint_policy: str = "all_actions"  # "effects" still commits effects with KV.
    checkpoint_reserve_bytes: int = 256 * 1024 * 1024
    suspend_preparation_seconds: float | None = None  # None retains the token-only bound.
    clock_interval_seconds: float = 60.0
    token_delay_seconds: float = 0.0
    turnover_reserve: int = 1024
    preparation_tokens: int = 128
    keep_prefix_tokens: int = 0  # 0 means keep the entire initialization prefix.
    max_action_bytes: int = 8192
    max_event_bytes: int = 16384

    def __post_init__(self):
        if self.backend not in {"llama", "demo"}:
            raise ValueError("backend must be llama or demo")
        if self.sampler_order not in {"legacy_v1", "llama_default_v1"}:
            raise ValueError("unsupported sampler_order")
        if self.prompt_format not in {"model", "plain", "jinja"}:
            raise ValueError("prompt_format must be model, jinja or plain")
        if any(type(value) is not bool for value in (self.swa_full, self.jinja_thinking, self.pack_checkpoints)):
            raise ValueError("swa_full, jinja_thinking and pack_checkpoints must be booleans")
        if self.n_ctx < 2048 or not 1 <= self.n_batch <= self.n_ctx:
            raise ValueError("n_ctx must be >= 2048; n_batch must fit in n_ctx")
        if not 256 <= self.turnover_reserve <= self.n_ctx // 2:
            raise ValueError("turnover_reserve must be between 256 and half n_ctx")
        if not 1 <= self.preparation_tokens <= self.turnover_reserve // 2:
            raise ValueError("preparation_tokens must fit in half turnover_reserve")
        if not 0 <= self.keep_prefix_tokens < self.n_ctx - self.turnover_reserve:
            raise ValueError("keep_prefix_tokens leaves insufficient context")
        if self.type_k not in {"f16", "q8_0", "q4_0"} or self.type_v not in {"f16", "q8_0", "q4_0"}:
            raise ValueError("KV types supported: f16, q8_0, q4_0")
        if self.type_v != "f16" and not self.flash_attn:
            raise ValueError("quantized V cache requires flash_attn")
        if not 0 < self.top_p <= 1 or not 0 <= self.min_p <= 1:
            raise ValueError("invalid top_p/min_p")
        if self.top_k < 0 or self.repeat_last_n < 0 or self.repeat_penalty <= 0:
            raise ValueError("invalid sampler configuration")
        if type(self.checkpoint_tokens) is not int or self.checkpoint_tokens < 0 or self.n_threads < 1:
            raise ValueError("checkpoint_tokens must be a nonnegative integer; n_threads must be positive")
        if not isinstance(self.checkpoint_policy, str) or self.checkpoint_policy not in {"all_actions", "effects"}:
            raise ValueError("checkpoint_policy must be all_actions or effects")
        if type(self.checkpoint_reserve_bytes) is not int or self.checkpoint_reserve_bytes < 0:
            raise ValueError("checkpoint_reserve_bytes must be a nonnegative integer")
        if (isinstance(self.checkpoint_interval_seconds, bool) or
                not isinstance(self.checkpoint_interval_seconds, (int, float)) or
                not math.isfinite(self.checkpoint_interval_seconds) or self.checkpoint_interval_seconds < 0):
            raise ValueError("checkpoint_interval_seconds must be finite and nonnegative")
        if not self.checkpoint_tokens and not self.checkpoint_interval_seconds:
            raise ValueError("at least one periodic checkpoint threshold must be enabled")
        if self.suspend_preparation_seconds is not None and (
                isinstance(self.suspend_preparation_seconds, bool) or
                not isinstance(self.suspend_preparation_seconds, (int, float)) or
                not math.isfinite(self.suspend_preparation_seconds) or self.suspend_preparation_seconds < 0):
            raise ValueError("suspend_preparation_seconds must be finite and nonnegative, or null")
        if self.max_action_bytes < 128 or self.max_event_bytes < 128:
            raise ValueError("action/event limits must be >= 128 bytes")
        for name in ("temperature", "clock_interval_seconds", "token_delay_seconds", "repeat_penalty"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {name}")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def read(cls, path: Path):
        obj = json.loads(path.read_text(encoding="utf-8"))
        unknown = obj.keys() - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
        if obj.get("model_path"):
            obj["model_path"] = str((path.resolve().parent / obj["model_path"]).resolve())
        return cls(**obj)
