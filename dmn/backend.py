from __future__ import annotations

import ctypes as C
from contextlib import contextmanager
import hashlib
import json
import mmap
import os
import platform
import random
import tempfile
from pathlib import Path

from .config import Config
from .diskspace import check_space
from .storage import write_durable


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tuples(value):
    return tuple(tuples(x) for x in value) if isinstance(value, list) else value


def top_k_candidates(np, scores, top_k):
    """Match stable full-sort ordering, including ties at the top-k boundary."""
    if not top_k or top_k >= len(scores):
        return np.argsort(-scores, kind="stable")
    cutoff = np.partition(scores, len(scores) - top_k)[len(scores) - top_k]
    greater = np.flatnonzero(scores > cutoff)
    tied = np.flatnonzero(scores == cutoff)[:top_k - len(greater)]
    candidates = np.concatenate((greater, tied))
    # flatnonzero gives ascending token IDs for equal scores. Keep that order
    # exactly as the former stable vocabulary-wide sort did.
    return candidates[np.argsort(-scores[candidates], kind="stable")]


class LlamaBackend:
    """One model, one native context, sequence 0. Never calls a completion API.

    All access belongs to the runtime inference thread. Sampling lives here so
    the exact RNG state is serializable independently of llama.cpp's samplers.
    """
    kind = "native_llama_kv"
    native_layout_policy = "pack_after_retirement_v1"
    pack_memory_limit_bytes = 256 * 1024 * 1024

    def __init__(self, config: Config):
        import llama_cpp as package
        import llama_cpp.llama_cpp as api
        import numpy as np
        from .native_logging import configure_native_logging

        configure_native_logging(api)
        self.api, self.np, self.config = api, np, config
        if config.pack_checkpoints:
            self.native_layout_policy = "pack_after_retirement_and_before_checkpoint_v1"
        self.model = self.ctx = self.batch = None
        self.tokens = []
        self.decoded_tokens = 0
        self.decode_calls = 0
        self.layout_packs = 0
        self._layout_pending = False
        self._state_work_dir = None
        self.storage_guard = self._check_storage
        self.logits = None
        self.rng = random.Random(config.seed)
        model_path = Path(config.model_path).resolve()
        if not model_path.is_file():
            raise ValueError(f"GGUF file not found: {model_path}")
        if "-of-" in model_path.stem:
            raise ValueError("split GGUF is not supported yet; use a single GGUF so its complete identity can be checked")
        required = ("llama_memory_can_shift", "llama_state_save_file", "llama_state_load_file",
                    "llama_model_get_vocab", "llama_memory_seq_pos_max", "llama_state_get_size",
                    "llama_state_get_data", "llama_state_set_data")
        if any(not hasattr(api, name) for name in required):
            raise RuntimeError("unsupported llama-cpp-python build; install the version pinned by this project")
        binary_root = Path(api._lib._name).resolve().parent
        binaries = {str(p.relative_to(binary_root)): sha256_file(p)
                    for p in sorted(binary_root.rglob("*"))
                    if p.is_file() and p.suffix in {".dll", ".so", ".dylib"}}
        if not binaries:
            raise RuntimeError("cannot fingerprint the llama.cpp native build")
        self.fingerprint = {"kind": self.kind, "model_sha256": sha256_file(model_path),
                            "binding_version": package.__version__, "native_binaries": binaries,
                            "binding_source_sha256": sha256_file(Path(api.__file__)),
                            "platform": platform.platform(), "machine": platform.machine(),
                            "numpy_version": np.__version__, "config": config.to_dict()}
        api.llama_backend_init()
        try:
            mp = api.llama_model_default_params()
            mp.n_gpu_layers = 0x7FFFFFFF if config.n_gpu_layers == -1 else config.n_gpu_layers
            self.model = api.llama_model_load_from_file(os.fsencode(model_path), mp)
            if not self.model:
                raise RuntimeError("llama.cpp could not load the model")
            self.vocab = api.llama_model_get_vocab(self.model)
            cp = api.llama_context_default_params()
            cp.n_ctx, cp.n_batch, cp.n_ubatch = config.n_ctx, config.n_batch, config.n_batch
            cp.n_seq_max = 1
            cp.n_threads = cp.n_threads_batch = config.n_threads
            cp.offload_kqv = config.offload_kqv
            cp.swa_full = config.swa_full
            cp.flash_attn_type = (api.LLAMA_FLASH_ATTN_TYPE_ENABLED if config.flash_attn
                                  else api.LLAMA_FLASH_ATTN_TYPE_DISABLED)
            cp.type_k = getattr(api, "GGML_TYPE_" + config.type_k.upper())
            cp.type_v = getattr(api, "GGML_TYPE_" + config.type_v.upper())
            self.ctx = api.llama_init_from_model(self.model, cp)
            if not self.ctx:
                raise RuntimeError("llama.cpp could not create the context")
            self.n_ctx = int(api.llama_n_ctx(self.ctx))
            self.n_vocab = int(api.llama_vocab_n_tokens(self.vocab))
            self.batch = api.llama_batch_init(config.n_batch, 0, 1)
            self.memory = api.llama_get_memory(self.ctx)
            self.can_shift = bool(api.llama_memory_can_shift(self.memory))
            self.fingerprint["actual_n_ctx"] = self.n_ctx
            self.fingerprint["system_info"] = api.llama_print_system_info().decode("utf-8", "replace")
            template = api.llama_model_chat_template(self.model, None)
            self.template = template.decode("utf-8") if template else None
            self.fingerprint["chat_template"] = self.template
            if config.prompt_format == "jinja":
                import jinja2
                import llama_cpp.llama_chat_format as formatter
                self.fingerprint["seed_renderer"] = {"engine": "Jinja2ChatFormatter",
                    "jinja2_version": jinja2.__version__, "formatter_sha256": sha256_file(Path(formatter.__file__))}
        except BaseException:
            self.close()
            raise

    def render_seed(self, text: str):
        if self.config.prompt_format == "plain":
            return text + "\n<internal_cognition>\n"
        if not self.template:
            raise ValueError("model has no chat template; explicitly select prompt_format=plain")
        if self.config.prompt_format == "jinja":
            from llama_cpp.llama_chat_format import Jinja2ChatFormatter
            def special_piece(token):
                if token < 0:
                    return ""
                size = 128
                while True:
                    output = C.create_string_buffer(size)
                    length = self.api.llama_token_to_piece(self.vocab, token, output, size, 0, True)
                    if length >= 0:
                        return output.raw[:length].decode("utf-8")
                    size = -length
            formatter = Jinja2ChatFormatter(self.template,
                eos_token=special_piece(self.api.llama_vocab_eos(self.vocab)),
                # The tokenizer inserts BOS when the vocabulary requests it.
                # Passing it to Jinja as well would produce a double BOS.
                bos_token="" if self.api.llama_vocab_get_add_bos(self.vocab) else special_piece(self.api.llama_vocab_bos(self.vocab)))
            return formatter(messages=[{"role": "user", "content": text}],
                             enable_thinking=self.config.jinja_thinking).prompt
        raw = text.encode("utf-8")
        messages = (self.api.llama_chat_message * 1)()
        messages[0].role, messages[0].content = b"user", raw
        size = max(4096, len(raw) * 2)
        while True:
            buffer = C.create_string_buffer(size)
            n = self.api.llama_chat_apply_template(self.template.encode(), messages, 1, True, buffer, size)
            if n < 0:
                raise ValueError("native chat template unsupported; supply a complete seed with prompt_format=plain")
            if n < size:
                return buffer.raw[:n].decode("utf-8")
            size = n + 1

    def tokenize(self, text: str, initial=False):
        raw = text.encode("utf-8")
        size = max(32, len(raw) + 16)
        while True:
            output = (self.api.llama_token * size)()
            n = self.api.llama_tokenize(self.vocab, raw, len(raw), output, size, initial, initial)
            if n >= 0:
                return list(output[:n])
            size = -n

    def eval(self, tokens):
        if len(self.tokens) + len(tokens) > self.n_ctx:
            raise ValueError("append would overflow context")
        for start in range(0, len(tokens), self.config.n_batch):
            part = tokens[start:start + self.config.n_batch]
            self.batch.n_tokens = len(part)
            for i, token in enumerate(part):
                self.batch.token[i] = token
                self.batch.pos[i] = len(self.tokens) + i
                self.batch.n_seq_id[i] = 1
                self.batch.seq_id[i][0] = 0
                self.batch.logits[i] = i == len(part) - 1
            code = self.api.llama_decode(self.ctx, self.batch)
            self.decode_calls += 1
            if code:
                raise RuntimeError(f"llama_decode failed ({code}); resume the last committed checkpoint")
            self.tokens.extend(part)
            self.decoded_tokens += len(part)
            pointer = self.api.llama_get_logits_ith(self.ctx, -1)
            if not pointer:
                raise RuntimeError("llama.cpp did not return logits")
            self.logits = self.np.ctypeslib.as_array(pointer, shape=(self.n_vocab,)).copy()
        if tokens and self._layout_pending:
            self._pack_native_layout()

    def _pack_native_layout(self):
        # Retirement leaves physical holes. Native restoration packs occupied
        # cells, which can change CUDA attention arithmetic despite preserving
        # every serialized K/V value. Pack at the retirement boundary so live
        # continuation and later restoration use the same layout. This copies
        # native state through host memory; it never evaluates retained tokens.
        # Call only AFTER decode has applied pending RoPE position shifts.
        size = self.api.llama_state_get_size(self.ctx)
        if size <= 0:
            raise RuntimeError("cannot size native state for cache packing")
        with self._state_buffer(size) as buffer:
            if self.api.llama_state_get_data(self.ctx, buffer, size) != size:
                raise RuntimeError("native cache packing could not capture complete state")
            if self.api.llama_state_set_data(self.ctx, buffer, size) != size:
                raise RuntimeError("native cache packing could not restore complete state")
        if self.api.llama_memory_seq_pos_max(self.memory, 0) != len(self.tokens) - 1:
            raise RuntimeError("native cache packing changed retained positions")
        self.layout_packs += 1
        self._layout_pending = False

    @contextmanager
    def _state_buffer(self, size):
        # Full SWA at 60K needs tens of GiB even with Q8. A second private RAM
        # allocation can exceed the host's commit limit. File-backed mapping
        # presents the same contiguous C buffer while allowing the OS to evict
        # pages to an owned temporary file beside the checkpoints.
        if size <= self.pack_memory_limit_bytes:
            self.last_layout_pack_storage = "memory"
            yield (C.c_uint8 * size)()
            return
        self.last_layout_pack_storage = "temporary_file_mapping"
        self.storage_guard(Path(self._state_work_dir or tempfile.gettempdir()), size, "native packing scratch")
        with tempfile.TemporaryFile(prefix=".dmn-pack-", dir=self._state_work_dir) as spill:
            spill.truncate(size)
            with mmap.mmap(spill.fileno(), size, access=mmap.ACCESS_WRITE) as mapped:
                anchor = C.c_uint8.from_buffer(mapped)
                pointer = C.cast(C.addressof(anchor), C.POINTER(C.c_uint8))
                # Do not retain an exported Python buffer when closing mmap.
                del anchor
                yield pointer

    def sample(self):
        if self.logits is None:
            raise RuntimeError("cannot sample without logits")
        np, c = self.np, self.config
        scores = self.logits.astype(np.float64, copy=True)
        if not np.isfinite(scores).any() or np.isnan(scores).any():
            raise RuntimeError("invalid logits")
        if c.repeat_last_n:
            for token in set(self.tokens[-c.repeat_last_n:]):
                scores[token] = scores[token] / c.repeat_penalty if scores[token] > 0 else scores[token] * c.repeat_penalty
        if c.temperature == 0:
            return int(np.argmax(scores))
        order = top_k_candidates(np, scores, c.top_k)
        # Preserve existing instances' ordering. llama-server's default chain
        # applies top-p/min-p before temperature, unlike the original DMN path.
        values = scores[order] / (c.temperature if c.sampler_order == "legacy_v1" else 1.0)
        probs = np.exp(values - values[0])
        probs /= probs.sum()
        keep = max(1, int(np.searchsorted(np.cumsum(probs), c.top_p, side="left")) + 1)
        order, probs = order[:keep], probs[:keep]
        selected = probs >= probs[0] * c.min_p
        order, probs = order[selected], probs[selected]
        if c.sampler_order == "llama_default_v1":
            values = scores[order] / c.temperature
            probs = np.exp(values - values[0])
        target = self.rng.random() * float(probs.sum())
        return int(order[min(int(np.searchsorted(np.cumsum(probs), target, side="right")), len(order)-1)])

    def piece(self, token):
        size = 128
        while True:
            buffer = C.create_string_buffer(size)
            n = self.api.llama_token_to_piece(self.vocab, token, buffer, size, 0, False)
            if n >= 0:
                return buffer.raw[:n]
            size = -n

    def is_eog(self, token):
        return bool(self.api.llama_vocab_is_eog(self.vocab, token))

    def shift(self, keep, discard):
        if not self.can_shift:
            raise RuntimeError("this model's native memory cannot shift")
        if not 0 <= keep < keep + discard < len(self.tokens):
            raise ValueError("invalid context retirement range")
        if not self.api.llama_memory_seq_rm(self.memory, 0, keep, keep + discard):
            raise RuntimeError("llama.cpp refused partial KV removal")
        self.api.llama_memory_seq_add(self.memory, 0, keep + discard, len(self.tokens), -discard)
        del self.tokens[keep:keep + discard]
        # RoPE shift is applied by the next decode. Never sample stale logits.
        self.logits = None
        self._layout_pending = True

    def _check_storage(self, path, size, purpose):
        check_space(path, size, self.config.checkpoint_reserve_bytes, purpose)

    def checkpoint_size_bytes(self):
        size = self.api.llama_state_get_size(self.ctx)
        if size <= 0:
            raise RuntimeError("cannot estimate native checkpoint size")
        # Native session token IDs, JSON token history, logits and fixed metadata.
        return size + len(self.tokens) * 32 + (0 if self.logits is None else self.logits.nbytes) + 1024 * 1024

    def save(self, directory: Path):
        if self.logits is None or self._layout_pending:
            raise RuntimeError("checkpoint requires a completed decode and cache-layout boundary")
        self._state_work_dir = directory.parent
        if self.config.pack_checkpoints:
            self._pack_native_layout()
        tokens = (self.api.llama_token * len(self.tokens))(*self.tokens)
        if not self.api.llama_state_save_file(self.ctx, os.fsencode(directory / "state.bin"), tokens, len(tokens)):
            raise RuntimeError("native state save failed")
        self.np.save(directory / "logits.npy", self.logits, allow_pickle=False)
        write_durable(directory / "engine.json", {"tokens": self.tokens, "rng": self.rng.getstate(),
                                                  "decoded_tokens": self.decoded_tokens,
                                                  "native_layout_policy": self.native_layout_policy,
                                                  "last_layout_pack_storage": getattr(self, "last_layout_pack_storage", None),
                                                  "layout_packs": self.layout_packs})

    def load(self, directory: Path):
        self._state_work_dir = directory.parent
        engine = json.loads((directory / "engine.json").read_text())
        tokens = (self.api.llama_token * self.n_ctx)()
        count = C.c_size_t()
        before = self.decode_calls
        if not self.api.llama_state_load_file(self.ctx, os.fsencode(directory / "state.bin"), tokens, self.n_ctx, C.byref(count)):
            raise RuntimeError("native state restore failed; transcript reconstruction was not attempted")
        if list(tokens[:count.value]) != engine["tokens"]:
            raise RuntimeError("native checkpoint tokens do not match metadata")
        if self.api.llama_memory_seq_pos_max(self.memory, 0) != len(engine["tokens"]) - 1:
            raise RuntimeError("restored KV positions do not match the saved sequence")
        self.tokens = engine["tokens"]
        self.rng.setstate(tuples(engine["rng"]))
        self.logits = self.np.load(directory / "logits.npy", allow_pickle=False)
        if self.logits.shape != (self.n_vocab,):
            raise RuntimeError("checkpoint logits shape mismatch")
        # No eval() call is permitted on this path.
        self.decoded_tokens = engine["decoded_tokens"]
        self.layout_packs = engine.get("layout_packs", 0)
        self._layout_pending = False
        return {"native_state_loaded": True, "prompt_tokens_reevaluated": 0,
                "restored_tokens": len(self.tokens), "decode_calls_during_load": self.decode_calls - before,
                "saved_native_layout_policy": engine.get("native_layout_policy", "legacy_unpacked_retirement"),
                "layout_packs": self.layout_packs}

    def close(self):
        if self.batch is not None:
            self.api.llama_batch_free(self.batch)
            self.batch = None
        if self.ctx:
            self.api.llama_free(self.ctx)
            self.ctx = None
        if self.model:
            self.api.llama_model_free(self.model)
            self.model = None

    def rebuild(self, directory: Path):
        engine = json.loads((directory / "engine.json").read_text())
        tokens = engine["tokens"]
        if not tokens or len(tokens) > self.n_ctx or any(type(t) is not int or not 0 <= t < self.n_vocab for t in tokens):
            raise ValueError("saved tokens cannot fit this model's vocabulary/context; no truncation performed")
        # Validate RNG before changing the context. A failed native loader may
        # have left a partial cache, so clear it even if the Python list is empty.
        self.rng.setstate(tuples(engine["rng"]))
        self.api.llama_memory_clear(self.memory, True)
        self.tokens, self.logits = [], None
        self._layout_pending = False
        self.layout_packs = 0
        self.decoded_tokens = engine["decoded_tokens"]
        self.eval(tokens)
        return {"native_state_loaded": False, "prompt_tokens_reevaluated": len(tokens),
                "restored_tokens": len(tokens), "sampler_rng_restored": True}


class DemoBackend:
    """A deterministic transport fixture, explicitly NOT a language model or KV proof."""
    kind = "demo_fixture_no_model"
    can_shift = True

    def __init__(self, config: Config, script: bytes | None = None):
        self.config, self.n_ctx = config, config.n_ctx
        self.tokens, self.decoded_tokens, self.index = [], 0, 0
        self.script = script or (b'Internal fixture text is not a user message.\n'
                                b'<dmn_action>{"op":"send_message","content":"Demo transport is running. This is a scripted fixture, not model cognition."}</dmn_action>\n'
                                b'<dmn_action>{"op":"sleep"}</dmn_action>\n')
        self.fingerprint = {"kind": self.kind, "config": config.to_dict(), "script_sha256": hashlib.sha256(self.script).hexdigest()}

    def render_seed(self, text):
        return text + "\n"

    def tokenize(self, text, initial=False):
        # Demo tokens are Unicode codepoints, not the native model's tokenization.
        return [ord(c) for c in text]

    def eval(self, tokens):
        if len(self.tokens) + len(tokens) > self.n_ctx:
            raise ValueError("context overflow")
        self.tokens.extend(tokens)
        self.decoded_tokens += len(tokens)

    def sample(self):
        value = self.script[self.index % len(self.script)]
        self.index += 1
        return value

    def piece(self, token):
        return bytes([token])

    def is_eog(self, token):
        return False

    def shift(self, keep, discard):
        del self.tokens[keep:keep + discard]

    def checkpoint_size_bytes(self):
        return len(self.tokens) * 16 + 4096

    def save(self, directory):
        write_durable(directory / "engine.json", {"tokens": self.tokens, "decoded_tokens": self.decoded_tokens, "index": self.index})

    def load(self, directory):
        obj = json.loads((directory / "engine.json").read_text())
        self.tokens, self.decoded_tokens, self.index = obj["tokens"], obj["decoded_tokens"], obj["index"]
        return {"native_state_loaded": False, "demo_fixture_restored": True, "prompt_tokens_reevaluated": 0}

    def close(self):
        pass

    def rebuild(self, directory):
        obj = json.loads((directory / "engine.json").read_text())
        self.tokens = []
        self.index, self.decoded_tokens = obj["index"], obj["decoded_tokens"]
        self.eval(obj["tokens"])
        return {"native_state_loaded": False, "demo_fixture_restored": True,
                "prompt_tokens_reevaluated": len(self.tokens)}


def make_backend(config):
    return DemoBackend(config) if config.backend == "demo" else LlamaBackend(config)
