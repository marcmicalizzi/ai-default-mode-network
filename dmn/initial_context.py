"""Validated text-context reconstruction; never a claim to restore lost KV.

The source server performs its real Jinja rendering and tokenization without
inference. No local imitation of Open WebUI's normalization is substituted.
"""
from __future__ import annotations

import dataclasses
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import secrets
import shutil
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, ProxyHandler

from .backend import sha256_file
from .config import Config
from .protocol import PROTOCOL, event_text
from .storage import write_durable
from .responses import responses_to_chat

SOURCE_BUILD = "b10502-0adcc3bb5"
CHAIN = ["penalties", "dry", "top_n_sigma", "top_k", "typ_p", "top_p", "min_p", "xtc", "temperature"]
SAMPLER_KEYS = ("temperature", "top_k", "top_p", "min_p", "repeat_penalty", "repeat_last_n")
DISABLED = {"dynatemp_range": 0, "top_n_sigma": -1, "typical_p": 1, "xtc_probability": 0,
            "presence_penalty": 0, "frequency_penalty": 0, "dry_multiplier": 0,
            "mirostat": 0, "adaptive_target": -1, "ignore_eos": False}
ALLOWED_REQUEST = {"model", "messages", "stream", "stream_options", "seed", *SAMPLER_KEYS,
    *DISABLED, "samplers", "max_tokens", "max_completion_tokens", "n_predict", "stop", "n",
    "logprobs", "top_logprobs", "user", "metadata", "chat_template_kwargs", "reasoning_effort",
    "reasoning_format", "enable_thinking", "reasoning_budget", "response_format", "tools", "tool_choice",
    "store", "parallel_tool_calls", "previous_response_id", "text"}


def validate_request(request):
    if not isinstance(request, dict) or not isinstance(request.get("model"), str):
        raise ValueError("provider request must include a model ID")
    unknown = request.keys() - ALLOWED_REQUEST
    if unknown:
        raise ValueError("unsupported provider fields: " + ", ".join(sorted(unknown)))
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("provider request must have messages")
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant", "tool", "developer"}:
            raise ValueError("unsupported initial-context message role")
        if message.keys() - {"role", "content", "reasoning_content", "tool_calls", "tool_call_id", "name", "id"}:
            raise ValueError("unsupported message fields; do not silently discard tools or metadata")
        content = message.get("content")
        text_parts = isinstance(content, list) and all(isinstance(part, dict) and (
            part.get("type") == "text" and isinstance(part.get("text"), str) or
            part.get("type") == "refusal" and isinstance(part.get("refusal"), str)) for part in content)
        if not isinstance(content, str) and not text_parts and not (message.get("role") == "assistant" and message.get("tool_calls") and content is None):
            raise ValueError("multimodal/structured message content needs a separate importer")
        if "reasoning_content" in message and not isinstance(message["reasoning_content"], str):
            raise ValueError("reasoning_content must be text")
    if request.get("n", 1) != 1:
        raise ValueError("one persistent sequence requires n=1")
    if request.get("tools") is not None and not isinstance(request["tools"], list):
        raise ValueError("tools must be an array of source tool definitions")
    if request.get("response_format", {"type": "text"}) not in (None, {"type": "text"}):
        raise ValueError("constrained response grammar is not supported")
    if request.get("reasoning_budget", -1) != -1:
        raise ValueError("a finite reasoning budget needs a native budget adapter")
    if request.get("previous_response_id"):
        raise ValueError("previous_response_id cannot replace an explicit captured context")
    if request.get("text", {"format": {"type": "text"}}) not in (None, {"format": {"type": "text"}}):
        raise ValueError("structured Responses output requires a separate adapter")


def resolve_sampler(request, props, config):
    defaults = props.get("default_generation_settings", {}).get("params", {})
    required = {*SAMPLER_KEYS, *DISABLED, "seed", "samplers"}
    if required - defaults.keys():
        raise ValueError("source /props lacks effective sampler defaults")
    effective = {**defaults, **{k: v for k, v in request.items() if k in required}}
    for key, disabled in DISABLED.items():
        if effective[key] != disabled:
            raise ValueError(f"unsupported active sampler option {key}={effective[key]!r}")
    if effective["samplers"] != CHAIN:
        raise ValueError("only the validated llama-server default sampler order is supported")
    seed = effective["seed"]
    if type(seed) is not int or seed < -1 or seed > 0xffffffff:
        raise ValueError("invalid source seed")
    if seed in {-1, 0xffffffff}:
        seed = secrets.randbits(32)
    settings = {key: effective[key] for key in SAMPLER_KEYS}
    # Preserve a known seed when supplied, but never imply the previous RNG
    # state exists. DMN owns a separately persisted Python RNG from this point.
    result = dataclasses.replace(config, **settings, seed=seed, sampler_order="llama_default_v1", system_prompt="")
    return result, {"source_effective": effective, "runtime_settings": {**settings, "seed": seed,
        "sampler_order": result.sampler_order}, "original_rng_restored": False,
        "implementation": "DMN NumPy/Python RNG, default llama-server filter ordering",
        "limitation": "Settings/order preserved; source sampler RNG and bit-identical sampling arithmetic are not restored.",
        "per_turn_controls_replaced_by_dmn": {k: request[k] for k in
            ("stop", "max_tokens", "max_completion_tokens", "n_predict", "stream", "reasoning_format", "tool_choice") if k in request},
        "source_tool_definitions_retained_as_context_only": len(request.get("tools") or [])}


def local_request(origin, route, body=None):
    parts = urlsplit(origin)
    if parts.scheme != "http" or parts.hostname != "127.0.0.1" or parts.path not in {"", "/"} or parts.query or parts.fragment or parts.username:
        raise ValueError("renderer must be a direct loopback llama-server origin")
    req = Request(origin.rstrip("/") + route, headers={"Content-Type": "application/json"},
                  data=None if body is None else json.dumps(body).encode())
    with build_opener(ProxyHandler({})).open(req, timeout=60) as response:
        return json.load(response)


def prepare_initial_context(request_path: Path, output: Path, config: Config, server_url: str,
                            keep_prefix_tokens=0, request_fn=local_request):
    if output.exists():
        raise ValueError("initial-context bundle requires a new output directory")
    if config.backend != "llama" or not config.model_path:
        raise ValueError("initial-context import requires a real model config")
    original_request = json.loads(request_path.read_text(encoding="utf-8"))
    is_responses = isinstance(original_request, dict) and "input" in original_request
    request = responses_to_chat(original_request) if is_responses else original_request
    validate_request(request)
    props = request_fn(server_url, "/props")
    if props.get("build_info") != SOURCE_BUILD:
        raise ValueError(f"renderer build must be the validated {SOURCE_BUILD}; got {props.get('build_info')}")
    model = Path(config.model_path).resolve()
    source_model = Path(props.get("model_path", "")).resolve()
    if not source_model.is_file() or source_model != model:
        raise ValueError("renderer model path must match the explicitly configured local GGUF")
    if not isinstance(props.get("chat_template"), str) or not props["chat_template"]:
        raise ValueError("renderer did not report its chat template")
    effective_config, sampler = resolve_sampler(request, props, config)
    rendered = request_fn(server_url, "/apply-template", request)
    prompt = rendered.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("renderer did not return a text prompt")
    tokens = request_fn(server_url, "/tokenize", {"content": prompt, "add_special": True, "parse_special": True})["tokens"]
    if not tokens or any(type(t) is not int or t < 0 for t in tokens):
        raise ValueError("renderer returned invalid token IDs")
    conversion = {"api": "responses" if is_responses else "chat_completions"}
    if is_responses:
        count = request_fn(server_url, "/v1/responses/input_tokens", original_request)["input_tokens"]
        if count != len(tokens):
            raise ValueError("Responses native token count disagrees with converted/rendered context")
        conversion.update(source="server-chat.cpp at 0adcc3bb5, text conversion port",
            native_responses_input_tokens=count, native_token_count_matched=True,
            limitation="Count agreement is checked independently; exact conversion additionally relies on the pinned source port and its supported-input tests.")
    prefix_method = "explicit_token_count"
    if keep_prefix_tokens == 0:
        # Find the template's stable prefix before the first non-system turn.
        # Keep every source system/developer message and tool definition in the
        # exact server request; replace only that turn's body in two probes.
        # Their common token prefix with the real prompt avoids guessing role
        # delimiters or cutting through a tokenizer piece.
        first = next((i for i, m in enumerate(request["messages"])
                      if m["role"] not in {"system", "developer"}), None)
        if first is None:
            raise ValueError("a system-only request requires an explicit keep-prefix token count")
        candidates = [tokens]
        for marker in ("A_DMN_PREFIX_BOUNDARY_" + secrets.token_hex(12), "Z_DMN_PREFIX_BOUNDARY_" + secrets.token_hex(12)):
            probe = deepcopy(request)
            probe["messages"][first] = {"role": request["messages"][first]["role"], "content": marker}
            variant = request_fn(server_url, "/apply-template", probe)["prompt"]
            candidates.append(request_fn(server_url, "/tokenize", {"content": variant, "add_special": True, "parse_special": True})["tokens"])
        keep_prefix_tokens = 0
        for group in zip(*candidates):
            if len(set(group)) != 1:
                break
            keep_prefix_tokens += 1
        prefix_method = "common_server_token_prefix_before_first_non_system_message"
    if type(keep_prefix_tokens) is not int or not 1 <= keep_prefix_tokens <= len(tokens):
        raise ValueError("keep-prefix must select a nonempty part of the captured prompt")
    if keep_prefix_tokens + config.turnover_reserve + 1536 >= config.n_ctx:
        raise ValueError("protected source prefix leaves insufficient context for the DMN contract")
    model_hash = sha256_file(model)
    output.mkdir(parents=True)
    shutil.copyfile(request_path, output / "provider-request.json")
    write_durable(output / "normalized-chat-request.json", request)
    write_durable(output / "conversion.json", conversion)
    write_durable(output / "server-props.json", props)
    write_durable(output / "sampler.json", sampler)
    write_durable(output / "config.json", effective_config.to_dict())
    write_durable(output / "tokens.json", tokens)
    (output / "rendered-prompt.txt").write_bytes(prompt.encode("utf-8"))
    (output / "chat-template.jinja").write_bytes(props["chat_template"].encode("utf-8"))
    manifest = {"schema": 1, "kind": "validated_initial_context", "continuity": "initial_context_reconstruction",
        "model_sha256": model_hash, "source_build": props["build_info"],
        "source_model_id": request["model"], "source_token_count": len(tokens),
        "source_api": conversion["api"],
        "keep_prefix_tokens": keep_prefix_tokens, "source_inference_performed": False,
        "source_prefix_method": prefix_method,
        "tokenization": {"add_special": True, "parse_special": True},
        "files": {p.name: sha256_file(p) for p in output.iterdir()},
        "limits": "Captured effective text context, not former KV. A separately recorded DMN contract is appended after these exact tokens. Original RNG is unavailable."}
    write_durable(output / "manifest.json", manifest)
    return manifest


def validate_bundle(bundle, backend):
    manifest = json.loads((bundle / "manifest.json").read_text())
    required = {"provider-request.json", "server-props.json", "sampler.json", "config.json", "tokens.json", "rendered-prompt.txt", "chat-template.jinja"}
    if manifest.get("schema") != 1 or manifest.get("kind") != "validated_initial_context" or not required <= manifest.get("files", {}).keys():
        raise ValueError("not a complete initial-context bundle")
    for name, digest in manifest["files"].items():
        if Path(name).name != name or not (bundle / name).is_file() or sha256_file(bundle / name) != digest:
            raise ValueError("initial-context integrity check failed")
    if backend.kind != "native_llama_kv" or backend.fingerprint["model_sha256"] != manifest["model_sha256"]:
        raise ValueError("initial-context model identity differs")
    template = (bundle / "chat-template.jinja").read_bytes().decode("utf-8")
    if backend.template != template:
        raise ValueError("initial-context template differs from runtime model")
    captured = Config(**json.loads((bundle / "config.json").read_text()))
    for key in (*SAMPLER_KEYS, "seed", "sampler_order"):
        if getattr(captured, key) != getattr(backend.config, key):
            raise ValueError(f"initial-context sampler differs: {key}")
    if backend.config.system_prompt:
        raise ValueError("initial-context reconstruction must use the captured system text")
    prompt = (bundle / "rendered-prompt.txt").read_bytes().decode("utf-8")
    tokens = json.loads((bundle / "tokens.json").read_text())
    if tokens != backend.tokenize(prompt, initial=True):
        raise ValueError("source and runtime tokenizers disagree; no reconstruction performed")
    if len(tokens) != manifest["source_token_count"] or not 1 <= manifest["keep_prefix_tokens"] <= len(tokens):
        raise ValueError("invalid source token boundary")
    return manifest, prompt, tokens


def initialize_runtime(runtime, bundle: Path):
    """Called only in the fresh-runtime path, before any seed or checkpoint."""
    if runtime.backend.tokens or runtime.store.latest() or runtime.store.next_event(0):
        raise ValueError("initial-context import requires an empty runtime")
    manifest, prompt, source_tokens = validate_bundle(bundle, runtime.backend)
    transition = event_text("dmn_transition", {
        "fact": "The preceding effective conversation context was reconstructed from text. Its former KV and RNG were unavailable. No inference occurred during the unloaded interval. Prior frontend tool definitions and calls remain historical context, not callable tools. Only the following DMN action contract is active from now on.",
        "source_token_count": len(source_tokens), **runtime.clock()}, runtime.now())
    transition += "\n" + PROTOCOL + "\n<internal_cognition>\n"
    transition_tokens = runtime.backend.tokenize(transition)
    keep = manifest["keep_prefix_tokens"]
    if (len(source_tokens) + len(transition_tokens) + runtime.config.turnover_reserve + 256 >= runtime.backend.n_ctx or
            keep + len(transition_tokens) + runtime.config.turnover_reserve + 256 >= runtime.backend.n_ctx):
        raise ValueError("captured context and DMN transition do not fit; no truncation or compaction was performed")
    destination = runtime.root / "import"
    shutil.copytree(bundle, destination)
    # Historical action-looking text goes directly to eval, never ActionParser.
    runtime._eval(source_tokens)
    runtime.state.update(continuity="initial_context_reconstruction", import_manifest=manifest,
        rendered_seed=prompt + transition, keep_prefix=keep,
        protected_protocol={"start": len(source_tokens), "end": len(source_tokens) + len(transition_tokens)},
        initial_context={"source_tokens": len(source_tokens), "transition_tokens": len(transition_tokens),
            "source_tokens_sha256": hashlib.sha256(json.dumps(source_tokens).encode()).hexdigest(),
            "native_state_restored": False, "historical_actions_executed": False})
    if len(source_tokens) == keep:
        runtime.state["keep_prefix"] = keep + len(transition_tokens)
        del runtime.state["protected_protocol"]
    runtime._eval(transition_tokens)
    runtime.checkpoint()
