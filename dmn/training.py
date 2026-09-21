"""Pinned CPU recipe and artifact contracts; never authorizes learning by itself."""
from __future__ import annotations

import ctypes
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re

from .backend import sha256_file
from .ending import _owned
from .learning import _fields
from .storage import json_text

KIND = "peft_gemma4_cpu_v1"
CONTINUE_KIND = "peft_gemma4_cpu_continue_v1"
KIND_V2 = "peft_gemma4_cpu_v2"
GPU_KIND = "peft_gemma4_nf4_v1"
KINDS = {KIND, CONTINUE_KIND, KIND_V2, GPU_KIND}
SCOPE = "reviewed_cpu_training_test_only"
CHECKS = ["artifact_integrity", "retained_tokens_and_rng", "tokenizer_parity",
          "base_unchanged", "adapter_roundtrip", "finite_training"]
PACKAGES = ("torch", "transformers", "peft", "safetensors", "tokenizers", "numpy")
CONVERTER_REVISION = "4df29be4f4c3673f428170fda944a5b19f743bb8"


def read_bound_json(reference):
    _fields(reference, "path sha256", "manifest reference")
    path = Path(reference["path"])
    if path.stat().st_size > 1024 * 1024 or sha256_file(path) != reference["sha256"]:
        raise ValueError("bound manifest changed or is too large")
    return json.loads(path.read_text(encoding="utf-8"))


def tree_manifest(root, *, python_only=False, skip_hf_download_cache=False):
    """Host-side helper: pin existing local assets, without copying or fetching."""
    root = Path(root).resolve()
    files = {}
    for path in sorted(root.rglob("*")):
        if skip_hf_download_cache and path.relative_to(root).parts[:2] == ('.cache', 'huggingface'):
            continue  # Download receipts/locks are not model assets and are never loaded.
        if "__pycache__" in path.parts or not path.is_file() or (python_only and path.suffix != ".py"):
            continue
        relative = path.relative_to(root).as_posix()
        files[relative] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return {"root": str(root), "files": files, "python_only": python_only}


def verify_tree(reference, *, base=False, adapter=False, adapter_limit_bytes=4 * 1024**2, allow_metadata=False):
    value = read_bound_json(reference)
    _fields(value, "root files python_only", "file manifest")
    root = Path(value["root"])
    if not root.is_absolute() or not isinstance(value["files"], dict) or not value["files"]:
        raise ValueError("manifest requires an absolute root and explicit files")
    if base and adapter:
        raise ValueError("ambiguous manifest scope")
    python_only = not (base or adapter)
    if value["python_only"] is not python_only:
        raise ValueError("unexpected manifest scope")
    if adapter and (not {"adapter_config.json", "adapter_model.safetensors"} <= value["files"].keys() or
                    value["files"].keys() - {"adapter_config.json", "adapter_model.safetensors", "README.md"} or
                    sum(item["bytes"] for item in value["files"].values()) > adapter_limit_bytes):
        raise ValueError("parent adapter requires only tiny local PEFT safetensors/config assets")
    for name, expected in value["files"].items():
        parts = PurePosixPath(name).parts
        if not parts or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", part) or part in {".", ".."} for part in parts):
            raise ValueError("invalid artifact-relative path")
        cursor = root
        for part in parts:
            child = cursor / part
            _owned(child, cursor)
            cursor = child
        if not cursor.is_file() or cursor.stat().st_size != expected["bytes"] or sha256_file(cursor) != expected["sha256"]:
            raise ValueError("bound training asset changed")
    # Every declared file was already hashed above. Detect additions without
    # rereading large safetensors shards a second time on every verification.
    actual_names = {path.relative_to(root).as_posix() for path in root.rglob("*")
                    if "__pycache__" not in path.parts and path.is_file() and
                    not (allow_metadata and path.relative_to(root).parts[:2] == ('.cache', 'huggingface')) and
                    (not python_only or path.suffix == ".py")}
    if actual_names != set(value["files"]) or str(root.resolve()) != value["root"]:
        raise ValueError("training asset set changed")
    if base:
        if not {"config.json", "tokenizer.json", "tokenizer_config.json"} <= value["files"].keys():
            raise ValueError("base requires explicit config and fast tokenizer files")
        ancillary = {"README.md", "chat_template.jinja"} if allow_metadata else set()
        if any("/" in name or (not name.endswith((".json", ".safetensors")) and name not in ancillary) for name in value["files"]):
            raise ValueError("base permits only flat JSON and safetensors assets; no executable or pickle weights")
    elif not adapter and not {"convert_hf_to_gguf.py", "convert_lora_to_gguf.py"} <= value["files"].keys():
        raise ValueError("pinned converter entrypoints are missing")
    return root


def validate_recipe(value):
    if isinstance(value, dict) and value.get("kind") == GPU_KIND:
        from .gpu_recipe import validate_recipe as validate_gpu
        return validate_gpu(value)
    _fields(value, "schema kind parent resources checks trainer", "CPU recipe")
    if value["schema"] != 1 or value["kind"] not in KINDS or value["checks"] != CHECKS:
        raise ValueError("unsupported CPU recipe or checks")
    parent = value["parent"]
    continuing = value["kind"] == CONTINUE_KIND or (value["kind"] == KIND_V2 and bool(parent.get("lora_adapters")))
    if parent.get("kind") != "native_llama_kv" or parent.get("research_lora"):
        raise ValueError("CPU recipe requires a native parent without research weight overrides")
    if not continuing and parent.get("lora_adapters"):
        raise ValueError("this first-adapter recipe requires an unadapted native parent; existing learning cannot be discarded")
    if continuing:
        from .adapters import AdapterSpec
        adapters = parent.get("lora_adapters")
        if not isinstance(adapters, list) or len(adapters) != 1:
            raise ValueError("continuation requires exactly one active adapter; no stacking or merging")
        old = adapters[0]
        _fields(old, "sha256 base_model_sha256 scale activation", "parent adapter")
        expected = AdapterSpec("identity-only", old["sha256"], old["base_model_sha256"], old["scale"]).identity()
        if old != expected or old["base_model_sha256"] != parent["model_sha256"] or old["scale"] <= 0:
            raise ValueError("continuation requires a positive whole-context parent strength and matching base")
    trainer = value["trainer"]
    fields = "python python_sha256 packages base_manifest converter_manifest converter_revision inference_name learning_rate seed"
    _fields(trainer, fields + (" parent_adapter_manifest" if continuing else "") +
            (" provenance_manifest" if value["kind"] == KIND_V2 else ""), "trainer")
    references = ["base_manifest", "converter_manifest"]
    if continuing:
        references.append("parent_adapter_manifest")
    if value["kind"] == KIND_V2:
        references.append("provenance_manifest")
    for key in references:
        reference = trainer[key]
        _fields(reference, "path sha256", "manifest reference")
        if not Path(reference["path"]).is_absolute() or not re.fullmatch(r"[0-9a-f]{64}", reference["sha256"]):
            raise ValueError("invalid manifest binding")
    if not Path(trainer["python"]).is_absolute() or not re.fullmatch(r"[0-9a-f]{64}", trainer["python_sha256"]):
        raise ValueError("trainer interpreter must have a bound absolute path")
    if set(trainer["packages"]) != set(PACKAGES) or any(not isinstance(v, str) or not v for v in trainer["packages"].values()):
        raise ValueError("explicit training package versions required")
    if trainer["converter_revision"] != CONVERTER_REVISION:
        raise ValueError("unvalidated converter revision")
    if not isinstance(trainer["inference_name"], str) or not trainer["inference_name"] or len(trainer["inference_name"]) > 256:
        raise ValueError("conversion requires the reviewed inference model name")
    rate = trainer["learning_rate"]
    if type(rate) not in (float, int) or not math.isfinite(rate) or not 0 < rate <= .1:
        raise ValueError("learning rate must be finite, positive and at most 0.1")
    if type(trainer["seed"]) is not int or not 0 <= trainer["seed"] < 2**32:
        raise ValueError("invalid training seed")
    if value["resources"]["max_vram_bytes"] != 0:
        raise ValueError("CPU recipe requires a zero GPU budget")


def parent_adapter_config(path, *, targets=None):
    """Narrow pinned PEFT contract; reject features the converter may ignore."""
    config = json.loads((path / "adapter_config.json").read_text())
    required = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "bias": "none", "lora_dropout": 0}
    neutral = dict.fromkeys(("alora_invocation_tokens", "arrow_config", "auto_mapping", "corda_config",
        "eva_config", "exclude_modules", "kasa_config", "layer_replication", "layers_pattern", "layers_to_transform",
        "lora_ga_config", "megatron_config", "modules_to_save", "monteclora_config", "revision", "target_parameters",
        "trainable_token_indices", "use_bdlora", "velora_config"))
    neutral.update(alpha_pattern={}, rank_pattern={}, loftq_config={}, ensure_weight_tying=False,
        fan_in_fan_out=False, lora_bias=False, use_dora=False, use_qalora=False, use_rslora=False,
        init_lora_weights=True, megatron_core="megatron.core", qalora_group_size=16)
    metadata = {"base_model_name_or_path", "inference_mode", "peft_version"}
    if (not isinstance(config, dict) or config.keys() - (required.keys() | neutral.keys() | metadata | {"r", "lora_alpha", "target_modules"}) or
            any(config.get(k) != v for k, v in required.items()) or
            any(config[k] != v for k, v in neutral.items() if k in config) or
            sorted(config.get("target_modules", [])) != sorted(targets or ["o_proj", "q_proj"]) or
            type(config.get("r")) is not int or not 1 <= config["r"] <= 64 or
            type(config.get("lora_alpha")) not in (int, float) or not 1 <= config["lora_alpha"] <= 1024 or
            int(config["lora_alpha"]) != config["lora_alpha"]):
        raise ValueError("unsupported parent PEFT configuration; only plain fixed-rank q_proj/o_proj LoRA is validated")
    return config


def compile_training(value):
    """Add executable semantics before the immutable plan is sealed/reviewed."""
    if value["recipe"]["kind"] == GPU_KIND:
        from .gpu_recipe import compile_training as compile_gpu
        return compile_gpu(value)
    prefs = value["preferences"]
    if not 1 <= prefs["rank"] <= 64 or not 1 <= prefs["steps"] <= 10000:
        raise ValueError("CPU recipe supports ranks 1..64 and 1..10000 steps")
    if (type(prefs["alpha"]) not in (int, float) or prefs["alpha"] != int(prefs["alpha"]) or
            not 1 <= prefs["alpha"] <= 1024):
        raise ValueError("CPU recipe requires integer alpha 1..1024")
    scale = ctypes.c_float(prefs["scale"]).value
    if not math.isfinite(scale) or (prefs["scale"] != 0 and scale == 0):
        raise ValueError("deployment strength is not representable")
    from .worker_limits import WorkerLimits
    WorkerLimits(value["resources"]["max_ram_bytes"], value["resources"]["max_training_seconds"])
    continuing = value["recipe"]["kind"] == CONTINUE_KIND or (value["recipe"]["kind"] == KIND_V2 and bool(value["parent"].get("lora_adapters")))
    profile = None
    if value["recipe"]["kind"] == KIND_V2:
        from .training_models import model_profile
        from .base_provenance import verify
        trainer = value["recipe"]["trainer"]
        proof, _ = verify(trainer["provenance_manifest"], trainer, value["parent"]["model_sha256"])
        base = verify_tree(trainer["base_manifest"], base=True)
        profile = model_profile(base, wrapped=True)
    training_scale = 1.
    lineage = None
    if continuing:
        reference = value["recipe"]["trainer"]["parent_adapter_manifest"]
        path = verify_tree(reference, adapter=True)
        config = parent_adapter_config(path, targets=profile["target_modules"] if profile else None)
        old = value["parent"]["lora_adapters"][0]
        if (prefs["rank"] != config["r"] or prefs["alpha"] != config["lora_alpha"] or scale != old["scale"]):
            raise ValueError("continuation preserves parent rank, alpha and deployment strength; implicit changes are refused")
        training_scale = old["scale"]
        lineage = {"parent_adapter": old, "parent_peft": {name: sha256_file(path / name) for name in
            ("adapter_config.json", "adapter_model.safetensors")}, "rank": config["r"], "alpha": config["lora_alpha"]}
    value.update(execution_scope=SCOPE, training_requested=True,
        adapter_operation=("continue_single_adapter; preserve factors/rank/alpha/strength; replace parent, never stack"
                           if continuing else "first_adapter_only; existing adapters rejected"),
        lineage=lineage,
        resource_enforcement="Windows aggregate committed-memory limit and process-tree watchdog. Disk preflight only. Tiny CPU integration gate remains mandatory; no production execution.",
        limitation="F32 base must reproduce the exact inference GGUF. No GPU, quantized-base provenance, hard disk quota or unattended service yet.",
        training={"dtype": "float32", "device": "cpu", "threads": 1, "batch_size": 1,
                  "example_order": "round_robin_in_reviewed_order", "target_modules": ["q_proj", "o_proj"],
                  "optimizer": "AdamW", "optimizer_reset": True, "betas": [.9, .999], "epsilon": 1e-8,
                  "weight_decay": 0., "max_gradient_norm": 1., "dropout": 0.,
                  "training_scale": training_scale, "deployment_scale_float32": scale,
                  "loss": "mean_cross_entropy_on_shifted_target_labels_only; no padding/truncation",
                  "evaluation": "selected-example loss at training and deployment scale; no heldout or benefit guarantee"})
    if profile:
        value["training"]["model_profile"] = profile
        value["training"]["target_modules"] = profile["target_modules"]
        value["training"]["base_provenance"] = {"revision": proof["revision"], "quantization": proof["request"]["quantization"],
            "tensor_types": proof["tensor_types"], "method": "local conversion/quantization reproduced before learning; cached proof revalidated"}
        value["limitation"] = "Tiny CPU integration only. Training uses original F32 weights; quantized inference is not QLoRA training. No GPU, hard disk quota or unattended service yet."
    return value


def validate_examples(examples, tokenizer, vocab_size, max_length):
    """No retokenization substitutions: fail if the approved native IDs differ."""
    for row in examples:
        prefix = tokenizer.encode(row["input"], add_special_tokens=False)
        tokens = tokenizer.encode(row["input"] + row["target"], add_special_tokens=False)
        if not prefix or tokens[:len(prefix)] != prefix or len(tokens) <= len(prefix) or tokens != row["tokens"]:
            raise ValueError("training tokenizer differs from the reviewed inference tokens/boundary")
        mask = [0] * len(prefix) + [1] * (len(tokens) - len(prefix))
        if row["loss_mask"] != mask or row["labels"] != [t if m else -100 for t, m in zip(tokens, mask)]:
            raise ValueError("reviewed target-only mask or labels are invalid")
        if len(tokens) > max_length or any(type(t) is not int or not 0 <= t < vocab_size for t in tokens):
            raise ValueError("reviewed example exceeds training model geometry; no truncation")


def read_completion(folder, compiled):
    if compiled.get("recipe", {}).get("kind") == GPU_KIND:
        from .gpu_recipe import read_completion as read_gpu_completion
        return read_gpu_completion(folder, compiled)
    from .sleep_plans import seal
    path = folder / "result.json"
    _owned(path, folder)
    result = json.loads(path.read_text())
    if (seal({k: v for k, v in result.items() if k != "revision"}) != result or result.get("schema") != 1 or
            result.get("training_performed") is not True or
            result.get("execution") != compiled["revision"] or result.get("completed") is not True):
        raise ValueError("training completion identity failed")
    required = {"adapter.gguf", "adapter/adapter_config.json", "adapter/adapter_model.safetensors", "base-check.gguf"}
    proven = compiled.get("recipe", {}).get("kind") == KIND_V2
    if proven:
        required.add("provenance.json")
    lineage = compiled.get("lineage")
    if lineage:
        required |= {"parent-check.gguf", "parent-adapter/adapter_config.json", "parent-adapter/adapter_model.safetensors"}
    if set(result["artifacts"]) != required:
        raise ValueError("training completion has unexpected artifacts")
    for name, digest in result["artifacts"].items():
        path = folder
        for part in PurePosixPath(name).parts:
            child = path / part
            _owned(child, path)
            path = child
        if sha256_file(path) != digest:
            raise ValueError("completed training artifact changed")
    if result["artifacts"]["base-check.gguf"] != compiled["parent"]["model_sha256"]:
        raise ValueError("training base does not reproduce inference identity")
    if proven and result["artifacts"]["provenance.json"] != compiled["recipe"]["trainer"]["provenance_manifest"]["sha256"]:
        raise ValueError("training used a different base provenance record")
    if result.get("lineage") != lineage:
        raise ValueError("training completion lineage differs from the reviewed parent")
    if lineage and (result["artifacts"]["parent-check.gguf"] != lineage["parent_adapter"]["sha256"] or
                    any(result["artifacts"]["parent-adapter/" + name] != digest for name, digest in lineage["parent_peft"].items()) or
                    result.get("parent_factors_loaded_exactly") is not True):
        raise ValueError("continuation does not reproduce the deployed parent adapter and PEFT factors")
    if (result.get("checks") != {key: True for key in CHECKS if key != "retained_tokens_and_rng"} or
            result.get("steps_completed") != compiled["preferences"]["steps"] or
            result.get("examples_sha256") != hashlib.sha256(json_text(compiled["examples"]).encode()).hexdigest()):
        raise ValueError("training checks, examples or step count differ from the reviewed plan")
    return result
