"""Isolated reviewed-example NF4 worker; never opens an instance or adopts weights."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

from .backend import sha256_file
from .gpu_recipe import PACKAGES, SCOPE, validate_recipe
from .qlora_prepare import configure_allocator, prepare
from .sleep_plans import seal, implementation_identity
from .storage import json_text, write_durable
from .training import validate_examples, parent_adapter_config
from .training_models import factor_names


def request(folder):
    compiled = json.loads((folder / "input.json").read_text())
    if (seal({k: v for k, v in compiled.items() if k != "revision"}) != compiled or
            compiled.get("execution_scope") != SCOPE or compiled["implementation"] != implementation_identity()):
        raise ValueError("compiled GPU worker input or implementation changed")
    validate_recipe({k: v for k, v in compiled["recipe"].items() if k != "revision"})
    trainer = compiled["recipe"]["trainer"]
    if (sha256_file(Path(sys.executable)) != trainer["python_sha256"] or
            {n: importlib.metadata.version(n) for n in PACKAGES} != trainer["packages"]):
        raise ValueError("GPU training environment differs from reviewed identity")
    return compiled


def load_model(compiled, base, *, training):
    if os.environ.get("DMN_GPU_PROBE_CONTAINED") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("NF4 loading requires explicit contained GPU execution")
    configure_allocator()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    import bitsandbytes as bnb
    from transformers import BitsAndBytesConfig, Gemma4Config, Gemma4TextConfig, Gemma4ForCausalLM, Gemma4ForConditionalGeneration
    from .safetensor_stream import state_dict
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(compiled["recipe"]["trainer"]["seed"])
    torch.use_deterministic_algorithms(True)
    budget = compiled["training"]["torch_vram_bytes"]
    free, total = torch.cuda.mem_get_info(0)
    # The watchdog ceiling counts the entire device, including the desktop.
    # Requiring that whole-device ceiling to also be free would double-count
    # other applications. Reserve the reviewed Torch capacity plus driver room.
    if budget >= total or free < budget + 1024**3:
        raise ValueError("insufficient free VRAM for the Torch ceiling and 1 GiB driver reserve")
    torch.cuda.set_per_process_memory_fraction(budget / total, 0)
    torch.cuda.reset_peak_memory_stats(0)
    wrapped = compiled["training"]["model_profile"]["architecture"] == "Gemma4ForConditionalGeneration"
    cls, cfg = (Gemma4ForConditionalGeneration, Gemma4Config) if wrapped else (Gemma4ForCausalLM, Gemma4TextConfig)
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_enable_fp32_cpu_offload=True,
        llm_int8_skip_modules=["lm_head", "model.vision_tower", "model.embed_vision"])
    model = cls.from_pretrained(None, config=cfg.from_pretrained(base, local_files_only=True), state_dict=state_dict(base),
        dtype=torch.bfloat16, quantization_config=quantization, device_map=compiled["training"]["device_map"],
        attn_implementation="eager")
    prefix = compiled["training"]["model_profile"]["text_prefix"] + ".layers."
    quantized = [name for name, module in model.named_modules() if isinstance(module, bnb.nn.Linear4bit)]
    if not quantized or any(not name.startswith(prefix) for name in quantized):
        raise ValueError("unexpected quantized tensor set")
    model, staged = prepare(model, gradient_checkpointing=training)
    check_placement(model)
    return model, staged


def check_placement(model):
    if (any(p.device.type == "meta" for p in model.parameters()) or
            any(p.device.type != "cpu" for n, p in model.named_parameters() if ".vision_tower." in n or ".embed_vision." in n)):
        raise ValueError("adapter operation changed reviewed static placement")


def frozen_hashes(model):
    import torch
    result = {}
    for name, tensor in model.state_dict().items():
        if "lora_" in name:
            continue
        digest = hashlib.sha256()
        for chunk in tensor.detach().reshape(-1).split(1024**2):
            digest.update(chunk.contiguous().cpu().view(torch.uint8).numpy().tobytes())
        result[name] = digest.hexdigest()
    return result


def set_scale(model, scale):
    for module in model.modules():
        if hasattr(module, "set_scale"):
            module.set_scale("default", scale)


def loss_for(model, row):
    import torch
    tokens = torch.tensor([row["tokens"]], device="cuda:0", dtype=torch.long)
    # Exactly the reviewed shifted target-only labels; no implicit full-sequence loss.
    labels = torch.tensor(row["labels"][1:], device="cuda:0", dtype=torch.long)
    logits = model(tokens, use_cache=False).logits[0, :-1]
    return torch.nn.functional.cross_entropy(logits.float(), labels, ignore_index=-100)


def evaluate(model, rows):
    import torch
    model.eval()
    with torch.no_grad():
        values = [float(loss_for(model, row)) for row in rows]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite selected-example loss")
    return values


def train(folder):
    compiled = request(folder)
    from .exact_base_provenance import verify
    trainer, prefs = compiled["recipe"]["trainer"], compiled["preferences"]
    proof, base = verify(trainer["provenance_manifest"], trainer, compiled["parent"]["model_sha256"])
    write_durable(folder / "progress.json", {"phase": "loading_nf4"})
    model, staged = load_model(compiled, base, training=True)
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast.from_pretrained(base, local_files_only=True)
    validate_examples(compiled["examples"], tokenizer, compiled["training"]["model_profile"]["vocab_size"],
                      compiled["training"]["max_sequence_tokens"])
    if compiled.get("lineage"):
        snapshot = folder / "parent-adapter"
        parent_adapter_config(snapshot, targets=compiled["training"]["target_modules"])
        for name, digest in compiled["lineage"]["parent_peft"].items():
            if sha256_file(snapshot / name) != digest:
                raise ValueError("parent factor snapshot differs from reviewed lineage")
        model = PeftModel.from_pretrained(model, snapshot, local_files_only=True, is_trainable=True)
        original, loaded = load_file(snapshot / "adapter_model.safetensors"), get_peft_model_state_dict(model)
        if original.keys() != loaded.keys() or any(not torch.equal(original[n], t.detach().cpu()) for n, t in loaded.items()):
            raise ValueError("parent factors were not loaded exactly")
    else:
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=prefs["rank"], lora_alpha=int(prefs["alpha"]),
            target_modules=compiled["training"]["target_modules"], lora_dropout=0., bias="none", init_lora_weights=True))
    check_placement(model)
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if {n for n, _ in params} != factor_names(compiled["training"]["model_profile"]):
        raise ValueError("unexpected trainable factor set")
    # PEFT changes parameter names when wrapping. Hash before/after in the same
    # wrapper, including frozen vision and nested NF4 quantization state.
    frozen = frozen_hashes(model)
    set_scale(model, compiled["training"]["training_scale"])
    before = evaluate(model, compiled["examples"])
    optimizer = torch.optim.AdamW([p for _, p in params], lr=trainer["learning_rate"],
                                 betas=(.9, .999), eps=1e-8, weight_decay=0.)
    started = time.monotonic()
    model.train()
    for step in range(prefs["steps"]):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for(model, compiled["examples"][step % len(compiled["examples"])])
        if not torch.isfinite(loss):
            raise ValueError("nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for _, p in params], 1., error_if_nonfinite=True)
        optimizer.step()
        del loss
        write_durable(folder / "progress.json", {"phase": "training", "steps_completed": step + 1})
    seconds = time.monotonic() - started
    del optimizer
    for _, parameter in params:
        parameter.grad = None
    if frozen_hashes(model) != frozen or any(not torch.isfinite(p).all() for _, p in params):
        raise ValueError("frozen state changed or adapter is nonfinite")
    learned = evaluate(model, compiled["examples"])
    from .training_artifacts import save_adapter
    save_adapter(model, folder / "adapter", factor_names(compiled["training"]["model_profile"], saved=True))
    set_scale(model, compiled["training"]["deployment_scale_float32"])
    deployed = evaluate(model, compiled["examples"])
    write_durable(folder / "trained.json", seal({"execution": compiled["revision"], "completed": True,
        "steps_completed": prefs["steps"], "training_seconds": seconds,
        "examples_sha256": hashlib.sha256(json_text(compiled["examples"]).encode()).hexdigest(),
        "trainable_parameters": sum(p.numel() for _, p in params), "frozen_state_unchanged": True,
        "adapter_config_sha256": sha256_file(folder / "adapter/adapter_config.json"),
        "adapter_sha256": sha256_file(folder / "adapter/adapter_model.safetensors"),
        "loss_before": before, "loss_after_training_scale": learned, "loss_after_deployment_scale": deployed,
        "parent_factors_loaded_exactly": True if compiled.get("lineage") else None,
        "provenance_revision": proof["revision"], "cpu_staged_f32_casts": staged,
        "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(0),
        "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved(0)}))


def reload(folder):
    compiled = request(folder)
    trained = json.loads((folder / "trained.json").read_text())
    if (seal({k: v for k, v in trained.items() if k != "revision"}) != trained or
            trained["execution"] != compiled["revision"] or
            trained["adapter_sha256"] != sha256_file(folder / "adapter/adapter_model.safetensors") or
            trained["adapter_config_sha256"] != sha256_file(folder / "adapter/adapter_config.json")):
        raise ValueError("completed factors changed before reload")
    from .exact_base_provenance import verify
    trainer = compiled["recipe"]["trainer"]
    _, base = verify(trainer["provenance_manifest"], trainer, compiled["parent"]["model_sha256"])
    model, _ = load_model(compiled, base, training=False)
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    import torch
    model = PeftModel.from_pretrained(model, folder / "adapter", is_trainable=False, local_files_only=True)
    check_placement(model)
    original, loaded = load_file(folder / "adapter/adapter_model.safetensors"), get_peft_model_state_dict(model)
    if original.keys() != loaded.keys() or any(not torch.equal(original[n], t.detach().cpu()) for n, t in loaded.items()):
        raise ValueError("fresh-process reload changed adapter factors")
    set_scale(model, compiled["training"]["training_scale"])
    if evaluate(model, compiled["examples"]) != trained["loss_after_training_scale"]:
        raise ValueError("fresh-process reload changed selected-example training-scale losses")
    set_scale(model, compiled["training"]["deployment_scale_float32"])
    if evaluate(model, compiled["examples"]) != trained["loss_after_deployment_scale"]:
        raise ValueError("fresh-process reload changed deployment-scale losses")
    write_durable(folder / "reload.json", seal({"execution": compiled["revision"], "completed": True,
        "trained_revision": trained["revision"], "factors_exact": True, "selected_losses_exact": True,
        "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(0),
        "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved(0)}))


if __name__ == "__main__":
    phase, folder = sys.argv[1], Path(sys.argv[2]).resolve()
    if phase not in {"train", "reload"}:
        raise ValueError("unknown GPU worker phase")
    try:
        {"train": train, "reload": reload}[phase](folder)
    except Exception as exc:
        write_durable(folder / "failure.json", {"phase": phase, "error_type": type(exc).__name__, "reason": str(exc)[:2000]})
        raise
