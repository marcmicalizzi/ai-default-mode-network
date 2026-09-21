"""Inspect or explicitly execute a tiny synthetic NF4 GPU training experiment.

Not a DMN recipe: no Runtime, private examples, live model, adoption or downloads.
Execution requires the separate contained launcher and a maintenance agreement.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dmn.backend import sha256_file
from dmn.storage import write_durable
from dmn.training_models import model_profile, factor_names


def inspect_fixture(folder):
    folder = Path(folder).resolve()
    marker = json.loads((folder / "fixture.json").read_text())
    base = folder / "base"
    files = list(base.iterdir())
    if (marker.get("synthetic_only") is not True or any(not p.is_file() or p.suffix not in {".json", ".safetensors"} for p in files) or
            sum(p.stat().st_size for p in files) > 4 * 1024**2):
        raise ValueError("GPU probe accepts only a generated tiny safetensors fixture")
    config = json.loads((base / "config.json").read_text())
    text = config.get("text_config", {})
    if (config.get("architectures") != ["Gemma4ForConditionalGeneration"] or
            any(text.get(k) != v for k, v in {"hidden_size": 256, "vocab_size": 263, "num_hidden_layers": 2}.items())):
        raise ValueError("GPU probe requires the known tiny full-wrapper geometry")
    return {"synthetic_only": True, "gpu_execution": False, "source": str(folder),
        "source_hashes": {p.name: sha256_file(p) for p in files}, "profile": model_profile(base, wrapped=True),
        "training": {"quantization": "NF4", "nested_quantization": True, "compute_dtype": "bfloat16",
            "rank": 2, "alpha": 4, "steps": 4, "batch_size": 1, "sequence_tokens": 6,
            "gradient_checkpointing": "non-reentrant", "optimizer": "AdamW", "learning_rate": .001},
        "gpu_policy": "explicit experiment only; PyTorch allocator cap is not a total process VRAM quota"}


def execute(folder, output, torch_vram_mib):
    plan = inspect_fixture(folder)  # Refuse large/live assets before importing CUDA.
    if os.environ.get("DMN_GPU_PROBE_CONTAINED") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("GPU execution requires the explicit contained research launcher")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    import torch
    import bitsandbytes as bnb
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig, Gemma4ForConditionalGeneration
    if not torch.cuda.is_available():
        raise ValueError("CUDA training is unavailable; no fallback or partial validation")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(814)
    budget = torch_vram_mib * 1024**2
    free, total = torch.cuda.mem_get_info(0)
    if not 256 * 1024**2 <= budget <= 2 * 1024**3 or free < budget + 512 * 1024**2:
        raise ValueError("tiny GPU probe needs its allocator budget plus 512 MiB free headroom")
    torch.cuda.set_per_process_memory_fraction(budget / total, 0)
    torch.cuda.reset_peak_memory_stats(0)
    base = Path(plan["source"]) / "base"
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["lm_head", "model.vision_tower", "model.embed_vision", "model.audio_tower", "model.embed_audio"])

    def load():
        return Gemma4ForConditionalGeneration.from_pretrained(base, local_files_only=True,
            quantization_config=quantization, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager")

    model = load()
    quantized = [name for name, module in model.named_modules() if isinstance(module, bnb.nn.Linear4bit)]
    if not quantized or any(not name.startswith("model.language_model.layers.") for name in quantized):
        raise ValueError("unexpected NF4 module set; vision/audio must remain frozen and outside quantization")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=2, lora_alpha=4,
        target_modules=plan["profile"]["target_modules"], lora_dropout=0., bias="none"))
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if {name for name, _ in trainable} != factor_names(plan["profile"]):
        raise ValueError("unexpected trainable GPU tensor set")

    def frozen_hashes(network):
        return {name: hashlib.sha256(t.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
                for name, t in network.state_dict().items() if "lora_" not in name}

    frozen = frozen_hashes(model)
    initial_factors = {name: p.detach().cpu().clone() for name, p in trainable}
    tokens = torch.tensor([[1, 70, 71, 72, 11, 41]], device="cuda:0")

    def loss(network):
        return torch.nn.functional.cross_entropy(network(tokens, use_cache=False).logits[:, -2].float(), tokens[:, -1])

    model.eval()
    with torch.no_grad():
        before = float(loss(model))
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=.001, weight_decay=0.)
    torch.cuda.synchronize()
    started = time.monotonic()
    model.train()
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        value = loss(model)
        if not torch.isfinite(value):
            raise ValueError("nonfinite GPU training loss")
        value.backward()
        torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1., error_if_nonfinite=True)
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    model.eval()
    with torch.no_grad():
        after = float(loss(model))
    if frozen_hashes(model) != frozen:
        raise ValueError("frozen NF4/base/vision state changed")
    if any(not torch.isfinite(p).all() for _, p in trainable):
        raise ValueError("nonfinite GPU adapter factors")
    changed_factors = [name for name, p in trainable if not torch.equal(p.detach().cpu(), initial_factors[name])]
    if not changed_factors:
        raise ValueError("GPU training made no adapter factor changes")
    model.save_pretrained(output / "adapter", safe_serialization=True, save_embedding_layers=False)
    parameters = sum(p.numel() for _, p in trainable)
    # Release the first graph/optimizer before checking serialization, avoiding
    # an artificial double-model peak. A new quantized base gets the same casts.
    del optimizer, trainable, model, value, initial_factors
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    restored = prepare_model_for_kbit_training(load(), use_gradient_checkpointing=False)
    restored = PeftModel.from_pretrained(restored, output / "adapter", local_files_only=True).eval()
    with torch.no_grad():
        reloaded = float(loss(restored))
    if not abs(reloaded - after) <= 1e-5:
        raise ValueError("GPU PEFT reload changed the selected-example loss")
    if inspect_fixture(folder)["source_hashes"] != plan["source_hashes"]:
        raise ValueError("source fixture changed")
    write_durable(output / "result.json", {"completed": True, "synthetic_only": True, "gpu_execution": True,
        "plan": plan, "device": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "bitsandbytes", "accelerate")},
        "quantized_modules": quantized, "trainable_parameters": parameters, "frozen_state_unchanged": True,
        "changed_factors": changed_factors,
        "loss_before": before, "loss_after": after, "loss_after_reload": reloaded, "training_seconds": elapsed,
        "torch_allocator_cap_bytes": budget, "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(0),
        "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved(0), "total_process_vram_quota_enforced": False,
        "beneficial_learning_certified": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--torch-vram-mib", type=int, default=1024)
    parser.add_argument("--execute", action="store_true", help="requires contained launcher; actually uses CUDA")
    args = parser.parse_args()
    if args.execute:
        if args.output is None:
            parser.error("execution requires a fresh --output directory")
        execute(args.fixture, args.output, args.torch_vram_mib)
    else:
        print(json.dumps(inspect_fixture(args.fixture), indent=2))
