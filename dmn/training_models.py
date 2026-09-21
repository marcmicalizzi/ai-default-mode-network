"""Explicit local Gemma text-training geometry; no model loading at import time."""
from __future__ import annotations

import json


def model_profile(base_path, *, wrapped=False):
    config = json.loads((base_path / "config.json").read_text())
    architecture = config.get("architectures")
    if config.get("auto_map"):
        raise ValueError("custom model code is not supported")
    if architecture == ["Gemma4ForCausalLM"]:
        text, prefix = config, "model"
    elif wrapped and architecture == ["Gemma4ForConditionalGeneration"]:
        text, prefix = config.get("text_config", {}), "model.language_model"
        if any((config.get(name) or {}).get("auto_map") for name in ("text_config", "vision_config", "audio_config")):
            raise ValueError("custom component code is not supported")
    else:
        raise ValueError("unsupported Gemma training architecture for this recipe")
    if (text.get("attention_dropout", 0) != 0 or text.get("enable_moe_block", False) or
            text.get("hidden_size_per_layer_input", 0) != 0):
        raise ValueError("this training recipe requires dense Gemma text without dropout or per-layer embeddings")
    layers = text.get("num_hidden_layers")
    if type(layers) is not int or not 1 <= layers <= 64:
        raise ValueError("unsupported decoder layer count")
    targets = [f"{prefix}.layers.{i}.self_attn.{target}" for i in range(layers) for target in ("q_proj", "o_proj")]
    return {"architecture": architecture[0], "text_prefix": prefix, "num_hidden_layers": layers,
            "vocab_size": text["vocab_size"], "max_position_embeddings": text["max_position_embeddings"],
            "target_modules": targets, "modality": "text_only; vision/audio/base tensors frozen"}


def load_base(path, profile):
    # Explicit classes only: no auto-model dispatch, remote code or downloads.
    from transformers import Gemma4ForCausalLM, Gemma4ForConditionalGeneration
    cls = {"Gemma4ForCausalLM": Gemma4ForCausalLM,
           "Gemma4ForConditionalGeneration": Gemma4ForConditionalGeneration}[profile["architecture"]]
    return cls.from_pretrained(path, local_files_only=True, attn_implementation="eager").float().cpu().eval()


def factor_names(profile, *, saved=False):
    suffix = "" if saved else ".default"
    return {f"base_model.model.{target}.lora_{side}{suffix}.weight"
            for target in profile["target_modules"] for side in ("A", "B")}
