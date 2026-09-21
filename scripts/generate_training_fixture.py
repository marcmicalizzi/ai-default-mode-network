"""Generate a tiny full Gemma wrapper with text and vision tensors, CPU only.

No instance is opened. The fixture exercises K-quant blocks, with one narrower
sliding-attention tensor taking the quantizer's documented fallback.
Run with the separate CPU training interpreter and a disposable output directory.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.probe_lora_training import cpu_environment


def generate(output, tokenizer):
    cpu_environment()
    import torch
    from transformers import Gemma4Config, Gemma4TextConfig, Gemma4VisionConfig, Gemma4ForConditionalGeneration
    if torch.version.cuda is not None:
        raise ValueError("fixture generation requires CPU-only PyTorch")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(54117)
    output.mkdir(parents=True, exist_ok=False)
    text = Gemma4TextConfig(vocab_size=263, hidden_size=256, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=1, num_key_value_heads=1, head_dim=128, global_head_dim=256,
        layer_types=["sliding_attention", "full_attention"], hidden_size_per_layer_input=0,
        sliding_window=64, max_position_embeddings=2048, attention_k_eq_v=True,
        bos_token_id=1, eos_token_id=2, pad_token_id=0, initializer_range=.05, final_logit_softcapping=30.)
    vision = Gemma4VisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, head_dim=16, patch_size=4,
        pooling_kernel_size=1, position_embedding_size=32)
    config = Gemma4Config(text_config=text, vision_config=vision, audio_config=None,
                          bos_token_id=1, eos_token_id=2, pad_token_id=0)
    config._attn_implementation = "eager"
    model = Gemma4ForConditionalGeneration(config).float().cpu().eval()
    model.save_pretrained(output / "base")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(tokenizer / name, output / "base" / name)
    size = sum(p.stat().st_size for p in (output / "base").iterdir())
    if size > 4 * 1024**2:
        raise ValueError("fixture exceeds the existing tiny training gate")
    (output / "fixture.json").write_text(json.dumps({"synthetic_only": True, "gpu_used": False,
        "parameters": sum(p.numel() for p in model.parameters()), "source_bytes": size,
        "architecture": "Gemma4ForConditionalGeneration", "vision_tensors_present": True}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    generate(args.output.resolve(), args.tokenizer.resolve())
