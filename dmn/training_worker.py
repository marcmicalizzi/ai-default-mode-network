"""Private CPU worker entrypoint. Consumes an already compiled, reviewed plan.

The supervisor owns authorization and process containment. This worker never
opens an instance database or checkpoint, and never installs or activates weights.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .backend import sha256_file
from .sleep_plans import seal, implementation_identity
from .storage import json_text, write_durable
from .training import CHECKS, PACKAGES, SCOPE, validate_recipe, validate_examples, verify_tree


def train(folder):
    compiled = json.loads((folder / "input.json").read_text())
    if (seal({k: v for k, v in compiled.items() if k != "revision"}) != compiled or
            compiled["execution_scope"] != SCOPE or compiled["implementation"] != implementation_identity()):
        raise ValueError("compiled worker input or implementation changed")
    recipe, prefs = compiled["recipe"], compiled["preferences"]
    validate_recipe({k: v for k, v in recipe.items() if k != "revision"})
    trainer = recipe["trainer"]
    if {name: importlib.metadata.version(name) for name in PACKAGES} != trainer["packages"]:
        raise ValueError("training environment differs from the reviewed versions")
    if sha256_file(Path(sys.executable)) != trainer["python_sha256"]:
        raise ValueError("training interpreter identity changed")
    base_path = verify_tree(trainer["base_manifest"], base=True)
    converter = verify_tree(trainer["converter_manifest"])
    if sum(p.stat().st_size for p in base_path.iterdir()) > 4 * 1024**2:
        raise ValueError("CPU integration gate currently permits only tiny local training bases")
    config_data = json.loads((base_path / "config.json").read_text())
    if (config_data.get("architectures") != ["Gemma4ForCausalLM"] or config_data.get("auto_map") or
            config_data.get("attention_dropout", 0) != 0):
        raise ValueError("CPU recipe requires the validated text-only Gemma4 architecture without dropout/custom code")

    # Conversion is deliberately before learning: an unrelated training base
    # must fail without spending any gradient steps on it. A later quantized
    # recipe needs a different, measured provenance proof.
    def convert(script, arguments):
        subprocess.run([sys.executable, str(converter / script), "--outtype", "f32", *arguments],
                       check=True, stdin=subprocess.DEVNULL, cwd=converter)

    convert("convert_hf_to_gguf.py", ["--model-name", trainer["inference_name"], "--outfile",
                                   str(folder / "base-check.gguf"), str(base_path)])
    if sha256_file(folder / "base-check.gguf") != compiled["parent"]["model_sha256"]:
        raise ValueError("HF training base does not reproduce the exact inference GGUF")

    import numpy as np
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import Gemma4ForCausalLM, PreTrainedTokenizerFast
    from safetensors.numpy import load_file

    if torch.version.cuda is not None:
        raise ValueError("CPU recipe requires a CPU-only PyTorch build")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(trainer["seed"])
    torch.use_deterministic_algorithms(True)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(base_path, local_files_only=True)
    base = Gemma4ForCausalLM.from_pretrained(base_path, local_files_only=True,
                                            attn_implementation="eager").float().cpu().eval()
    validate_examples(compiled["examples"], tokenizer, base.config.vocab_size, base.config.max_position_embeddings)

    def tensor_hash(tensor):
        return hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()

    frozen = [(p, tensor_hash(p)) for p in base.parameters()]
    model = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", r=prefs["rank"], lora_alpha=int(prefs["alpha"]),
        target_modules=["q_proj", "o_proj"], lora_dropout=0., bias="none", init_lora_weights=True))
    params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if len(params) != base.config.num_hidden_layers * 4 or any("lora_" not in name for name, _ in params):
        raise ValueError("unexpected trainable tensor set")
    examples = compiled["examples"]

    def loss_for(row, network=model):
        tokens = torch.tensor([row["tokens"]], dtype=torch.long)
        labels = torch.tensor(row["labels"][1:], dtype=torch.long)
        logits = network(tokens, use_cache=False).logits[0, :-1]
        return torch.nn.functional.cross_entropy(logits.float(), labels, ignore_index=-100)

    def evaluate(network=model):
        network.eval()
        with torch.no_grad():
            values = [float(loss_for(row, network)) for row in examples]
        if not all(np.isfinite(values)):
            raise ValueError("nonfinite selected-example loss")
        return values

    before = evaluate()
    optimizer = torch.optim.AdamW([p for _, p in params], lr=trainer["learning_rate"],
                                 betas=(.9, .999), eps=1e-8, weight_decay=0.)
    model.train()
    started = time.monotonic()
    for step in range(prefs["steps"]):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for(examples[step % len(examples)])
        if not torch.isfinite(loss):
            raise ValueError("nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for _, p in params], 1., error_if_nonfinite=True)
        optimizer.step()
    seconds = time.monotonic() - started
    if any(p.requires_grad or tensor_hash(p) != digest for p, digest in frozen):
        raise ValueError("frozen base weights changed")
    if any(not torch.isfinite(p).all() for _, p in params):
        raise ValueError("nonfinite learned adapter")
    learned = evaluate()
    model.save_pretrained(folder / "adapter", safe_serialization=True, save_embedding_layers=False)
    reloaded = PeftModel.from_pretrained(Gemma4ForCausalLM.from_pretrained(base_path, local_files_only=True,
        attn_implementation="eager"), folder / "adapter").eval()
    if evaluate(reloaded) != learned:
        raise ValueError("PEFT save/reload changed selected-example losses")
    for module in model.modules():
        if hasattr(module, "set_scale"):
            module.set_scale("default", compiled["training"]["deployment_scale_float32"])
    deployed = evaluate()
    convert("convert_lora_to_gguf.py", ["--base", str(base_path), "--outfile", str(folder / "adapter.gguf"),
                                     str(folder / "adapter")])
    sys.path.insert(0, str(converter / "gguf-py"))
    import gguf
    reader = gguf.GGUFReader(folder / "adapter.gguf")
    factors = {t.name: t.data for t in reader.tensors}
    peft = load_file(folder / "adapter/adapter_model.safetensors")
    expected = {}
    for layer in range(base.config.num_hidden_layers):
        for hf, native in (("q_proj", "attn_q"), ("o_proj", "attn_output")):
            for side in ("A", "B"):
                expected[f"blk.{layer}.{native}.weight.lora_{side.lower()}"] = peft[
                    f"base_model.model.model.layers.{layer}.self_attn.{hf}.lora_{side}.weight"]
    if factors.keys() != expected.keys() or len(peft) != len(expected):
        raise ValueError("converted tensor set differs")
    for name, values in expected.items():
        np.testing.assert_array_equal(factors[name], values)
    alpha = reader.fields["adapter.lora.alpha"]
    if float(alpha.parts[alpha.data[0]][0]) != prefs["alpha"]:
        raise ValueError("converted alpha differs")
    verify_tree(trainer["base_manifest"], base=True)
    verify_tree(trainer["converter_manifest"])
    # Final, hash-bound completion is written only after every check. A supervisor
    # interrupted before recording Candidate can recover this without retraining.
    artifacts = {name: sha256_file(folder / name) for name in (
        "base-check.gguf", "adapter.gguf", "adapter/adapter_config.json", "adapter/adapter_model.safetensors")}
    result = seal({"schema": 1, "execution": compiled["revision"], "completed": True,
        "training_performed": True, "steps_completed": prefs["steps"], "training_seconds": seconds,
        "examples_sha256": hashlib.sha256(json_text(examples).encode()).hexdigest(),
        "trainable_parameters": sum(p.numel() for _, p in params), "artifacts": artifacts,
        "checks": {key: True for key in CHECKS if key != "retained_tokens_and_rng"},
        "loss_before": before, "loss_after_training_scale": learned, "loss_after_deployment_scale": deployed,
        "beneficial_learning_certified": False, "optimizer_reset": True})
    for name in artifacts:
        with (folder / name).open("r+b") as stream:
            os.fsync(stream.fileno())
    write_durable(folder / "result.json.partial", result)
    (folder / "result.json.partial").rename(folder / "result.json")
    from .ending import _sync_directory
    _sync_directory(folder)


if __name__ == "__main__":
    folder = Path(sys.argv[1]).resolve()
    try:
        train(folder)
    except Exception as exc:
        # Private plan feedback, not an operator-visible cognition/log dump.
        execution = json.loads((folder / "input.json").read_text()).get("revision")
        write_durable(folder / "failure.json", seal({"execution": execution,
            "error_type": type(exc).__name__, "reason": str(exc)[:2000]}))
        raise
