"""CPU conversion stages for reviewed NF4 factors; no inference or instance access."""
import json
from pathlib import Path
import subprocess
import sys

from .backend import sha256_file
from .gpu_training_worker import request
from .sleep_plans import seal
from .storage import write_durable
from .training import verify_tree


def convert(folder, parent=False):
    compiled = request(folder)
    trainer = compiled["recipe"]["trainer"]
    converter = verify_tree(trainer["converter_manifest"])
    from .exact_base_provenance import verify
    _, base = verify(trainer["provenance_manifest"], trainer, compiled["parent"]["model_sha256"])
    if parent:
        # Preserve original model-card metadata for byte-identical parent GGUF.
        from .gpu_recipe import MAX_ADAPTER_BYTES
        adapter = verify_tree(trainer["parent_adapter_manifest"], adapter=True, adapter_limit_bytes=MAX_ADAPTER_BYTES)
    else:
        adapter = folder / "adapter"
    output = folder / ("parent-check.gguf" if parent else "adapter.gguf")
    subprocess.run([sys.executable, str(converter / "convert_lora_to_gguf.py"), "--outtype", "f32",
                    "--base", str(base), "--outfile", str(output), str(adapter)], check=True, stdin=subprocess.DEVNULL)
    if parent:
        if sha256_file(output) != compiled["lineage"]["parent_adapter"]["sha256"]:
            raise ValueError("parent PEFT files do not reproduce the deployed adapter")
        return
    sys.path.insert(0, str(converter / "gguf-py"))
    import gguf
    import numpy as np
    from safetensors.numpy import load_file
    reader = gguf.GGUFReader(output)
    actual = {t.name: t.data for t in reader.tensors}
    factors = load_file(adapter / "adapter_model.safetensors")
    expected = {}
    profile = compiled["training"]["model_profile"]
    for layer in range(profile["num_hidden_layers"]):
        for hf, native in (("q_proj", "attn_q"), ("o_proj", "attn_output")):
            for side in ("A", "B"):
                expected[f"blk.{layer}.{native}.weight.lora_{side.lower()}"] = factors[
                    f"base_model.model.{profile['text_prefix']}.layers.{layer}.self_attn.{hf}.lora_{side}.weight"]
    if actual.keys() != expected.keys() or len(factors) != len(expected):
        raise ValueError("converted factor set differs from the reviewed targets")
    for name, values in expected.items():
        np.testing.assert_array_equal(actual[name], values)
    if reader.fields["adapter.lora.alpha"].contents() != compiled["preferences"]["alpha"]:
        raise ValueError("converted alpha differs")
    verify_tree(trainer["converter_manifest"])
    write_durable(folder / "converted.json", seal({"execution": compiled["revision"], "completed": True,
        "factor_count": len(expected), "factors_exact": True, "alpha_equal": True,
        "adapter_sha256": sha256_file(output), "peft_sha256": sha256_file(adapter / "adapter_model.safetensors")}))


if __name__ == "__main__":
    phase, folder = sys.argv[1], Path(sys.argv[2]).resolve()
    if phase not in {"parent", "convert"}:
        raise ValueError("unknown conversion stage")
    try:
        convert(folder, parent=phase == "parent")
    except Exception as exc:
        write_durable(folder / "failure.json", {"phase": phase, "error_type": type(exc).__name__, "reason": str(exc)[:2000]})
        raise
