"""Review contract for static-placement Gemma NF4 training.

Compilation does not enable execution. The production supervisor must separately
enforce the reviewed resource envelope before this recipe can run for an instance.
"""
from __future__ import annotations

import copy
import ctypes
import math
from pathlib import Path

from .backend import sha256_file
from .learning import _fields
from .training import (GPU_KIND, KIND_V2, PACKAGES as CPU_PACKAGES, CHECKS,
                       parent_adapter_config, verify_tree)
from .training_models import model_profile
from .worker_limits import WorkerLimits
from .qlora_prepare import ALLOCATOR
from .training_artifacts import MAX_ADAPTER_BYTES

PACKAGES = (*CPU_PACKAGES, "bitsandbytes", "accelerate")
SCOPE = "reviewed_nf4_training_v1"


def validate_recipe(value):
    _fields(value, "schema kind parent resources checks trainer", "NF4 recipe")
    if value["schema"] != 1 or value["kind"] != GPU_KIND or value["checks"] != CHECKS:
        raise ValueError("unsupported NF4 recipe or checks")
    trainer = value["trainer"]
    gpu = trainer.get("gpu")
    _fields(gpu, "torch_vram_bytes max_sequence_tokens max_steps max_rank", "GPU limits")
    for key, limit in gpu.items():
        if type(limit) is not int or limit < 1:
            raise ValueError("GPU recipe limits must be positive integers")
    if (gpu["max_sequence_tokens"] > 256 or gpu["max_steps"] > 64 or gpu["max_rank"] > 2 or
            gpu["torch_vram_bytes"] > 24 * 1024**3):
        raise ValueError("NF4 recipe exceeds the supported workload envelope")
    if set(trainer.get("packages", {})) != set(PACKAGES):
        raise ValueError("NF4 recipe requires explicit versions including bitsandbytes and accelerate")
    if any(not isinstance(v, str) or not v for v in trainer["packages"].values()):
        raise ValueError("invalid NF4 package version")
    _fields(value["resources"], "max_training_seconds max_ram_bytes max_vram_bytes max_disk_bytes", "resources")
    if any(type(v) is not int or v < 1 for v in value["resources"].values()):
        raise ValueError("NF4 resource ceilings must be positive integers")
    # Include explicit room for CUDA/context/library allocations outside Torch.
    # This margin is not misrepresented as a total-process GPU allocation quota.
    if value["resources"]["max_vram_bytes"] < gpu["torch_vram_bytes"] + 1024**3:
        raise ValueError("GPU envelope requires at least 1 GiB beyond the Torch allocator ceiling")
    WorkerLimits(value["resources"]["max_ram_bytes"], value["resources"]["max_training_seconds"])
    # Reuse the structural parent, source, converter and parameter checks. This
    # validates shape only; CPU provenance/execution is never substituted.
    cpu = copy.deepcopy(value)
    cpu["kind"] = KIND_V2
    cpu["resources"]["max_vram_bytes"] = 0
    del cpu["trainer"]["gpu"]
    cpu["trainer"]["packages"] = {k: trainer["packages"][k] for k in CPU_PACKAGES}
    from .training import validate_recipe as validate_cpu
    validate_cpu(cpu)


def compile_training(value):
    from .exact_base_provenance import verify
    recipe, prefs = value["recipe"], value["preferences"]
    validate_recipe({k: v for k, v in recipe.items() if k != "revision"})
    trainer, gpu = recipe["trainer"], recipe["trainer"]["gpu"]
    for key in ("rank", "steps"):
        if type(prefs[key]) is not int or not 1 <= prefs[key] <= gpu["max_" + key]:
            raise ValueError("requested " + key + " exceeds the offered NF4 envelope; no silent reduction")
    if (type(prefs["alpha"]) not in (int, float) or not math.isfinite(prefs["alpha"]) or
            prefs["alpha"] != int(prefs["alpha"]) or not 1 <= prefs["alpha"] <= 1024):
        raise ValueError("NF4 recipe requires integer alpha 1..1024")
    scale = ctypes.c_float(prefs["scale"]).value
    if not math.isfinite(scale) or (prefs["scale"] != 0 and scale == 0):
        raise ValueError("deployment strength is not representable")
    for row in value["examples"]:
        if not 2 <= len(row["tokens"]) <= gpu["max_sequence_tokens"]:
            raise ValueError("example exceeds the offered NF4 length; no truncation or splitting")
    proof, base = verify(trainer["provenance_manifest"], trainer, value["parent"]["model_sha256"], verify_assets=False)
    profile = model_profile(base, wrapped=True)
    lineage, training_scale = None, 1.
    if value["parent"].get("lora_adapters"):
        parent_path = verify_tree(trainer["parent_adapter_manifest"], adapter=True,
                                  adapter_limit_bytes=MAX_ADAPTER_BYTES)
        config = parent_adapter_config(parent_path, targets=profile["target_modules"])
        old = value["parent"]["lora_adapters"][0]
        if (prefs["rank"] != config["r"] or prefs["alpha"] != config["lora_alpha"] or scale != old["scale"]):
            raise ValueError("continuation preserves parent rank, alpha and deployment strength")
        training_scale = old["scale"]
        lineage = {"parent_adapter": old, "parent_peft": {name: sha256_file(parent_path / name) for name in
            ("adapter_config.json", "adapter_model.safetensors")}, "rank": config["r"], "alpha": config["lora_alpha"]}
    value.update(execution_scope=SCOPE, training_requested=True, lineage=lineage,
        native_wake={'max_seconds': value['resources']['max_training_seconds'],
                     'additional_to_training_chain': True, 'sampling': False,
                     'ram_and_device_allowances': 'same_as_training; inference is closed before worker launch'},
        adapter_operation=("continue_single_adapter; preserve factors/rank/alpha/strength; replace parent, never stack"
                           if lineage else "first_adapter_only; existing adapters rejected"),
        resource_enforcement="Windows job RAM/process/time limits; Torch allocator ceiling; whole-device VRAM watchdog (other applications count, transient overshoot between polls remains possible); bounded trusted adapter serialization and native-workspace preflight. No OS filesystem or GPU allocation quota. Execution requires the explicitly enabled supervised service and current host resource offer.",
        limitation="No guarantee of benefit or retained prior learning. Exact retained-text reconstruction only; visual positions prevent deep sleep. No implicit source selection, truncation or extra steps.",
        training={"dtype": "bfloat16_compute_frozen_nf4_with_f32_nonquantized_parameters", "device": "cuda:0",
            "device_map": ({"": 0, "model.vision_tower": "cpu", "model.embed_vision": "cpu"}
                           if profile["architecture"] == "Gemma4ForConditionalGeneration" else {"": 0}),
            "threads": 2, "batch_size": 1, "example_order": "round_robin_in_reviewed_order",
            "target_modules": profile["target_modules"], "model_profile": profile,
            "optimizer": "AdamW", "optimizer_reset": True, "betas": [.9, .999], "epsilon": 1e-8,
            "weight_decay": 0., "max_gradient_norm": 1., "dropout": 0.,
            "training_scale": training_scale, "deployment_scale_float32": scale,
            "quantization": "NF4", "nested_quantization": True, "gradient_checkpointing": "non-reentrant",
            "allocator_configuration": ALLOCATOR, "torch_vram_bytes": gpu["torch_vram_bytes"],
            "max_sequence_tokens": gpu["max_sequence_tokens"], "source_loader": "bounded_tensor_stream_v1",
            "loss": "mean_cross_entropy_on_shifted_target_labels_only; no padding/truncation",
            "evaluation": "selected-example loss at training and deployment scale; no heldout or benefit guarantee",
            "base_provenance": {"revision": proof["revision"], "method": proof["method"]}})
    return value


def read_completion(folder, compiled):
    import hashlib
    import json
    from .ending import _owned
    from .sleep_plans import seal
    from .storage import json_text

    def receipt(name):
        path = folder / name
        _owned(path, folder)
        if path.stat().st_size > 1024**2:
            raise ValueError("oversized NF4 completion receipt")
        value = json.loads(path.read_text())
        if (seal({k: v for k, v in value.items() if k != "revision"}) != value or
                value.get("execution") != compiled["revision"]):
            raise ValueError("NF4 completion identity changed")
        return value

    result, trained, reloaded, converted = [receipt(name) for name in
        ("result.json", "trained.json", "reload.json", "converted.json")]
    if any(record.get("completed") is not True for record in (result, trained, reloaded, converted)):
        raise ValueError("incomplete NF4 stages")
    expected = {"trained.json", "reload.json", "converted.json", "provenance.json", "adapter.gguf",
                "adapter/adapter_config.json", "adapter/adapter_model.safetensors"}
    lineage = compiled.get("lineage")
    if lineage:
        expected |= {"parent-check.gguf", "parent-adapter/adapter_config.json", "parent-adapter/adapter_model.safetensors"}
    if set(result["artifacts"]) != expected:
        raise ValueError("unexpected NF4 completion artifacts")
    for name, digest in result["artifacts"].items():
        path = folder
        for part in name.split("/"):
            child = path / part
            _owned(child, path)
            path = child
        if path.stat().st_size > MAX_ADAPTER_BYTES or sha256_file(path) != digest:
            raise ValueError("NF4 completion artifact changed or exceeds limit")
    if (result["artifacts"]["provenance.json"] != compiled["recipe"]["trainer"]["provenance_manifest"]["sha256"] or
            trained.get("frozen_state_unchanged") is not True or reloaded.get("trained_revision") != trained["revision"] or
            reloaded.get("factors_exact") is not True or reloaded.get("selected_losses_exact") is not True or
            converted.get("factors_exact") is not True or converted.get("alpha_equal") is not True or
            converted.get("factor_count") != len(compiled["training"]["target_modules"]) * 2 or
            converted.get("adapter_sha256") != result["artifacts"]["adapter.gguf"] or
            converted.get("peft_sha256") != result["artifacts"]["adapter/adapter_model.safetensors"] or
            trained.get("adapter_sha256") != converted["peft_sha256"] or
            trained.get("adapter_config_sha256") != result["artifacts"]["adapter/adapter_config.json"]):
        raise ValueError("NF4 conversion/reload/provenance checks differ")
    if (result.get("training_performed") is not True or result.get("lineage") != lineage or
            result.get("steps_completed") != compiled["preferences"]["steps"] or
            result.get("steps_completed") != trained.get("steps_completed") or
            result.get("examples_sha256") != hashlib.sha256(json_text(compiled["examples"]).encode()).hexdigest() or
            result.get("examples_sha256") != trained.get("examples_sha256") or
            result.get("checks") != {key: True for key in CHECKS if key != "retained_tokens_and_rng"}):
        raise ValueError("NF4 completed workload differs from reviewed plan")
    if lineage and (result["artifacts"]["parent-check.gguf"] != lineage["parent_adapter"]["sha256"] or
                    trained.get("parent_factors_loaded_exactly") is not True or
                    any(result["artifacts"]["parent-adapter/" + name] != digest for name, digest in lineage["parent_peft"].items())):
        raise ValueError("NF4 continuation lineage differs")
    combined = receipt("process.json")["result"]
    stages = (["parent"] if lineage else []) + ["train", "reload", "convert"]
    if (combined.get("succeeded") is not True or combined.get("active_processes") != 0 or
            set(combined.get("stages", {})) != set(stages) or
            combined.get("limits") != {"max_committed_bytes": compiled["resources"]["max_ram_bytes"],
                                       "max_seconds": compiled["resources"]["max_training_seconds"]} or
            not 0 <= combined.get("elapsed_seconds", -1) <= compiled["resources"]["max_training_seconds"] + 1):
        raise ValueError("NF4 process supervision is incomplete")
    total_seconds = 0.
    for stage in stages:
        process = receipt(stage + "-process.json")["result"]
        if (process != combined["stages"][stage] or not process["succeeded"] or process["active_processes"] != 0 or
                process["returncode"] != 0 or process["outcome"] != "exited" or
                process["limits"]["max_committed_bytes"] != compiled["resources"]["max_ram_bytes"] or
                process.get("memory_enforcement") != "windows_job_aggregate_commit" or
                process.get("wall_time_enforcement") != "supervisor_watchdog" or
                not 0 <= process["elapsed_seconds"] <= process["limits"]["max_seconds"] + 1 or
                not 0 < process["limits"]["max_seconds"] <= compiled["resources"]["max_training_seconds"]):
            raise ValueError("NF4 stage supervision differs from the reviewed limits")
        total_seconds += process["elapsed_seconds"]
        if stage in {'train', 'reload'}:
            observed = process.get('device_memory', {})
            if (observed.get('max_bytes') != compiled['resources']['max_vram_bytes'] or
                    observed.get('scope') != 'entire_device' or
                    observed.get('transient_overshoot_possible') is not True or
                    not 0 <= observed.get('observed_peak_bytes', -1) <= observed['max_bytes']):
                raise ValueError('NF4 device memory supervision differs from the reviewed allowance')
    if total_seconds > compiled["resources"]["max_training_seconds"] + 1:
        raise ValueError("NF4 stages exceeded the reviewed duration")
    return result
