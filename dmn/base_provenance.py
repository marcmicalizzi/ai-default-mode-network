"""Reusable, locally reproduced base provenance. Still a tiny CPU-only gate.

The record is a checked local derivation, not a signature or a claim about a
publisher's unknown conversion pipeline. It never authorizes model learning.
"""
from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .backend import sha256_file
from .ending import _owned, _sync_directory
from .learning import _fields
from .sleep_plans import seal
from .storage import write_durable
from .training import KIND, CHECKS, PACKAGES, read_bound_json, validate_recipe, verify_tree
from .training_models import model_profile
from .worker_limits import WorkerLimits, run_cpu_worker

CONVERSION_KEYS = ("python", "python_sha256", "packages", "base_manifest", "converter_manifest",
                   "converter_revision", "inference_name")


def implementation():
    return {name: sha256_file(Path(__file__).with_name(name)) for name in (
        "base_provenance.py", "provenance_native.py", "training_models.py", "training.py",
        "worker_limits.py", "backend.py", "ending.py", "storage.py")}


def verify_quantizer(value):
    _fields(value, "python python_sha256 binding_version binding_source binding_source_sha256 binaries", "quantizer")
    if not value["binaries"]:
        raise ValueError("native quantizer binary identity is empty")
    for name, digest in {value["python"]: value["python_sha256"], value["binding_source"]: value["binding_source_sha256"],
                         **value["binaries"]}.items():
        if not Path(name).is_absolute() or sha256_file(Path(name)) != digest:
            raise ValueError("quantizer asset changed")


def validate_request(request):
    _fields(request, "schema kind conversion quantizer quantization", "provenance request")
    if request["schema"] != 1 or request["kind"] != "tiny_cpu_base_provenance_v1":
        raise ValueError("unsupported provenance request")
    conversion = request["conversion"]
    _fields(conversion, " ".join(CONVERSION_KEYS), "conversion")
    validate_recipe({"schema": 1, "kind": KIND, "parent": {"kind": "native_llama_kv"},
        "resources": {"max_vram_bytes": 0}, "checks": CHECKS, "trainer": {**conversion, "learning_rate": .01, "seed": 0}})
    if request["quantization"] not in {"F32", "Q8_0", "Q4_0", "Q4_K_M"}:
        raise ValueError("unsupported provenance quantization")
    if request["quantization"] == "F32":
        if request["quantizer"] is not None:
            raise ValueError("F32 provenance must not declare quantization")
    else:
        verify_quantizer(request["quantizer"])


def prepare(output, request, limits):
    """Host preparation only: no instance, approval, adapter or inference state."""
    validate_request(request)
    conversion = request["conversion"]
    if sha256_file(Path(conversion["python"])) != conversion["python_sha256"]:
        raise ValueError("conversion interpreter changed")
    verify_tree(conversion["base_manifest"], base=True)
    verify_tree(conversion["converter_manifest"])
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_durable(output / "input.json", seal({"request": request, "implementation": implementation(),
                                             "limits": dataclasses.asdict(limits)}))
    process = run_cpu_worker(conversion["python"], ["-m", "dmn.base_provenance", str(output)],
        cwd=Path(__file__).resolve().parents[1], log=output / "worker.log", limits=limits)
    write_durable(output / "process.json", seal({"result": process, "input_sha256": sha256_file(output / "input.json")}))
    if not process["succeeded"]:
        reason = process["outcome"]
        if (output / "failure.json").exists():
            problem = json.loads((output / "failure.json").read_text())
            if problem.get("input_sha256") == sha256_file(output / "input.json"):
                reason += ": " + problem["reason"]
        raise ValueError("base provenance worker failed: " + reason)
    reference = {"path": str(output / "result.json"), "sha256": sha256_file(output / "result.json")}
    verify(reference, conversion, sha256_file(output / "base.gguf"))
    return reference


def verify(reference, conversion, inference_sha256):
    record = read_bound_json(reference)
    if (seal({k: v for k, v in record.items() if k != "revision"}) != record or
            record.get("schema") != 1 or record.get("completed") is not True or
            record.get("implementation") != implementation()):
        raise ValueError("base provenance completion or implementation changed")
    folder = Path(reference["path"]).resolve().parent
    for name in ("input.json", "process.json"):
        _owned(folder / name, folder)
    source = json.loads((folder / "input.json").read_text())
    process = json.loads((folder / "process.json").read_text())
    if (seal({k: v for k, v in source.items() if k != "revision"}) != source or
            seal({k: v for k, v in process.items() if k != "revision"}) != process or
            process["input_sha256"] != sha256_file(folder / "input.json") or
            record["input_revision"] != source["revision"] or source["implementation"] != implementation() or
            not process["result"]["succeeded"] or process["result"]["limits"] != source["limits"] or
            source["request"] != record["request"]):
        raise ValueError("base provenance supervision or request is incomplete")
    request = record["request"]
    validate_request(request)
    if request["conversion"] != {key: conversion[key] for key in CONVERSION_KEYS}:
        raise ValueError("training source/converter differs from the proven base")
    base = verify_tree(conversion["base_manifest"], base=True)
    verify_tree(conversion["converter_manifest"])
    if model_profile(base, wrapped=True) != record["model_profile"]:
        raise ValueError("provenance model geometry changed")
    if set(record["artifacts"]) != {"base-f32.gguf", "base.gguf"}:
        raise ValueError("unexpected provenance artifacts")
    for name, digest in record["artifacts"].items():
        _owned(folder / name, folder)
        if sha256_file(folder / name) != digest:
            raise ValueError("provenance artifact changed")
    if record["artifacts"]["base.gguf"] != inference_sha256:
        raise ValueError("proven base does not match the active inference GGUF")
    if sha256_file(Path(conversion["python"])) != conversion["python_sha256"]:
        raise ValueError("proven conversion interpreter changed")
    return record, folder


def worker(folder):
    data = json.loads((folder / "input.json").read_text())
    if seal({k: v for k, v in data.items() if k != "revision"}) != data or data["implementation"] != implementation():
        raise ValueError("provenance input or implementation changed")
    request = data["request"]
    validate_request(request)
    conversion = request["conversion"]
    if (sha256_file(Path(sys.executable)) != conversion["python_sha256"] or
            {name: importlib.metadata.version(name) for name in PACKAGES} != conversion["packages"]):
        raise ValueError("conversion environment changed")
    import torch
    if torch.version.cuda is not None:
        raise ValueError("provenance requires CPU-only PyTorch")
    base = verify_tree(conversion["base_manifest"], base=True)
    converter = verify_tree(conversion["converter_manifest"])
    if sum(p.stat().st_size for p in base.iterdir()) > 4 * 1024**2:
        raise ValueError("provenance currently permits only tiny local fixtures")
    profile = model_profile(base, wrapped=True)
    subprocess.run([sys.executable, str(converter / "convert_hf_to_gguf.py"), "--outtype", "f32",
        "--model-name", conversion["inference_name"], "--outfile", str(folder / "base-f32.gguf"), str(base)],
        check=True, stdin=subprocess.DEVNULL, cwd=converter)
    if request["quantization"] == "F32":
        shutil.copyfile(folder / "base-f32.gguf", folder / "base.gguf")
        native = None
    else:
        subprocess.run([request["quantizer"]["python"], "-m", "dmn.provenance_native", str(folder)],
                       check=True, stdin=subprocess.DEVNULL)
        native = json.loads((folder / "native.json").read_text())
        if (native["identity"] != request["quantizer"] or native["source_sha256"] != sha256_file(folder / "base-f32.gguf") or
                native["output_sha256"] != sha256_file(folder / "base.gguf")):
            raise ValueError("native quantization receipt differs")
    sys.path.insert(0, str(converter / "gguf-py"))
    import gguf
    reader = gguf.GGUFReader(folder / "base.gguf")
    counts = {}
    for tensor in reader.tensors:
        name = tensor.tensor_type.name
        counts[name] = counts.get(name, 0) + 1
    expected = "Q4_K" if request["quantization"] == "Q4_K_M" else request["quantization"]
    if not counts.get(expected):
        raise ValueError("requested tensor quantization was not actually produced")
    verify_tree(conversion["base_manifest"], base=True)
    verify_tree(conversion["converter_manifest"])
    validate_request(request)
    artifacts = {name: sha256_file(folder / name) for name in ("base-f32.gguf", "base.gguf")}
    for name in artifacts:
        with (folder / name).open("r+b") as stream:
            os.fsync(stream.fileno())
    write_durable(folder / "result.json.partial", seal({"schema": 1, "completed": True, "request": request,
        "input_revision": data["revision"], "implementation": implementation(), "model_profile": profile,
        "artifacts": artifacts, "tensor_types": counts, "quantizer_result": native,
        "scope": "tiny_cpu_only; reproduction of these artifacts, not a publisher attestation"}))
    (folder / "result.json.partial").rename(folder / "result.json")
    _sync_directory(folder)


if __name__ == "__main__":
    folder = Path(sys.argv[1]).resolve()
    try:
        worker(folder)
    except Exception as exc:
        write_durable(folder / "failure.json", {"input_sha256": sha256_file(folder / "input.json"),
            "reason": type(exc).__name__ + ": " + str(exc)[:2000]})
        raise
