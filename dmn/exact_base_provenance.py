"""Cached exact tensor-payload evidence for the explicitly audited Gemma4 source.

This records numerical derivation, not an unknown publisher's command line or
byte-identical file metadata. Chat-template differences are retained explicitly:
training uses reviewed token IDs and waking never renders another template.
"""
from __future__ import annotations

import json
from pathlib import Path

from .backend import sha256_file
from .learning import _fields
from .sleep_plans import seal
from .storage import write_durable
from .training import read_bound_json, verify_tree
from .training_models import model_profile

METHOD = "exact_payload_and_inference_metadata_v1"
SOURCE_REPO = "llmfan46/gemma-4-31B-it-uncensored-heretic"
SOURCE_REVISION = "d5bfc0d99e308beb9805440806161ad0233df357"
BASE_SHA = "7c65a35e7c4e53cba6c5e02cc9eeb850eb4251f4d9ad120c2caa6de23c5a6395"
AUDITS = ("tensor", "tokenizer", "metadata")


def implementation():
    return {name: sha256_file(Path(__file__).with_name(name))
            for name in ("exact_base_provenance.py", "training_models.py", "safetensor_stream.py")}


def read_audit(folder):
    folder = Path(folder).resolve()
    data = {}
    references = {}
    for name in ("input.json", "result.json", "process.json"):
        path = folder / name
        ref = {"path": str(path), "sha256": sha256_file(path)}
        data[name] = read_bound_json(ref)
        references[name] = ref
    process = data["process.json"]
    if (process.get("succeeded") is not True or process.get("active_processes") != 0 or
            process.get("returncode") != 0 or process.get("outcome") != "exited"):
        raise ValueError("base audit lacks successful complete worker supervision")
    return data, references


def check_evidence(audits, base, source, inference, converter, *, hash_source=True):
    tensors, tokens, metadata = [audits[k]["result.json"] for k in AUDITS]
    requests = {k: audits[k]["input.json"] for k in AUDITS}
    for result in (tensors, tokens, metadata):
        if result.get("completed") is not True or result.get("gguf_sha256") != inference:
            raise ValueError("audit completion or inference identity differs")
    if (inference != BASE_SHA or source.get("repo") != SOURCE_REPO or source.get("revision") != SOURCE_REVISION or
            tensors.get("source_revision") != SOURCE_REVISION):
        raise ValueError("this provenance policy requires the explicitly audited source revision and GGUF")
    for kind in ("tensor", "tokenizer"):
        if (Path(requests[kind]["source"]).resolve() / "model" != base.resolve() or
                Path(requests[kind]["gguf_python"]).resolve() != converter.resolve() / "gguf-py"):
            raise ValueError("audit source or converter differs from bound training assets")
    if (Path(requests["metadata"]["source"]).resolve() != base.resolve() or
            Path(requests["metadata"]["converter"]).resolve() != converter.resolve()):
        raise ValueError("metadata audit used different assets")
    if (requests["tensor"].get("full") is not True or
            tensors.get("full_tensor_payload_comparison") is not True or
            tensors.get("all_compared_bytes_equal") is not True or tensors.get("mismatches") != [] or
            tensors.get("tensor_types") != {"F32": 421, "Q4_K": 355, "Q6_K": 56}):
        raise ValueError("complete exact tensor comparison is required; sampled evidence cannot enable training")
    if (tokens.get("tokens_scores_types_equal") is not True or tokens.get("special_tokens_equal") is not True or
            tokens.get("vocabulary_entries") != 262144 or not isinstance(tokens.get("chat_template_equal"), bool)):
        raise ValueError("complete token vocabulary and special-token comparison is required")
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if tokens.get("source_files", {}).get(name) != sha256_file(base / name):
            raise ValueError("tokenizer assets changed since the audit")
    if (metadata.get("all_inference_metadata_equal") is not True or metadata.get("keys_compared") != 21 or
            metadata.get("mismatches") != [] or metadata.get("source_config_sha256") != sha256_file(base / "config.json")):
        raise ValueError("source-derived inference metadata differs or changed")
    quantizer = Path(requests["tensor"]["ggml_base"])
    if sha256_file(quantizer) != tensors.get("quantizer_sha256"):
        raise ValueError("audited quantizer changed")
    # Verify the actual downloaded payloads against the pinned source listing;
    # a similarly named directory or modified source is never interchangeable.
    for name, entry in source["files"].items():
        path = base / name
        if path.stat().st_size != entry["size"]:
            raise ValueError("pinned source asset size differs")
        if hash_source and entry.get("lfs_sha256") and sha256_file(path) != entry["lfs_sha256"]:
            raise ValueError("pinned source asset hash differs")
    return {"source": {"repo": SOURCE_REPO, "revision": SOURCE_REVISION},
            "tensor_payloads_bit_equal": True, "inference_metadata_equal": True,
            "token_vocabulary_equal": True, "chat_template_equal": tokens["chat_template_equal"],
            "source_chat_template_sha256": tokens["source_chat_template_sha256"],
            "gguf_chat_template_sha256": tokens["gguf_chat_template_sha256"],
            "template_policy": "retain original inference template and exact token IDs; never substitute the HF template",
            "full_gguf_file_reproduced": False}


def prepare(output, trainer, *, tensor_audit, tokenizer_audit, metadata_audit):
    """Assemble checked local audit evidence; never reads an instance or trains."""
    output = Path(output).resolve()
    base = verify_tree(trainer["base_manifest"], base=True, allow_metadata=True)
    converter = verify_tree(trainer["converter_manifest"])
    source_path = base.parent / "source.json"
    source_ref = {"path": str(source_path), "sha256": sha256_file(source_path)}
    source = read_bound_json(source_ref)
    audits, references = {}, {}
    for kind, folder in zip(AUDITS, (tensor_audit, tokenizer_audit, metadata_audit)):
        audits[kind], references[kind] = read_audit(folder)
    evidence = check_evidence(audits, base, source, BASE_SHA, converter)
    for path in {Path(audits[kind]["input.json"]["gguf"]).resolve() for kind in AUDITS}:
        if sha256_file(path) != BASE_SHA:
            raise ValueError("audited inference artifact changed")
    record = seal({"schema": 1, "method": METHOD, "completed": True, "implementation": implementation(),
        "base_manifest": trainer["base_manifest"], "converter_manifest": trainer["converter_manifest"],
        "source_manifest": source_ref, "audits": references, "inference_sha256": BASE_SHA,
        "model_profile": model_profile(base, wrapped=True), "evidence": evidence})
    if output.exists():
        raise ValueError("provenance output must be fresh")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_durable(output, record)
    return {"path": str(output), "sha256": sha256_file(output)}


def verify(reference, trainer, inference_sha256, *, verify_assets=True):
    record = read_bound_json(reference)
    if record.get("request", {}).get("kind") == "tiny_cpu_base_provenance_v1":
        # Reuse the stronger whole-file reproduction proof for generated tiny
        # fixtures. Its conversion environment stays separately pinned; NF4
        # training must not pretend that its CUDA interpreter performed it.
        from .base_provenance import verify as verify_reproduction
        conversion = record["request"]["conversion"]
        if any(trainer[key] != conversion[key] for key in ("base_manifest", "converter_manifest")):
            raise ValueError("NF4 training assets differ from the reproduced tiny base")
        proven, _ = verify_reproduction(reference, conversion, inference_sha256)
        base = Path(read_bound_json(trainer["base_manifest"])["root"])
        return {**proven, "method": "exact_gguf_reproduction_v1"}, base
    _fields(record, "schema method completed implementation base_manifest converter_manifest source_manifest audits inference_sha256 model_profile evidence revision", "exact provenance")
    if (seal({k: v for k, v in record.items() if k != "revision"}) != record or record["schema"] != 1 or
            record["method"] != METHOD or record["completed"] is not True or record["implementation"] != implementation() or
            record["inference_sha256"] != inference_sha256 or
            any(record[key] != trainer[key] for key in ("base_manifest", "converter_manifest"))):
        raise ValueError("exact provenance, implementation or training asset binding changed")
    # Compilation binds immutable manifests without hashing tens of GB on the
    # inference thread. Every actual training/reload worker rechecks all bytes.
    if verify_assets:
        base = verify_tree(trainer["base_manifest"], base=True, allow_metadata=True)
        converter = verify_tree(trainer["converter_manifest"])
    else:
        base = Path(read_bound_json(trainer["base_manifest"])["root"])
        converter = Path(read_bound_json(trainer["converter_manifest"])["root"])
    source = read_bound_json(record["source_manifest"])
    if set(record["audits"]) != set(AUDITS):
        raise ValueError("missing audit evidence")
    audits = {}
    for kind, refs in record["audits"].items():
        if set(refs) != {"input.json", "result.json", "process.json"}:
            raise ValueError("incomplete audit evidence")
        audits[kind] = {name: read_bound_json(ref) for name, ref in refs.items()}
        process = audits[kind]["process.json"]
        if not process.get("succeeded") or process.get("active_processes") != 0:
            raise ValueError("audit supervision changed")
    if (check_evidence(audits, base, source, inference_sha256, converter, hash_source=False) != record["evidence"] or
            model_profile(base, wrapped=True) != record["model_profile"]):
        raise ValueError("exact provenance evidence changed")
    return record, base
