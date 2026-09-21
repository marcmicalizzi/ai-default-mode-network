"""CPU-only second learning cycle and quantized-base transfer on the tiny fixture.

Consumes a successful probe_lora_training output, never an instance. Two
disposable candidates continue the same adapter: new examples alone, or new
examples with an explicit replay set. Optimizer state is reset in both cases.
"""
from __future__ import annotations

import argparse
from collections import Counter
import ctypes as C
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.probe_lora_training import (CONVERTER_REVISION, cpu_environment, digest,
    drift, open_native, run_child, scores, verify_factors, write_json)


def validate_parent(parent):
    report = json.loads((parent / "report.json").read_text())
    if not report.get("completed") or not report.get("synthetic_only"):
        raise ValueError("a completed synthetic training probe is required")
    config = json.loads((parent / "base/config.json").read_text())
    for key, expected in (("hidden_size", 64), ("num_hidden_layers", 6), ("vocab_size", 263)):
        if config.get(key) != expected:
            raise ValueError("only the generated tiny Gemma4 fixture is allowed")
    paths = ("base/model.safetensors", "base/config.json", "base/tokenizer.json",
             "base/tokenizer_config.json", "base.gguf", "adapter.gguf",
             "adapter/adapter_model.safetensors", "adapter/adapter_config.json", "examples.json")
    identity = {}
    for name in paths:
        path = parent / name
        if path.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("input exceeds tiny fixture size limit")
        identity[name] = digest(path)
        if identity[name] != report["artifact_hashes"].get(name):
            raise ValueError(f"parent artifact changed: {name}")
    adapter = json.loads((parent / "adapter/adapter_config.json").read_text())
    if (adapter["r"] != 2 or adapter["lora_alpha"] != 4 or
            set(adapter["target_modules"]) != {"q_proj", "o_proj"}):
        raise ValueError("unexpected parent adapter recipe")
    return identity


def datasets(parent):
    import numpy as np
    original = json.loads((parent / "examples.json").read_text())
    data = {"old_train": original["train"], "old_test": original["confirmation"],
            "control": original["control"]}
    rng = np.random.default_rng(58312)
    for split, n, low, high in (("new_train", 128, 60, 100), ("new_test", 64, 100, 140)):
        data[split] = [{"tokens": [1, *rng.integers(low, high, 6).tolist(), 13 + i % 2],
                        "target": 43 + i % 2} for i in range(n)]
    return data


def train_candidates(parent, output, data):
    import numpy as np
    import psutil
    import torch
    from peft import PeftModel
    from transformers import Gemma4ForCausalLM
    if torch.version.cuda is not None:
        raise ValueError("this experiment requires CPU-only PyTorch")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    report = {}
    parent_logits = None
    for name in ("new_only", "with_replay"):
        torch.manual_seed(915)
        base = Gemma4ForCausalLM.from_pretrained(parent / "base", local_files_only=True,
                                               attn_implementation="eager").float().cpu()
        frozen = [(p, p.detach().clone()) for p in base.parameters()]
        model = PeftModel.from_pretrained(base, parent / "adapter", is_trainable=True)
        params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        if sum(p.numel() for _, p in params) != 5120 or any("lora_" not in n for n, _ in params):
            raise AssertionError("unexpected trainable parameters")

        def evaluate():
            model.eval()
            with torch.no_grad():
                return {split: model(torch.tensor([r["tokens"] for r in data[split]]),
                                     use_cache=False).logits[:, -1].numpy().copy()
                        for split in ("old_test", "new_test", "control")}

        starting_logits = evaluate()
        if parent_logits is None:
            parent_logits = starting_logits
        for split in parent_logits:
            np.testing.assert_array_equal(starting_logits[split], parent_logits[split])
        # Both recipes have the same compute/step budget. Replay spends half
        # that budget on explicitly selected old examples, not hidden history.
        optimizer = torch.optim.AdamW([p for _, p in params], lr=.01, weight_decay=0)
        model.train()
        started = time.perf_counter()
        losses = []
        for step in range(256):
            if name == "new_only":
                start = (step % 16) * 8
                rows = data["new_train"][start:start + 8]
            else:
                start = (step % 32) * 4
                rows = data["new_train"][start:start + 4] + data["old_train"][start:start + 4]
            optimizer.zero_grad(set_to_none=True)
            logits = model(torch.tensor([r["tokens"] for r in rows]), use_cache=False).logits[:, -1]
            loss = torch.nn.functional.cross_entropy(logits, torch.tensor([r["target"] for r in rows]))
            if not torch.isfinite(loss):
                raise AssertionError("nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for _, p in params], 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        seconds = time.perf_counter() - started
        if any(p.requires_grad or not torch.equal(p, before) for p, before in frozen):
            raise AssertionError("base weights changed")
        candidate = output / name
        candidate.mkdir()
        learned = evaluate()
        model.save_pretrained(candidate / "adapter", safe_serialization=True, save_embedding_layers=False)
        reloaded = PeftModel.from_pretrained(Gemma4ForCausalLM.from_pretrained(
            parent / "base", local_files_only=True, attn_implementation="eager"), candidate / "adapter").eval()
        with torch.no_grad():
            for split, expected in learned.items():
                actual = reloaded(torch.tensor([r["tokens"] for r in data[split]]),
                                  use_cache=False).logits[:, -1].numpy()
                np.testing.assert_array_equal(actual, expected)
        np.savez(candidate / "reference-logits.npz", **learned)
        report[name] = {"training_seconds": seconds, "steps": 256, "batch_size": 8,
            "learning_rate": .01, "optimizer_reset": True, "rank": 2, "trainable_parameters": 5120,
            "new_supervised_tokens": 2048 if name == "new_only" else 1024,
            "replay_supervised_tokens": 0 if name == "new_only" else 1024,
            "loss_first": losses[0], "loss_last": losses[-1],
            "scores": {k: scores(v, data[k]) for k, v in learned.items()},
            "control_drift_from_parent": drift(parent_logits["control"], learned["control"]),
            "base_unchanged": True, "peft_reload_exact": True,
            "parent_adapter_sha256": digest(parent / "adapter/adapter_model.safetensors"),
            "candidate_adapter_sha256": digest(candidate / "adapter/adapter_model.safetensors")}
        write_json(candidate / "training.json", report[name])
        del model, reloaded, optimizer, base, frozen, params
    report["parent_scores"] = {k: scores(v, data[k]) for k, v in parent_logits.items()}
    report["process_peak_rss_bytes"] = getattr(psutil.Process().memory_info(), "peak_wset", None)
    write_json(output / "training.json", report)
    return report


def native_matrix(output, parent):
    import numpy as np
    import llama_cpp.llama_cpp as api
    from dmn.native_logging import configure_native_logging
    configure_native_logging(api)
    # Check the name with the existing CPU-only fixture guard before quantizing.
    guard = open_native(parent)
    guard.close()
    data = json.loads((output / "examples.json").read_text())
    models = output / "models"
    models.mkdir()
    shutil.copyfile(parent / "base.gguf", models / "f32.gguf")
    for name, ftype in (("q8_0", api.LLAMA_FTYPE_MOSTLY_Q8_0), ("q4_0", api.LLAMA_FTYPE_MOSTLY_Q4_0)):
        params = api.llama_model_quantize_default_params()
        params.nthread, params.ftype, params.pure = 1, ftype, True
        if api.llama_model_quantize(os.fsencode(parent / "base.gguf"),
                                  os.fsencode(models / f"{name}.gguf"), C.byref(params)):
            raise RuntimeError(f"{name} base quantization failed")
    report = {"completed": False, "evaluations": {}, "cpu_threads": 1, "gpu_layers": 0}
    saved_logits = {}
    for name, adapter in (("parent", parent / "adapter.gguf"),
                          ("new_only", output / "new_only/adapter.gguf"),
                          ("with_replay", output / "with_replay/adapter.gguf")):
        for precision in ("f32", "q8_0", "q4_0"):
            folder = output / "native" / name / precision
            folder.mkdir(parents=True)
            shutil.copyfile(models / f"{precision}.gguf", folder / "base.gguf")
            shutil.copyfile(adapter, folder / "adapter.gguf")
            backend = open_native(folder, 1., compact=True)
            values = {}
            try:
                for split in ("old_test", "new_test", "control"):
                    rows = []
                    for row in data[split]:
                        backend.api.llama_memory_clear(backend.memory, True)
                        backend.tokens, backend.logits = [], None
                        backend.eval(row["tokens"])
                        rows.append(backend.logits.copy())
                    values[split] = np.stack(rows)
                reference = values if precision == "f32" else saved_logits[name]
                report["evaluations"][f"{name}_{precision}"] = {
                    "scores": {k: scores(v, data[k]) for k, v in values.items()},
                    "drift_from_f32_base": {k: drift(reference[k], v) for k, v in values.items()},
                    "base_sha256": digest(folder / "base.gguf"),
                    "adapter_sha256": digest(folder / "adapter.gguf"),
                    "native_fingerprint": backend.fingerprint}
                if precision == "f32":
                    saved_logits[name] = values
            finally:
                backend.close()
            write_json(output / "native.json", report)
    report["wake"] = second_wake(output)
    report["completed"] = True
    write_json(output / "native.json", report)


def second_wake(output):
    import numpy as np
    from dmn.recovery import restore_checkpoint
    from scripts.probe_lora_wake import save, continuation
    parent = output / "native/parent/q4_0"
    target = output / "native/with_replay/q4_0"
    backend = open_native(parent, 1., compact=True)
    try:
        backend.eval(backend.tokenize('Synthetic retained history. <dmn_action>{"op":"send_message",'
                                      '"content":"Historical text only."}</dmn_action> ' * 3, initial=True))
        backend.shift(16, 64)
        backend.eval([11])
        tokens, rng = list(backend.tokens), backend.rng.getstate()
        save(backend, output / "before-second-wake", 1)
    finally:
        backend.close()
    backend = open_native(target, 1., compact=True)
    try:
        try:
            restore_checkpoint(backend, output / "before-second-wake", "strict")
        except ValueError as exc:
            if "environment differs" not in str(exc):
                raise
        else:
            raise AssertionError("ordinary restore accepted the old configuration")
        if backend.tokens or backend.decode_calls:
            raise AssertionError("strict rejection decoded old state")
        evidence = backend.rebuild(output / "before-second-wake")
        if backend.tokens != tokens or backend.rng.getstate() != rng:
            raise AssertionError("second wake changed tokens or RNG")
        logits = backend.logits.copy()
        save(backend, output / "after-second-wake", 1)
        next_tokens, next_logits = continuation(backend)
        write_json(output / "continuation.json", next_tokens)
        np.save(output / "continuation.npy", next_logits, allow_pickle=False)
    finally:
        backend.close()
    backend = open_native(target, 1., compact=True)
    try:
        backend.eval(tokens)
        np.testing.assert_array_equal(backend.logits, logits)
    finally:
        backend.close()
    run_child([sys.executable, __file__, "--restart", output], output / "restart.log")
    return {"rebuild": evidence, "retained_tokens_and_rng_equal": True,
            "fresh_rebuild_logits_equal": True,
            "restart": json.loads((output / "restart.json").read_text())}


def restart(output):
    import numpy as np
    from dmn.recovery import restore_checkpoint
    from scripts.probe_lora_wake import continuation
    backend = open_native(output / "native/with_replay/q4_0", 1., compact=True)
    try:
        _, restored = restore_checkpoint(backend, output / "after-second-wake")
        proof = backend.verify_loaded_snapshot(output / "after-second-wake",
                                              digest(output / "after-second-wake/state.bin"))
        tokens, logits = continuation(backend)
        if tokens != json.loads((output / "continuation.json").read_text()):
            raise AssertionError("fresh-process continuation differs")
        np.testing.assert_array_equal(logits, np.load(output / "continuation.npy", allow_pickle=False))
        write_json(output / "restart.json", {"restore": restored, "verification": proof,
                   "fresh_process": True, "eight_tokens_and_logits_equal": True})
    finally:
        backend.close()


def run(parent, output, converter, native_python):
    parent, output, converter, native_python = [Path(p).resolve() for p in (parent, output, converter, native_python)]
    identity = validate_parent(parent)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {"completed": False, "synthetic_only": True, "runtime_constructed": False,
              "actions_executed": False, "gpu_used": False, "quantized_training": False,
              "parent_identity": identity, "converter_revision_expected": CONVERTER_REVISION}
    try:
        data = datasets(parent)
        write_json(output / "examples.json", data)
        report["training"] = train_candidates(parent, output, data)
        report["conversion"] = {}
        for name in ("new_only", "with_replay"):
            folder = output / name
            run_child([sys.executable, converter / "convert_lora_to_gguf.py", "--base", parent / "base",
                       "--outtype", "f32", "--outfile", folder / "adapter.gguf", folder / "adapter"],
                      folder / "conversion.log")
            report["conversion"][name] = verify_factors(folder, converter)
        run_child([native_python, __file__, "--native", output, "--parent", parent], output / "native.log")
        report["native"] = json.loads((output / "native.json").read_text())
        sys.path.insert(0, str(converter / "gguf-py"))
        import gguf
        report["quantized_bases"] = {}
        for precision in ("f32", "q8_0", "q4_0"):
            path = output / "models" / f"{precision}.gguf"
            reader = gguf.GGUFReader(path)
            counts = Counter(t.tensor_type.name for t in reader.tensors)
            if precision != "f32" and counts[precision.upper()] == 0:
                raise AssertionError("quantization did not produce the requested tensor type")
            report["quantized_bases"][precision] = {"tensor_types": dict(counts), "bytes": path.stat().st_size}
            del reader
        if validate_parent(parent) != identity:
            raise AssertionError("original experiment changed")
        report.update(completed=True, parent_unchanged=True)
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        report["versions"] = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
        report["converter_source_hashes"] = {p.relative_to(converter).as_posix(): digest(p)
            for p in sorted(converter.rglob("*.py"))}
        report["artifact_hashes"] = {p.relative_to(output).as_posix(): digest(p)
            for p in sorted(output.rglob("*")) if p.is_file()}
        report["artifact_bytes"] = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
        write_json(output / "report.json", report)
    return report


def main():
    cpu_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--converter", type=Path)
    parser.add_argument("--native-python", type=Path)
    parser.add_argument("--native", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--restart", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.restart:
        restart(args.restart.resolve())
    elif args.native and args.parent:
        validate_parent(args.parent.resolve())
        native_matrix(args.native.resolve(), args.parent.resolve())
    elif args.parent and args.output and args.converter and args.native_python:
        report = run(args.parent, args.output, args.converter, args.native_python)
        print(json.dumps({k: report[k] for k in ("completed", "elapsed_seconds", "artifact_bytes")}, indent=2))
    else:
        parser.error("--parent, --output, --converter and --native-python are required")


if __name__ == "__main__":
    main()
