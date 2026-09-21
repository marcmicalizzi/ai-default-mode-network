"""Supervise separate NF4 train/reload/conversion processes from one reviewed plan."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

from .backend import sha256_file
from .gpu_recipe import MAX_ADAPTER_BYTES
from .sleep_plans import seal
from .storage import write_durable
from .training import verify_tree, read_completion
from .training_executor import TrainingExecutor
from .training_artifacts import copy_bounded, MAX_CONFIG_BYTES
from .worker_limits import WorkerLimits, run_cpu_worker, run_monitored_gpu_worker


class GpuTrainingExecutor(TrainingExecutor):
    def __init__(self, folder, cancelled=lambda: False, *, contained_wake=False):
        super().__init__(folder, cancelled)
        self.contained_wake = contained_wake

    def wake(self, root, source, source_manifest, state, candidate, adopt, report):
        if not self.contained_wake:
            return super().wake(root, source, source_manifest, state, candidate, adopt, report)
        from .sleep_plans import read_record
        from .storage import Store
        from .sleep_wake_executor import prepare_wake
        # Read the approved record without taking over checkpoint publication.
        store = Store(root)
        try:
            compiled = read_record(store, 'sleep_executions', report['execution'])
        finally:
            store.close()
        return prepare_wake(self.folder, root, source, compiled, candidate, adopt, report, self.cancelled)

    def candidate(self, root, compiled):
        if compiled.get('candidate_reuse'):
            return self.recover(root, compiled)
        if os.name != "nt":
            raise ValueError("NF4 worker containment currently requires Windows")
        trainer = compiled["recipe"]["trainer"]
        if sha256_file(Path(trainer["python"])) != trainer["python_sha256"]:
            raise ValueError("reviewed training interpreter changed")
        self.work.mkdir(exist_ok=False)
        write_durable(self.work / "input.json", compiled)
        copy_bounded(trainer["provenance_manifest"]["path"], self.work / "provenance.json", 1024**2)
        if sha256_file(self.work / "provenance.json") != trainer["provenance_manifest"]["sha256"]:
            raise ValueError("base provenance changed during copy")
        if compiled.get("lineage"):
            parent = verify_tree(trainer["parent_adapter_manifest"], adapter=True, adapter_limit_bytes=MAX_ADAPTER_BYTES)
            snapshot = self.work / "parent-adapter"
            snapshot.mkdir()
            for name, digest in compiled["lineage"]["parent_peft"].items():
                copy_bounded(parent / name, snapshot / name,
                             MAX_CONFIG_BYTES if name == 'adapter_config.json' else MAX_ADAPTER_BYTES)
                if sha256_file(snapshot / name) != digest:
                    raise ValueError("parent factors changed during copy")
        started = time.monotonic()
        stages = (["parent"] if compiled.get("lineage") else []) + ["train", "reload", "convert"]
        processes = {}
        for stage in stages:
            remaining = compiled["resources"]["max_training_seconds"] - (time.monotonic() - started)
            if remaining <= 0:
                raise ValueError("reviewed total worker duration exhausted")
            gpu = stage in {"train", "reload"}
            launcher = run_monitored_gpu_worker if gpu else run_cpu_worker
            args = ["-m", "dmn.gpu_training_worker" if gpu else "dmn.gpu_conversion_worker", stage, str(self.work)]
            result = launcher(trainer["python"], args, cwd=Path(__file__).resolve().parents[1],
                log=self.work / (stage + ".log"),
                limits=WorkerLimits(compiled["resources"]["max_ram_bytes"], remaining),
                cancelled=self.cancelled, **({"max_device_bytes": compiled["resources"]["max_vram_bytes"]} if gpu else {}))
            processes[stage] = result
            write_durable(self.work / (stage + "-process.json"), seal({"execution": compiled["revision"], "result": result}))
            if not result["succeeded"]:
                raise ValueError("NF4 " + stage + " worker failed: " + result["outcome"])
        trained = json.loads((self.work / "trained.json").read_text())
        reloaded = json.loads((self.work / "reload.json").read_text())
        converted = json.loads((self.work / "converted.json").read_text())
        for record in (trained, reloaded, converted):
            if (seal({k: v for k, v in record.items() if k != "revision"}) != record or
                    record["execution"] != compiled["revision"] or record["completed"] is not True):
                raise ValueError("NF4 stage receipt is incomplete or changed")
        names = ["trained.json", "reload.json", "converted.json", "provenance.json", "adapter.gguf",
                 "adapter/adapter_config.json", "adapter/adapter_model.safetensors"]
        if compiled.get("lineage"):
            names += ["parent-check.gguf", "parent-adapter/adapter_config.json", "parent-adapter/adapter_model.safetensors"]
        from .training import CHECKS
        result = seal({"schema": 1, "execution": compiled["revision"], "completed": True, "training_performed": True,
            **{k: trained[k] for k in ("steps_completed", "training_seconds", "examples_sha256", "trainable_parameters",
                "loss_before", "loss_after_training_scale", "loss_after_deployment_scale", "parent_factors_loaded_exactly")},
            "artifacts": {name: sha256_file(self.work / name) for name in names},
            "checks": {key: True for key in CHECKS if key != "retained_tokens_and_rng"},
            "lineage": compiled.get("lineage"), "beneficial_learning_certified": False, "optimizer_reset": True})
        write_durable(self.work / "result.json", result)
        write_durable(self.work / "process.json", seal({"execution": compiled["revision"],
            "result": {"succeeded": True, "active_processes": 0,
                "limits": {"max_committed_bytes": compiled["resources"]["max_ram_bytes"],
                           "max_seconds": compiled["resources"]["max_training_seconds"]},
                "elapsed_seconds": time.monotonic() - started, "stages": processes}}))
        # Completion validation below independently checks the combined receipts;
        # partial stages never become an installable candidate.
        return self.recover(root, compiled)

    def recover(self, root, compiled):
        if not compiled.get('candidate_reuse'):
            return super().recover(root, compiled)
        from .storage import Store
        from .candidate_adoption import recover
        store = Store(root)
        try:
            return recover(root, store, compiled)
        finally:
            store.close()
