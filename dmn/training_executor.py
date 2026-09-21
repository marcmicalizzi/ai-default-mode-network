"""Bridge reviewed plans to a real worker; still gated to tiny CPU integration."""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import shutil

from .adapters import AdapterSpec
from .backend import sha256_file
from .deep_sleep import FixtureExecutor
from .ending import _owned, _sync_directory
from .storage import write_durable
from .sleep_plans import seal
from .training import KIND_V2, read_completion, verify_tree
from .worker_limits import WorkerLimits, run_cpu_worker


class TrainingExecutor(FixtureExecutor):
    def __init__(self, folder, cancelled=lambda: False):
        super().__init__()
        self.folder, self.cancelled = folder, cancelled

    @property
    def work(self):
        return self.folder / "worker"

    def candidate(self, root, compiled):
        trainer = compiled["recipe"]["trainer"]
        if os.name != "nt":
            raise ValueError("CPU worker containment currently requires Windows; no unbounded fallback")
        if sha256_file(Path(trainer["python"])) != trainer["python_sha256"]:
            raise ValueError("reviewed training interpreter changed")
        verify_tree(trainer["base_manifest"], base=True)
        verify_tree(trainer["converter_manifest"])
        if compiled["recipe"]["kind"] == KIND_V2:
            from .base_provenance import verify
            verify(trainer["provenance_manifest"], trainer, compiled["parent"]["model_sha256"])
        if compiled.get("lineage"):
            verify_tree(trainer["parent_adapter_manifest"], adapter=True)
        # Any directory here means work may have started. Re-entry uses recover,
        # never deletes partial files or silently performs another training run.
        self.work.mkdir(exist_ok=False)
        _owned(self.work, self.folder)
        write_durable(self.work / "input.json", compiled)
        result = run_cpu_worker(trainer["python"], ["-m", "dmn.training_worker", str(self.work)],
            cwd=Path(__file__).resolve().parents[1], log=self.work / "worker.log",
            limits=WorkerLimits(compiled["resources"]["max_ram_bytes"], compiled["resources"]["max_training_seconds"]),
            cancelled=self.cancelled)
        write_durable(self.work / "process.json", seal({"execution": compiled["revision"], "result": result}))
        if not result["succeeded"]:
            reason = result["outcome"]
            failure = self.work / "failure.json"
            if failure.exists():
                _owned(failure, self.work)
                problem = json.loads(failure.read_text())
                if (seal({k: v for k, v in problem.items() if k != "revision"}) == problem and
                        problem["execution"] == compiled["revision"]):
                    reason += ": " + problem["error_type"] + ": " + problem["reason"]
            raise ValueError("training worker failed: " + reason + "; no candidate adopted")
        return self.recover(root, compiled)

    def recover(self, root, compiled):
        """A completed receipt can be validated/copied without redoing training."""
        _owned(self.work, self.folder)
        result = read_completion(self.work, compiled)
        process = self.work / "process.json"
        _owned(process, self.work)
        evidence = json.loads(process.read_text())
        if (seal({k: v for k, v in evidence.items() if k != "revision"}) != evidence or
                evidence["execution"] != compiled["revision"] or not evidence["result"]["succeeded"] or
                evidence["result"]["limits"]["max_committed_bytes"] != compiled["resources"]["max_ram_bytes"] or
                evidence["result"]["limits"]["max_seconds"] != compiled["resources"]["max_training_seconds"]):
            raise ValueError("worker supervision is incomplete or failed; candidate remains inactive")
        digest = result["artifacts"]["adapter.gguf"]
        directory = root / "adapters"
        directory.mkdir(exist_ok=True)
        _owned(directory, root)
        target = directory / (digest + ".gguf")
        if not target.exists():
            partial = target.with_suffix(".gguf.partial")
            if partial.exists():
                _owned(partial, directory)
                partial.unlink()  # Interrupted copy only; the sealed source is intact.
            with (self.work / "adapter.gguf").open("rb") as source, partial.open("xb") as output:
                shutil.copyfileobj(source, output, 64 * 1024)
                output.flush()
                os.fsync(output.fileno())
            if sha256_file(partial) != digest:
                raise ValueError("candidate changed during managed copy")
            partial.rename(target)
            _sync_directory(directory)
        _owned(target, directory)
        if sha256_file(target) != digest:
            raise ValueError("managed candidate changed")
        spec = AdapterSpec(str(target), digest, compiled["parent"]["model_sha256"], compiled["preferences"]["scale"])
        return {"adapters": [dataclasses.asdict(spec)], "training_performed": True,
                "execution": compiled["revision"], "receipt": result["revision"],
                "lineage": result.get("lineage"),
                "training": {key: result[key] for key in ("steps_completed", "training_seconds", "trainable_parameters",
                    "loss_before", "loss_after_training_scale", "loss_after_deployment_scale", "beneficial_learning_certified")}}

    def validate(self, root, compiled, candidate):
        if self.recover(root, compiled) != candidate:
            raise ValueError("candidate differs from the validated worker completion")
