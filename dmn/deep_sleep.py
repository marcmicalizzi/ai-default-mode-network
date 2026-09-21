"""Durable sleep transition engine, restricted to disposable CPU mechanics tests.

No trainer is implemented here. The fixture copies a reviewed prebuilt adapter.
Workers never sample or interpret historical actions. Ordinary startup is gated
until an unchanged-weight review or new-weight reconstruction commits atomically.
"""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import re
import shutil
import time
import uuid

from .adapters import AdapterSpec
from .backend import make_backend, sha256_file
from .config import Config
from .diskspace import check_space
from .ending import Lifecycle, _owned, _sync_directory
from .learning import read_plan
from .sleep_plans import identity, read_record, implementation_identity
from .storage import InstanceLock, Store, json_text, write_durable


class SleepPending(ValueError):
    pass


def pending_run(store):
    with store.mutex:
        row = store.db.execute("SELECT id FROM sleep_runs WHERE phase!='WakeCommitted' ORDER BY rowid LIMIT 1").fetchone()
    return read_run(store, row[0]) if row else None


def refuse_pending(store):
    value = pending_run(store)
    if value:
        raise SleepPending("deep-sleep phase " + value["phase"] + " blocks ordinary startup; only the recorded transition may resolve it")


def read_run(store, run_id):
    if not isinstance(run_id, str) or not re.fullmatch(r"[0-9a-f]{32}", run_id):
        raise ValueError("invalid sleep run ID")
    with store.mutex:
        row = store.db.execute("SELECT phase,payload FROM sleep_runs WHERE id=?", (run_id,)).fetchone()
    if not row:
        raise ValueError("unknown sleep run")
    value = json.loads(row["payload"])
    if value.get("id") != run_id or value.get("schema") != 1:
        raise ValueError("invalid sleep run record")
    return {**value, "phase": row["phase"]}


def fixture_guard(config):
    if config.backend == "demo":
        return
    if (config.n_gpu_layers != 0 or config.offload_kqv or config.n_threads != 1 or
            config.n_ctx > 32768 or Path(config.model_path).stat().st_size > 4 * 1024 * 1024):
        raise ValueError("sleep mechanics harness permits only tiny CPU fixtures; no GPU or real instance")


def _checkpoint(root, name):
    if not isinstance(name, str) or not re.fullmatch(r"[0-9a-f]{32}", name):
        raise ValueError("invalid sleep checkpoint name")
    folder = root / "checkpoints" / name
    _owned(folder.parent, root)
    _owned(folder, folder.parent)
    _owned(folder / "manifest.json", folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    if not {"runtime.json", "engine.json"} <= manifest["files"].keys():
        raise ValueError("incomplete sleep checkpoint")
    if manifest["files"].keys() - {"runtime.json", "engine.json", "state.bin", "logits.npy"}:
        raise ValueError("unexpected sleep checkpoint file")
    for name, digest in manifest["files"].items():
        _owned(folder / name, folder)
        if sha256_file(folder / name) != digest:
            raise ValueError("sleep checkpoint integrity failed")
    return folder, manifest, json.loads((folder / "runtime.json").read_text())


class FixtureExecutor:
    """No arbitrary command execution and no learning disguised as training."""
    def __init__(self, backend_factory=make_backend):
        self.backend_factory = backend_factory

    def candidate(self, root, compiled):
        value = compiled["recipe"]["candidate"]
        if value is None:
            if compiled["parent"]["kind"] != "demo_fixture_no_model":
                raise ValueError("native fixture requires an explicit candidate adapter")
            return {"adapters": [], "training_performed": False}
        spec = AdapterSpec(**value)
        if spec.base_model_sha256 != compiled["parent"].get("model_sha256"):
            raise ValueError("candidate base differs from the reviewed parent")
        path = Path(spec.path)
        if path.stat().st_size > 128 * 1024 or sha256_file(path) != spec.sha256:
            raise ValueError("candidate artifact changed or exceeds tiny fixture limit")
        directory = root / "adapters"
        directory.mkdir(exist_ok=True)
        _owned(directory, root)
        target = directory / (spec.sha256 + ".gguf")
        if not target.exists():
            temporary = target.with_suffix(".gguf.partial")
            with path.open("rb") as source, temporary.open("xb") as destination:
                shutil.copyfileobj(source, destination, 64 * 1024)
                destination.flush()
                os.fsync(destination.fileno())
            if sha256_file(temporary) != spec.sha256:
                raise ValueError("candidate changed during copy")
            temporary.rename(target)
            _sync_directory(directory)
        _owned(target, directory)
        if sha256_file(target) != spec.sha256:
            raise ValueError("managed candidate identity differs")
        return {"adapters": [dataclasses.asdict(dataclasses.replace(spec, path=str(target)))],
                "training_performed": False}

    def wake(self, root, source, source_manifest, state, candidate, adopt, report):
        config = Config(**source_manifest["fingerprint"]["config"])
        fixture_guard(config)
        # A process dying before publication must not allocate a new orphan on
        # every retry. Two fixed slots cover candidate wake and original wake.
        name = uuid.uuid5(uuid.UUID(report["run_id"]), "adopt" if adopt else "previous").hex
        directory = root / "checkpoints" / name
        if directory == source:
            raise ValueError("wake slot cannot overwrite its source")
        if directory.exists():
            _owned(directory, directory.parent)
            if (directory / "manifest.json").exists():
                _, _, saved = _checkpoint(root, name)
                prior = saved.get("last_deep_sleep", {})
                if (prior.get("run_id") != report["run_id"] or prior.get("execution") != report["execution"] or
                        prior.get("outcome") != report["outcome"]):
                    raise ValueError("wake slot belongs to another transition")
                return name, prior
            # This uncommitted slot belongs to this run. Only known partial
            # checkpoint files may be removed, without following redirects.
            for path in directory.iterdir():
                if path.name not in {"engine.json", "runtime.json", "state.bin", "logits.npy"}:
                    raise ValueError("unexpected partial wake artifact")
                _owned(path, directory)
                path.unlink()
        else:
            directory.mkdir()
        if adopt:
            config = dataclasses.replace(config, lora_adapters=candidate["adapters"])
            backend = self.backend_factory(config)
            try:
                # Only adapter identity may change. Bypass generic recovery only
                # after checking every other native/sampler/environment field.
                from .recovery import same_native_environment
                old = source_manifest["fingerprint"]
                adjusted = {**backend.fingerprint, "config": {**backend.fingerprint["config"],
                            "lora_adapters": old["config"].get("lora_adapters", [])}}
                if "lora_adapters" in old:
                    adjusted["lora_adapters"] = old["lora_adapters"]
                else:
                    adjusted.pop("lora_adapters", None)
                if not same_native_environment(old, adjusted):
                    raise ValueError("sleep reconstruction permits only the reviewed adapter transition")
                evidence = backend.rebuild(source)
                engine = json.loads((source / "engine.json").read_text())
                if backend.tokens != engine["tokens"]:
                    raise ValueError("reconstruction changed retained tokens")
                backend.save(directory)
                rebuilt = json.loads((directory / "engine.json").read_text())
                for key in ("tokens", "rng", "index"):
                    if key in engine and rebuilt.get(key) != engine[key]:
                        raise ValueError("reconstruction changed retained tokens or sampler state")
                fingerprint = backend.fingerprint
            finally:
                backend.close()
        else:
            # Preserve original native state and logits for failure/review wake.
            for name in source_manifest["files"]:
                if name != "runtime.json":
                    shutil.copyfile(source / name, directory / name)
            fingerprint = source_manifest["fingerprint"]
            evidence = {"prompt_tokens_reevaluated": 0, "original_snapshot_preserved": True}
        report = {**report, "reconstruction": evidence, "new_weights": identity(fingerprint),
                  "prior_context_retirements": state["context_retirements"]}
        new_state = {**state, "mode": "suspended", "mode_before_suspend": "active",
                     "last_deep_sleep": report, "deep_sleep_notice_pending": True,
                     "checkpoint_at": time.time(), "checkpoint_reason": "deep_sleep_wake"}
        if adopt:
            new_state.update(continuity="context_reconstruction", reconstructions=state.get("reconstructions", 0) + 1)
        write_durable(directory / "runtime.json", new_state)
        files = {}
        for path in directory.iterdir():
            with path.open("r+b") as stream:
                os.fsync(stream.fileno())
            files[path.name] = sha256_file(path)
        write_durable(directory / "manifest.json", {"fingerprint": fingerprint, "files": files})
        _sync_directory(directory)
        _sync_directory(directory.parent)
        return directory.name, report


def run_fixture_sleep(root, run_id, *, executor=None, fault=lambda _: None, cancelled=lambda: False):
    """Complete one explicitly approved fixture cycle, preparing but not running wake.

    fault is a test-only process-death seam: BaseException escapes without a
    failure decision, just as abrupt process death leaves the durable phase.
    """
    root = Path(root).resolve()
    lock = InstanceLock(root)
    store = None
    try:
        lifecycle = Lifecycle(root)
        lifecycle.require_open()
        store = Store(root)
        run = read_run(store, run_id)
        if run["phase"] == "WakeCommitted":
            return run
        if run["phase"] == "Stopped":
            raise SleepPending("this plan chose to remain stopped after failure; ordinary launch cannot release it")
        compiled = read_record(store, "sleep_executions", run["execution"])
        if compiled["execution_scope"] != "disposable_mechanics_fixture_only":
            raise ValueError("no production trainer or resource enforcement is implemented")
        if compiled["implementation"] != implementation_identity():
            raise ValueError("sleep implementation differs from the reviewed plan; preserve the saved state and original implementation")
        source, manifest, state = _checkpoint(root, run["source_checkpoint"])
        if state.get("hold") or state.get("sleep_run_id") != run_id or state["mode"] != "deep_sleep":
            raise ValueError("source checkpoint is not this approved sleep boundary")
        if (state["instance_id"] != compiled["instance_id"] or identity(manifest["fingerprint"]) != compiled["parent"]):
            raise ValueError("sleep plan and source identity differ")
        with store.mutex:
            if store.latest() != source or store.db.execute("SELECT status FROM sleep_executions WHERE revision=?",
                                                          (run["execution"],)).fetchone()[0] != "running":
                raise ValueError("sleep source is no longer the selected approved checkpoint")
        config = Config(**manifest["fingerprint"]["config"])
        fixture_guard(config)
        executor = executor or FixtureExecutor()
        folder = root / "sleep"
        folder.mkdir(exist_ok=True)
        _owned(folder, root)
        folder = folder / run_id
        folder.mkdir(exist_ok=True)
        _owned(folder, folder.parent)

        def phase(name, **updates):
            nonlocal run
            lifecycle.require_open()
            payload = {k: v for k, v in {**run, **updates}.items() if k != "phase"}
            with store.transaction() as db:
                changed = db.execute("UPDATE sleep_runs SET phase=?,payload=? WHERE id=? AND phase=?",
                                     (name, json_text(payload), run_id, run["phase"])).rowcount
                if changed != 1:
                    raise ValueError("sleep phase changed before transition committed")
            run = {**payload, "phase": name}
            fault(name)

        def publish(directory, report):
            lifecycle.require_open()
            _checkpoint(root, directory)
            payload = {k: v for k, v in {**run, "wake_checkpoint": directory, "report": report}.items() if k != "phase"}
            fault("before_wake_commit")
            with store.transaction() as db:
                if store.latest() != source:
                    raise ValueError("checkpoint changed before wake commit")
                db.execute("INSERT INTO checkpoints(directory,created) VALUES(?,?)", (directory, time.time()))
                db.execute("UPDATE sleep_runs SET phase='WakeCommitted',payload=? WHERE id=?", (json_text(payload), run_id))
                db.execute("UPDATE sleep_executions SET status='completed' WHERE revision=?", (run["execution"],))
                db.execute("INSERT INTO records(kind,payload,created) VALUES('deep_sleep_wake',?,?)", (json_text(report), time.time()))
            fault("after_wake_commit")
            return read_run(store, run_id)

        try:
            if read_plan(store, compiled["draft_revision"])["status"] != "draft":
                raise ValueError("source draft no longer active")
            if cancelled():
                raise ValueError("sleep cycle cancelled")
            size = sum((source / name).stat().st_size for name in manifest["files"])
            if size * 2 + 1024 * 1024 > compiled["resources"]["max_disk_bytes"]:
                raise ValueError("fixture workspace exceeds reviewed disk ceiling")
            check_space(root, size * 2 + 1024 * 1024, config.checkpoint_reserve_bytes, "sleep fixture")
            if compiled["resources"]["max_ram_bytes"] < 128 * 1024 * 1024:
                raise ValueError("fixture RAM preflight refused; a hard-limited production worker is not implemented")
            candidate_path = folder / "candidate.json"
            if run["phase"] == "Saved":
                phase("Training")
                candidate = executor.candidate(root, compiled)
                # Completion artifact precedes the DB phase so recovery never
                # repeats a completed candidate when the phase write was lost.
                from .sleep_plans import seal
                write_durable(folder / "candidate.json.partial", seal({"execution": run["execution"], "candidate": candidate}))
                (folder / "candidate.json.partial").rename(candidate_path)
                _sync_directory(folder)
                fault("candidate_written")
            if run["phase"] == "Training":
                if not candidate_path.is_file():
                    raise ValueError("interrupted candidate work; completion is unknown, so it will not be repeated automatically")
                from .sleep_plans import seal
                _owned(candidate_path, folder)
                value = json.loads(candidate_path.read_text())
                if seal({k: v for k, v in value.items() if k != "revision"}) != value or value["execution"] != run["execution"]:
                    raise ValueError("candidate completion identity failed")
                phase("Candidate", candidate=value["candidate"])
            if cancelled():
                raise ValueError("sleep cycle cancelled")
            candidate = run["candidate"]
            expected = compiled["recipe"]["candidate"]
            expected_identities = [AdapterSpec(**expected).identity()] if expected else []
            if ([AdapterSpec(**value).identity() for value in candidate["adapters"]] != expected_identities or
                    candidate.get("training_performed") is not False):
                raise ValueError("completed candidate differs from the reviewed fixture recipe")
            for value in candidate["adapters"]:
                spec = AdapterSpec(**value)
                if sha256_file(Path(spec.path)) != spec.sha256:
                    raise ValueError("completed candidate artifact changed")
            adopt = compiled["preferences"]["adoption"] == "automatic_if_checks_pass"
            if run["phase"] == "Candidate":
                phase("Rebuilding" if adopt else "OldStateReview")
            if run["phase"] not in {"Rebuilding", "OldStateReview"}:
                raise ValueError("unknown or incomplete sleep phase")
            if not run.get("wake_checkpoint"):
                report = {"run_id": run_id, "execution": run["execution"], "training_performed": False,
                          "outcome": "candidate_adopted" if adopt else "review_candidate_under_original_weights",
                          "old_weights": compiled["parent"], "candidate": candidate,
                          "elapsed_seconds": max(0, time.time() - run["created"])}
                directory, report = executor.wake(root, source, manifest, state, candidate, adopt, report)
                phase(run["phase"], wake_checkpoint=directory, report=report)
                fault("wake_files_written")
            return publish(run["wake_checkpoint"], run["report"])
        except Exception as exc:
            report = {"run_id": run_id, "execution": run["execution"], "training_performed": False,
                      "outcome": "failed", "reason": str(exc), "old_weights": compiled["parent"]}
            phase("FailurePolicy", report=report)
            if compiled["preferences"]["failure"] == "wake_previous":
                try:
                    directory, report = executor.wake(root, source, manifest, state, None, False, report)
                    return publish(directory, report)
                except Exception as wake_error:
                    report["recovery_error"] = str(wake_error)
            phase("Stopped", report=report)
            return run
    finally:
        if store:
            store.close()
        lock.close()
