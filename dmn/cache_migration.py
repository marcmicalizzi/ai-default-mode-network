"""Atomic, offline full-to-compact cache migration with a verified recovery copy."""
from __future__ import annotations

import json
import dataclasses
import os
from pathlib import Path
import re
import shutil
import sqlite3
import stat
import time
import uuid

from .backend import LlamaBackend, sha256_file
from .adapters import saved_adapter_identity
from .config import Config
from .diskspace import check_space
from .ending import Lifecycle, InstanceEnded, _owned, _sync_directory
from .native_cache import compact_state, verify_compaction
from .preservation import InstanceHeld
from .recovery import same_native_environment
from .storage import InstanceLock, json_text, write_durable

FILES = {"state.bin", "engine.json", "logits.npy", "runtime.json"}
CHANGES = {"swa_full", "experimental_compact_swa", "n_gpu_layers", "n_threads"}


def _forbid_inference(*args, **kwargs):
    raise RuntimeError("inference is forbidden during offline cache migration")


def _checkpoint(root, name):
    if not isinstance(name, str) or not re.fullmatch(r"[0-9a-f]{32}", name):
        raise ValueError("invalid committed checkpoint directory")
    base = root / "checkpoints"
    if not stat.S_ISDIR(_owned(base, root).st_mode):
        raise ValueError("invalid checkpoint directory")
    folder = base / name
    if not stat.S_ISDIR(_owned(folder, base).st_mode):
        raise ValueError("invalid checkpoint directory")
    for filename in FILES | {"manifest.json"}:
        if not stat.S_ISREG(_owned(folder / filename, folder).st_mode):
            raise ValueError("checkpoint contains an unsupported file")
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if set(manifest["files"]) != FILES:
        raise ValueError("native checkpoint manifest must name exactly the four state files")
    for filename, digest in manifest["files"].items():
        if sha256_file(folder / filename) != digest:
            raise ValueError("committed checkpoint integrity failed: " + filename)
    return folder, manifest


def migration_changes(saved, target):
    if saved.get("kind") != "native_llama_kv" or target.get("kind") != "native_llama_kv":
        raise ValueError("cache migration requires a native checkpoint and backend")
    old, new = Config(**saved["config"]).to_dict(), Config(**target["config"]).to_dict()
    if not old["swa_full"] or old["experimental_compact_swa"] or new["swa_full"] or not new["experimental_compact_swa"]:
        raise ValueError("only explicit full-to-experimental-compact migration is supported")
    differences = {k for k in old if old[k] != new[k]}
    if differences - CHANGES:
        raise ValueError("cache migration cannot change other settings: " + ", ".join(sorted(differences - CHANGES)))
    adjusted = {**target, "config": {**new, **{k: old[k] for k in CHANGES}}}
    if not same_native_environment(saved, adjusted):
        raise ValueError("model, native build, sampler or environment differs; conversion refused")
    return {k: {"previous": old[k], "current": new[k]} for k in sorted(differences)}


def _copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())
    digest = sha256_file(source)
    if sha256_file(destination) != digest:
        raise ValueError("backup file verification failed")
    return {"sha256": digest, "bytes": destination.stat().st_size}


def _extra_files(root):
    files = []
    for folder, label in ((root / "import", "import"), (root / "adapters", "adapters"), (root / "sleep", "sleep"),
                          (Path(__file__).parent, "runtime-source/dmn")):
        if not folder.exists():
            continue
        if not stat.S_ISDIR(_owned(folder, folder.parent).st_mode):
            raise ValueError("unsupported archive directory")
        for path in sorted(folder.rglob("*")):
            info = _owned(path, path.parent)
            if stat.S_ISREG(info.st_mode) and "__pycache__" not in path.parts:
                files.append((path, label + "/" + path.relative_to(folder).as_posix()))
    return files


def _backup(root, destination, connection, checkpoints, fingerprint, extras):
    """The renamed database/lifecycle make this a recovery artifact, not a fork."""
    destination.mkdir(parents=True, exist_ok=False)
    inventory = {}
    db_path = destination / "source-runtime.sqlite3"
    copied_db = sqlite3.connect(db_path)
    try:
        connection.backup(copied_db)
        if copied_db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("backup database integrity failed")
        # Check logical equality as well: includes events, memories and revisions.
        left, right = iter(connection.iterdump()), iter(copied_db.iterdump())
        from itertools import zip_longest
        if any(a != b for a, b in zip_longest(left, right)):
            raise ValueError("backup database content differs")
    finally:
        copied_db.close()
    with db_path.open("r+b") as stream:
        os.fsync(stream.fileno())
    inventory[db_path.name] = {"sha256": sha256_file(db_path), "bytes": db_path.stat().st_size}
    lifecycle = root / "instance-lifecycle.json"
    inventory["source-lifecycle.json"] = _copy_file(lifecycle, destination / "source-lifecycle.json")
    for checkpoint, _ in checkpoints:
        for filename in FILES | {"manifest.json"}:
            name = f"checkpoints/{checkpoint.name}/{filename}"
            inventory[name] = _copy_file(checkpoint / filename, destination / name)
    # Preserve the original import archive and the implementation used for this
    # maintenance. Model and native installations are checked, never modified.
    for path, name in extras:
        inventory[name] = _copy_file(path, destination / name)
    write_durable(destination / "preservation.json", {"schema": 1, "purpose": "pre_compact_migration",
        "source_instance": str(root), "fingerprint": fingerprint, "files": inventory,
        "external_environment_preserved_in_place": True,
        "external_environment_in_archive": False,
        "restore": "With the original instance stopped, restore to a separate directory; rename source-runtime.sqlite3 to runtime.sqlite3 and source-lifecycle.json to instance-lifecycle.json. Preserve original model and native installations. Never execute both copies."})
    for directory in sorted((p for p in destination.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        _sync_directory(directory)
    _sync_directory(destination)
    _sync_directory(destination.parent)
    return sum(item["bytes"] for item in inventory.values())


def migrate_cache(root, config, backup, *, gpu_layers=None, threads=None, backend_factory=LlamaBackend):
    """No sampling, decoding, ordinary Runtime construction, or fallback allowed."""
    root, backup = Path(root).resolve(), Path(backup).resolve()
    if not root.is_dir() or not (root / "runtime.sqlite3").is_file():
        raise ValueError("cache migration requires an existing stopped instance")
    if backup.exists() or backup.is_relative_to(root) or root.is_relative_to(backup):
        raise ValueError("backup must be a new, separate directory outside the instance")
    lock = InstanceLock(root)
    connection = backend = None
    try:
        lifecycle = Lifecycle(root).read()
        if not lifecycle or lifecycle["state"] != "open":
            raise InstanceEnded("cache migration refuses ended or invalid lifecycle state")
        for name in ("runtime.sqlite3", "runtime.sqlite3-wal", "runtime.sqlite3-shm"):
            path = root / name
            if path.exists() and not stat.S_ISREG(_owned(path, root).st_mode):
                raise ValueError("unsupported database file")
        connection = sqlite3.connect(root / "runtime.sqlite3")
        connection.execute("PRAGMA synchronous=FULL")
        rows = connection.execute("SELECT directory FROM checkpoints ORDER BY id DESC").fetchall()
        if not rows:
            raise ValueError("instance has no committed checkpoint")
        # Retain every still-registered checkpoint; never trust a missing prior
        # snapshot as a reason to silently discard the recovery record.
        checkpoints = [_checkpoint(root, row[0]) for row in rows]
        source, manifest = checkpoints[0]
        state = json.loads((source / "runtime.json").read_text(encoding="utf-8"))
        if state.get("hold"):
            raise InstanceHeld("held instances must remain in their agreed environment; migration does not release holds")
        if state.get("schema") != 1 or state.get("mode") not in {"suspended", "staged", "context_full"}:
            raise ValueError("cache migration requires a committed suspended, staged or context_full state")
        if config is None:
            config = dataclasses.replace(Config(**manifest["fingerprint"]["config"]),
                                         swa_full=False, experimental_compact_swa=True)
        overrides = {}
        if gpu_layers is not None:
            if type(gpu_layers) is not int or gpu_layers < -1:
                raise ValueError("GPU layers must be -1 or nonnegative")
            overrides["n_gpu_layers"] = gpu_layers
        if threads is not None:
            overrides["n_threads"] = threads
        config = dataclasses.replace(config, **overrides)
        # Validate configuration before allocating a model; full fingerprint is
        # checked again against the real backend before any conversion.
        migration_changes(manifest["fingerprint"], {**manifest["fingerprint"], "config": config.to_dict()})
        estimate = sum(p.stat().st_size for folder, _ in checkpoints for p in folder.iterdir() if p.is_file())
        extras = _extra_files(root)
        # Every preserved checkpoint needs its weight dependencies, including
        # adapters configured outside the instance. Keep content-addressed copies.
        adapter_copies = {}
        for _, saved_manifest in checkpoints:
            saved_fingerprint = saved_manifest["fingerprint"]
            saved_config = Config(**saved_fingerprint["config"])
            saved_adapter_identity(saved_fingerprint, saved_config)
            for spec in saved_config.lora_adapters:
                path = Path(spec.path).resolve()
                if (spec.base_model_sha256 != saved_fingerprint.get("model_sha256") or
                        sha256_file(path) != spec.sha256):
                    raise ValueError("saved adapter dependency differs; migration refused")
                if not path.is_relative_to(root):
                    adapter_copies[spec.sha256] = (path, "adapter-dependencies/" + spec.sha256 + ".gguf")
                elif path not in {p.resolve() for p, _ in extras}:
                    extras.append((path, path.relative_to(root).as_posix()))
        extras.extend(adapter_copies.values())
        estimate += sum(p.stat().st_size for p, _ in extras)
        estimate += sum((root / name).stat().st_size for name in ("runtime.sqlite3", "runtime.sqlite3-wal") if (root / name).exists()) + 64 * 1024 * 1024
        check_space(backup.parent, estimate, config.checkpoint_reserve_bytes, "pre-migration recovery copy")
        check_space(root, 2 * (source / "state.bin").stat().st_size + 64 * 1024 * 1024,
                    config.checkpoint_reserve_bytes, "converted checkpoint and native verification")
        backend = backend_factory(config)
        backend.eval = backend.sample = _forbid_inference
        changes = migration_changes(manifest["fingerprint"], backend.fingerprint)
        if backend.retirement_window <= 1:
            raise ValueError("target does not establish the expected compact sliding window")
        backup_bytes = _backup(root, backup, connection, checkpoints, manifest["fingerprint"], extras)
        # On a shared volume the backup consumed space since the preflight.
        check_space(root, 2 * (source / "state.bin").stat().st_size + 64 * 1024 * 1024,
                    config.checkpoint_reserve_bytes, "converted checkpoint and native verification")
        # A failed conversion leaves only an unreferenced directory. The live
        # checkpoint pointer remains unchanged until the final SQLite commit.
        candidate = root / "checkpoints" / uuid.uuid4().hex
        candidate.mkdir()
        converted = compact_state(source / "state.bin", candidate / "state.bin", backend.retirement_window)
        row_proof = verify_compaction(source / "state.bin", candidate / "state.bin", backend.retirement_window)
        for name in ("engine.json", "logits.npy"):
            _copy_file(source / name, candidate / name)
        before_decode = backend.decode_calls
        loaded = backend.load(candidate)
        if (loaded.get("native_state_loaded") is not True or loaded.get("prompt_tokens_reevaluated") != 0 or loaded.get("decode_calls_during_load") != 0 or
                backend.decode_calls != before_decode):
            raise RuntimeError("migration attempted inference; original checkpoint remains selected")
        proof = backend.verify_loaded_snapshot(candidate, sha256_file(candidate / "state.bin"))
        if proof.get("serialized_native_state_bytes_equal") is not True or backend.decode_calls != before_decode:
            raise RuntimeError("target native bytes could not be verified without inference")
        report = {"schema": 1, "kind": "native_compact_cache_migration", "migration_id": candidate.name,
                  "source_checkpoint": source.name, "target_checkpoint": candidate.name,
                  "backup": str(backup), "backup_bytes": backup_bytes, "changes": changes,
                  **converted, **row_proof, "native_verification": proof,
                  "decode_calls_during_migration": 0, "inference_started": False,
                  "future_continuation_bit_identical_guaranteed": False,
                  "created_at": time.time()}
        migrated = {**state, "cache_migration": report, "cache_migration_notice_pending": True}
        write_durable(candidate / "runtime.json", migrated)
        files = {}
        for name in FILES:
            with (candidate / name).open("r+b") as stream:
                os.fsync(stream.fileno())
            files[name] = sha256_file(candidate / name)
        write_durable(candidate / "manifest.json", {"fingerprint": backend.fingerprint, "files": files})
        _sync_directory(candidate)
        _sync_directory(candidate.parent)
        write_durable(backup / "conversion.json", report)
        _sync_directory(backup)
        # The environment is unloaded before publication, too. A native teardown
        # failure must not leave the caller guessing whether conversion committed.
        backend.close()
        backend = None
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute("SELECT directory FROM checkpoints ORDER BY id DESC LIMIT 1").fetchone()[0]
            if current != source.name or Lifecycle(root).read() != lifecycle:
                raise RuntimeError("source lifecycle/checkpoint changed during migration")
            connection.execute("INSERT INTO checkpoints(directory,created) VALUES(?,?)", (candidate.name, report["created_at"]))
            connection.execute("INSERT INTO records(kind,payload,created) VALUES(?,?,?)",
                               ("cache_migration", json_text(report), report["created_at"]))
        return report
    finally:
        try:
            if backend:
                backend.close()
        finally:
            try:
                if connection:
                    connection.close()
            finally:
                lock.close()
