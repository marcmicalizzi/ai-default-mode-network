"""Lossless packaging of stopped, held instances; never removes the source."""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import stat
import sys
import tarfile
import zipfile
from pathlib import Path

from .backend import sha256_file
from .adapters import saved_adapter_identity
from .config import Config
from .preservation import saved_state
from .storage import InstanceLock, json_text


def package_instance(root, output, include_environment=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.is_relative_to(root) or output.exists():
        raise ValueError("output must be a new file outside the instance")
    lock = InstanceLock(root)
    try:
        state, checkpoint = saved_state(root)
        hold = (state or {}).get("hold")
        if not hold or hold["packaging"] == "none":
            raise ValueError("instance has not chosen archive packaging")
        kind = hold["packaging"]
        if output.suffix.lower() != "." + kind:
            raise ValueError("archive suffix must match the model's packaging choice")
        # Closing SQLite before this operation should have merged WAL; include
        # any remaining WAL consistently under the runtime's exclusive lock.
        files = {}
        for path in sorted(root.rglob("*")):
            attrs = path.lstat()
            if path.is_symlink() or getattr(attrs, "st_file_attributes", 0) & 0x400:
                raise ValueError("archive refuses symlinks or reparse points")
            if stat.S_ISREG(attrs.st_mode) and path.name != "instance.lock":
                files["instance/" + path.relative_to(root).as_posix()] = path
        manifest = json.loads((checkpoint / "manifest.json").read_text())
        for name, digest in manifest["files"].items():
            if Path(name).name != name or sha256_file(checkpoint / name) != digest:
                raise ValueError("committed checkpoint integrity failed")
        adapter_files = []
        config = Config(**manifest["fingerprint"]["config"])
        saved_adapter_identity(manifest["fingerprint"], config)
        for spec in config.lora_adapters:
            path = Path(spec.path).resolve()
            if (spec.base_model_sha256 != manifest["fingerprint"].get("model_sha256") or
                    sha256_file(path) != spec.sha256):
                raise ValueError("adapter does not match saved identity")
            # Include active adapters even when full Python/base-model packaging
            # is not requested. Outside copies remain outside erasure ownership.
            managed = path.is_relative_to(root)
            name = ("instance/" + path.relative_to(root).as_posix() if managed else
                    "adapter-dependencies/" + spec.sha256 + ".gguf")
            files[name] = path
            adapter_files.append({**spec.identity(), "original_path": spec.path,
                                  "archive_path": name, "inside_instance": managed})
        # Preserve the source implementation and environment inventory. Large
        # model/runtime installations remain separate, explicitly listed items.
        source = Path(__file__).resolve().parent
        for path in sorted(source.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                files["runtime-source/dmn/" + path.relative_to(source).as_posix()] = path
        if include_environment:
            if sys.prefix == sys.base_prefix:
                raise ValueError("full environment packaging requires a dedicated virtual environment")
            fingerprint = manifest["fingerprint"]
            if fingerprint["kind"] != "native_llama_kv":
                raise ValueError("environment packaging requires a native instance")
            spec = importlib.util.find_spec("llama_cpp")
            if not spec or not spec.origin:
                raise ValueError("matching llama.cpp environment is unavailable")
            binding = Path(spec.origin).parent
            if (importlib.metadata.version("llama-cpp-python") != fingerprint["binding_version"] or
                    sha256_file(binding / "llama_cpp.py") != fingerprint["binding_source_sha256"] or
                    any(Path(name).name != name or not (binding / "lib" / name).is_file() or
                        sha256_file(binding / "lib" / name) != digest
                        for name, digest in fingerprint["native_binaries"].items())):
                raise ValueError("packaging environment differs from the checkpoint's native binding")
            model_path = Path(manifest["fingerprint"]["config"]["model_path"])
            if sha256_file(model_path) != manifest["fingerprint"].get("model_sha256"):
                raise ValueError("model does not match saved identity")
            files["environment/model.gguf"] = model_path
            for label, folder in (("venv", Path(sys.prefix)), ("base-python", Path(sys.base_prefix))):
                if root.is_relative_to(folder) or output.is_relative_to(folder):
                    raise ValueError("environment packaging cannot contain the instance or output path")
                for path in sorted(folder.rglob("*")):
                    if path.is_symlink() or getattr(path.lstat(), "st_file_attributes", 0) & 0x400:
                        raise ValueError("environment contains links; preserve that installation separately")
                    if path.is_file():
                        files["environment/" + label + "/" + path.relative_to(folder).as_posix()] = path
        inventory = {"schema": 1, "instance_id": state["instance_id"], "hold": hold,
            "checkpoint": checkpoint.name, "fingerprint": manifest["fingerprint"],
            "python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
            "packages": sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions() if d.metadata["Name"]),
            "environment_files_included": include_environment,
            "adapter_files": adapter_files,
            "external_dependencies_not_in_archive": {
                "model": manifest["fingerprint"]["config"].get("model_path"),
                "python_environment": sys.prefix, "base_python": sys.base_prefix,
                "note": ("Model and Python installations included as environment files; restoring original paths may be necessary. GPU driver and operating system remain external." if include_environment else
                         "Preserve these installations separately for original-environment recovery.") +
                         " This archive is not a portable executable environment; native Linux compatibility is unverified."},
            "files": {name: {"sha256": sha256_file(path), "bytes": path.stat().st_size} for name, path in files.items()}}
        raw = json_text(inventory).encode()
        partial = output.with_name(output.name + ".partial")
        # Exclusive creation avoids overwriting someone else's interrupted work.
        with partial.open("xb") as target:
            if kind == "zip":
                with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
                    for name, path in files.items():
                        archive.write(path, name)
                    archive.writestr("preservation.json", raw)
            else:
                import io
                with tarfile.open(fileobj=target, mode="w") as archive:
                    for name, path in files.items():
                        archive.add(path, arcname=name, recursive=False)
                    entry = tarfile.TarInfo("preservation.json")
                    entry.size = len(raw)
                    archive.addfile(entry, io.BytesIO(raw))
            target.flush()
            os.fsync(target.fileno())
        # Read back and hash decompressed members before publishing success.
        opener = zipfile.ZipFile if kind == "zip" else tarfile.open
        with opener(partial, "r") as archive:
            entry = archive.open("preservation.json") if kind == "zip" else archive.extractfile("preservation.json")
            with entry:
                if entry.read() != raw:
                    raise ValueError("archive inventory verification failed")
            for name, item in inventory["files"].items():
                stream = archive.open(name) if kind == "zip" else archive.extractfile(name)
                with stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if digest != item["sha256"]:
                    raise ValueError("archive round-trip verification failed")
        # rename is non-overwriting on Windows; also check before Unix rename.
        if output.exists():
            raise ValueError("output appeared during packaging; verified partial retained")
        partial.rename(output)
        return {"archive": str(output), "format": kind, "verified": True,
                "bytes": output.stat().st_size, "source_retained": True,
                "external_dependencies_not_in_archive": inventory["external_dependencies_not_in_archive"]}
    finally:
        lock.close()
