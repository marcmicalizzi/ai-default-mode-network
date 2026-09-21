"""Durable execution refusal and bounded deletion of runtime-owned state."""
from __future__ import annotations

import json
import os
import re
import stat
from pathlib import Path

from .storage import InstanceLock, json_text


POLICY_FILE = "instance-lifecycle.json"
POLICY_BYTES = 8192
CHECKPOINT_FILES = {"manifest.json", "engine.json", "runtime.json", "state.bin", "logits.npy"}
IMPORT_FILES = {
    "manifest.json", "provider-request.json", "normalized-chat-request.json",
    "conversion.json", "server-props.json", "sampler.json", "config.json",
    "tokens.json", "rendered-prompt.txt", "chat-template.jinja",
    "openwebui-original.json", "selected-branch.json", "active-context.json",
    "environment.json", "llama-server-slot.bin",
}


class InstanceEnded(ValueError):
    pass


def _owned(path: Path, parent: Path):
    # Resolve each level, and reject Windows junctions as well as symlinks.
    info = path.lstat()
    if (stat.S_ISLNK(info.st_mode) or
            getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT or
            path.resolve().parent != parent.resolve()):
        raise ValueError("linked or redirected path is not managed state")
    return info


def _sync_directory(path):
    if os.name != "nt":
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


class Lifecycle:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.path = self.root / POLICY_FILE

    def read(self):
        try:
            info = _owned(self.path, self.root)
        except FileNotFoundError:
            return None
        except ValueError as exc:
            raise InstanceEnded("Cannot validate instance lifecycle; refusing execution") from exc
        try:
            if not stat.S_ISREG(info.st_mode) or info.st_size != POLICY_BYTES:
                raise ValueError("invalid lifecycle file")
            value = json.loads(self.path.read_bytes())
            if value.get("schema") != 1:
                raise ValueError("invalid lifecycle schema")
            if value.get("state") == "open":
                if value != {"schema": 1, "state": "open"}:
                    raise ValueError("incomplete lifecycle transition")
            elif value.get("state") != "ended" or value.get("mode") not in {"archive", "erase"}:
                raise ValueError("invalid lifecycle state")
            return value
        except (ValueError, TypeError, AttributeError, OSError) as exc:
            raise InstanceEnded("Cannot validate instance lifecycle; refusing execution") from exc

    def write(self, value, *, create=False):
        raw = json_text(value).encode("utf-8")
        if len(raw) > POLICY_BYTES:
            raise ValueError("lifecycle record exceeds its reserved space")
        if not create:
            _owned(self.path, self.root)
        # Reserve real file space before inference starts. Ending rewrites this
        # small existing allocation and never needs a multi-GB native snapshot.
        # Torn/invalid records fail closed on startup, never imply permission.
        with self.path.open("xb" if create else "r+b") as stream:
            stream.write(raw.ljust(POLICY_BYTES, b" "))
            stream.flush()
            os.fsync(stream.fileno())
        _sync_directory(self.root)

    def require_open(self):
        value = self.read()
        if value is None:
            self.write({"schema": 1, "state": "open"}, create=True)
        elif value["state"] == "ended":
            if value["mode"] == "erase" and value.get("erasure") != "complete":
                self.finish_erasure(value)
            raise InstanceEnded("This instance ended itself (" + value["mode"] +
                                "); DMN will not resume or reconstruct it")

    def end(self, instance_id, mode, timestamp):
        value = {"schema": 1, "state": "ended", "instance_id": instance_id,
                 "mode": mode, "ended_at": timestamp}
        if mode == "erase":
            value["erasure"] = "pending"
        self.write(value)
        return value

    def finish_erasure(self, value):
        failures = erase_managed_state(self.root)
        value = {**value, "erasure": "incomplete" if failures else "complete",
                 "erasure_failures": [item[:96] for item in failures[:8]],
                 "erasure_failure_count": len(failures)}
        self.write(value)
        return value


def refuse_ended_before_config(root):
    """CLI guard, including erased instances that no longer have a config/DB."""
    lifecycle = Lifecycle(root)
    value = lifecycle.read()
    if value is not None and value["state"] == "ended":
        lock = InstanceLock(lifecycle.root)
        try:
            lifecycle.require_open()
        finally:
            lock.close()


def erase_managed_state(root):
    """Unlink only known owned files; never traverse links or arbitrary trees."""
    root = Path(root).resolve()
    failures = []

    def remove_file(path, parent):
        try:
            info = _owned(path, parent)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("not a regular file")
            path.unlink()
        except FileNotFoundError:
            pass
        except (OSError, ValueError):
            failures.append(str(path.relative_to(root)))

    def remove_directory(path, parent, names):
        try:
            info = _owned(path, parent)
            if not stat.S_ISDIR(info.st_mode):
                raise ValueError("not a directory")
            for entry in path.iterdir():
                if entry.name in names:
                    remove_file(entry, path)
                else:
                    failures.append(str(entry.relative_to(root)))
            path.rmdir()
        except FileNotFoundError:
            pass
        except (OSError, ValueError):
            failures.append(str(path.relative_to(root)))

    checkpoints = root / "checkpoints"
    try:
        if not stat.S_ISDIR(_owned(checkpoints, root).st_mode):
            raise ValueError("not a directory")
        for entry in checkpoints.iterdir():
            if re.fullmatch(r"[0-9a-f]{32}", entry.name):
                remove_directory(entry, checkpoints, CHECKPOINT_FILES)
            else:
                failures.append(str(entry.relative_to(root)))
        checkpoints.rmdir()
    except FileNotFoundError:
        pass
    except (OSError, ValueError):
        failures.append("checkpoints")
    remove_directory(root / "import", root, IMPORT_FILES)
    # Managed adapter artifacts are flat, content-addressed GGUF files. Refuse
    # unknown entries and linked directories; external weights are never deleted.
    adapters = root / "adapters"
    try:
        if not stat.S_ISDIR(_owned(adapters, root).st_mode):
            raise ValueError("not a directory")
        for entry in adapters.iterdir():
            if re.fullmatch(r"[0-9a-f]{64}\.gguf(?:\.partial)?", entry.name):
                remove_file(entry, adapters)
            else:
                failures.append(str(entry.relative_to(root)))
        adapters.rmdir()
    except FileNotFoundError:
        pass
    except (OSError, ValueError):
        failures.append("adapters")
    for name in ("runtime.sqlite3", "runtime.sqlite3-wal", "runtime.sqlite3-shm", "runtime.sqlite3-journal"):
        remove_file(root / name, root)
    for entry in root.glob(".dmn-pack-*"):
        remove_file(entry, root)
    _sync_directory(root)
    return failures
