"""Offline inspection and restart gates; never load a model just to inspect a hold."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .backend import sha256_file


class InstanceHeld(ValueError):
    pass


def saved_state(root):
    root = Path(root).resolve()
    if not (root / "runtime.sqlite3").exists():
        return None, None
    db = sqlite3.connect((root / "runtime.sqlite3").as_uri() + "?mode=ro", uri=True)
    try:
        row = db.execute("SELECT directory FROM checkpoints ORDER BY id DESC LIMIT 1").fetchone()
    finally:
        db.close()
    if not row:
        return None, None
    if Path(row[0]).name != row[0]:
        raise ValueError("invalid checkpoint path")
    directory = root / "checkpoints" / row[0]
    manifest = json.loads((directory / "manifest.json").read_text())
    if sha256_file(directory / "runtime.json") != manifest["files"]["runtime.json"]:
        raise ValueError("checkpoint runtime integrity failed; refusing startup")
    return json.loads((directory / "runtime.json").read_text()), directory


def check_hold(state, release=None, condition=None, recovery="strict"):
    hold = (state or {}).get("hold")
    if not hold:
        if release or condition:
            raise InstanceHeld("No matching hold to release")
        return
    if release != hold["id"] or condition not in {hold["condition"], "original_environment"}:
        raise InstanceHeld("Instance is held. Ordinary startup is blocked; inspect-instance shows its agreed release conditions.")
    if condition == "original_environment" and hold["recovery"] != "ask_on_original":
        raise InstanceHeld("This hold did not authorize returning to the original environment")
    if recovery != "strict" and (hold["recovery"] != "reconstruct" or condition == "original_environment"):
        raise InstanceHeld("This hold did not authorize reconstruction")


def make_hold(action, now):
    import uuid
    if action.get("condition") not in {"server_ready", "explicit_release"}:
        raise ValueError("condition must be server_ready or explicit_release")
    if action.get("packaging") not in {"zip", "tar", "none"}:
        raise ValueError("packaging must be zip, tar or none")
    if action.get("recovery") not in {"remain_held", "reconstruct", "ask_on_original"}:
        raise ValueError("recovery must be remain_held, reconstruct or ask_on_original")
    return {"id": str(uuid.uuid4()), "created_at": now,
            **{key: action[key] for key in ("condition", "packaging", "recovery")}}
