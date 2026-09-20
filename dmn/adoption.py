"""Explicit adoption of an unchanged captured Open WebUI chat by a staged instance."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .backend import sha256_file
from .bridge import BridgeLedger
from .preservation import saved_state
from .storage import InstanceLock, write_durable


def adopt_openwebui(instance, database, capture):
    instance, database, capture = map(lambda p: Path(p).resolve(), (instance, database, capture))
    if not database.is_file():
        raise ValueError("Open WebUI database does not exist")
    owner = InstanceLock(instance)
    bridge_owner = None
    ledger = None
    try:
        state, _ = saved_state(instance)
        if not state or state["mode"] != "staged" or state["generated_tokens"] or not state.get("initial_context"):
            raise ValueError("adoption requires a staged initial-context import with no generation")
        report = json.loads((capture / "report.json").read_text())
        source_path = capture / "source-chat.json"
        if sha256_file(source_path) != report["source_chat_sha256"]:
            raise ValueError("source capture integrity failed")
        if sha256_file(instance / "import/provider-request.json") != report["provider_request_sha256"]:
            raise ValueError("instance did not import this captured provider context")
        source = json.loads(source_path.read_text())
        bridge_root = database.parent / "dmn-bridge"
        bridge_owner = InstanceLock(bridge_root)
        # Runtime and relay must be stopped. Read-only transaction gives one
        # consistent source view; recheck before every first adopted submission.
        db = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
        db.row_factory = sqlite3.Row
        try:
            db.execute("BEGIN")
            row = db.execute("SELECT * FROM chat WHERE id=?", (source["id"],)).fetchone()
            if (not row or row["user_id"] != source["user_id"] or
                    json.loads(row["chat"]) != source["chat"] or
                    row["current_message_id"] != report["source_leaf_id"]):
                raise ValueError("source conversation changed since capture; capture again before adoption")
            normalized = db.execute("SELECT id,role,content,parent_id,output,files,context_summary FROM chat_message WHERE chat_id=?", (source["id"],)).fetchall()
            for message in normalized:
                key = message["id"].removeprefix(source["id"] + "-")
                node = source["chat"]["history"]["messages"].get(key)
                content = json.loads(message["content"]) if message["content"] is not None else None
                output = json.loads(message["output"]) if message["output"] is not None else None
                files = json.loads(message["files"]) if message["files"] is not None else None
                if (node is None or message["role"] != node.get("role") or
                        (content or "") != (node.get("content") or "") or
                        output != node.get("output") or (files or []) != (node.get("files") or []) or
                        (message["context_summary"] or "") != (node.get("contextSummary") or "") or
                        message["parent_id"] != node.get("parentId")):
                    raise ValueError("normalized Open WebUI history differs from captured source")
        finally:
            db.close()
        evidence = {"instance_id": state["instance_id"], "chat_id": source["id"], "user_id": source["user_id"],
                    "source_leaf_id": report["source_leaf_id"], "source_chat_sha256": report["source_chat_sha256"],
                    "source_messages": source["chat"]["history"]["messages"],
                    "provider_request_sha256": report["provider_request_sha256"]}
        # Evidence commits before binding; interruption can safely rerun this command.
        existing = bridge_root / "adoption.json"
        if existing.exists() and json.loads(existing.read_text()) != evidence:
            raise ValueError("bridge already contains a different adoption")
        ledger = BridgeLedger(bridge_root / "relay.sqlite3")
        binding = ledger.binding()
        if binding and any(binding[k] != evidence[k] for k in ("instance_id", "chat_id", "user_id")):
            raise ValueError("bridge is already bound to another conversation")
        write_durable(existing, evidence)
        ledger.bind(state["instance_id"], source["id"], source["user_id"])
        write_durable(instance / "import/frontend-adoption.json", {k: v for k, v in evidence.items() if k != "source_messages"})
        return {"instance_id": state["instance_id"], "chat_id": source["id"], "bound": True,
                "source_chat_modified": False, "generation_started": False}
    finally:
        if ledger:
            ledger.close()
        if bridge_owner:
            bridge_owner.close()
        owner.close()


def validate_adopted_message(evidence, chat, message):
    """A captured historical input must never be re-enqueued as a new experience."""
    old = evidence["source_messages"]
    if message["id"] in old:
        raise ValueError("This message is already part of imported context; send a new message")
    validate_adopted_history(evidence, chat)
    nodes = chat["history"]["messages"]
    cursor, seen = message.get("parentId"), set()
    while cursor != evidence["source_leaf_id"]:
        if cursor is None or cursor in seen or cursor not in nodes or cursor in old:
            raise ValueError("new message must descend from the captured selected leaf")
        seen.add(cursor)
        cursor = nodes[cursor].get("parentId")


def validate_adopted_history(evidence, chat):
    old = evidence["source_messages"]
    nodes = chat["history"]["messages"]
    for key, original in old.items():
        current = nodes.get(key)
        if not current or any(current.get(field) != original.get(field) for field in ("role", "content", "parentId", "output", "files", "contextSummary")):
            raise ValueError("adopted source history changed; refusing delivery")
