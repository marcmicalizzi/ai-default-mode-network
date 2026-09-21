"""Durable, model-independent bookkeeping for the Open WebUI adapter."""
from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
import uuid
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, ProxyHandler


MODEL_ID = "dmn"


class RuntimeClient:
    def __init__(self, url: str, instance_id: str):
        parts = urlsplit(url)
        if parts.scheme != "http" or parts.hostname not in {"127.0.0.1", "localhost", "::1"} or parts.username or parts.path not in {"", "/"}:
            raise ValueError("DMN URL must be a loopback HTTP origin")
        uuid.UUID(instance_id)
        self.url, self.instance_id = url.rstrip("/"), instance_id

    def request(self, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        request = Request(self.url + path, data=data, headers={"Content-Type": "application/json", "X-DMN-Request": "1"})
        # Never route private runtime traffic through environment HTTP proxies.
        with build_opener(ProxyHandler({})).open(request, timeout=10) as response:
            return json.load(response)

    def status(self):
        status = self.request("/api/status")
        if status["instance_id"] != self.instance_id:
            raise ValueError("DMN instance changed; refusing to attach this conversation")
        if (status.get("multi_user") or {}).get("enabled"):
            raise ValueError("This single-user Open WebUI adapter cannot attach to an experimental multi-user runtime")
        return status

    def enqueue(self, chat_id, message_id, content):
        return self.request("/api/events", {"instance_id": self.instance_id, "content": content,
                                           "idempotency_key": f"openwebui:{chat_id}:{message_id}"})

    def messages(self, after):
        return self.request(f"/api/messages?after={after}&instance_id={self.instance_id}")


class BridgeLedger:
    def __init__(self, path: Path):
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=FULL;
            CREATE TABLE IF NOT EXISTS binding (
                singleton INTEGER PRIMARY KEY CHECK(singleton=1), instance_id TEXT NOT NULL,
                chat_id TEXT NOT NULL, user_id TEXT NOT NULL, cursor INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS receipts (
                message_id TEXT PRIMARY KEY, digest TEXT NOT NULL, assistant_id TEXT NOT NULL,
                event_id INTEGER);
        ''')

    def binding(self, chat_id=None):
        row = self.db.execute("SELECT * FROM binding").fetchone()
        return dict(row) if row else None

    def bind(self, instance_id, chat_id, user_id):
        with self.db:
            prior = self.binding()
            if prior and any(prior[k] != v for k, v in {"instance_id": instance_id, "chat_id": chat_id, "user_id": user_id}.items()):
                raise ValueError("This DMN instance is already bound to another conversation or owner")
            self.db.execute("INSERT OR IGNORE INTO binding VALUES(1,?,?,?,0)", (instance_id, chat_id, user_id))

    def receipt(self, message_id, content, assistant_id):
        digest = hashlib.sha256(content.encode()).hexdigest()
        with self.db:
            prior = self.db.execute("SELECT * FROM receipts WHERE message_id=?", (message_id,)).fetchone()
            if prior and prior["digest"] != digest:
                raise ValueError("Editing a delivered message cannot rewind DMN; send a new message")
            self.db.execute("INSERT OR IGNORE INTO receipts VALUES(?,?,?,NULL)", (message_id, digest, assistant_id))
        return dict(prior) if prior else None

    def get_receipt(self, message_id, chat_id=None):
        row = self.db.execute("SELECT * FROM receipts WHERE message_id=?", (message_id,)).fetchone()
        return dict(row) if row else None

    def accepted(self, message_id, event_id, chat_id=None):
        with self.db:
            self.db.execute("UPDATE receipts SET event_id=? WHERE message_id=?", (event_id, message_id))

    def advance(self, cursor):
        with self.db:
            self.db.execute("UPDATE binding SET cursor=max(cursor,?)", (cursor,))

    def placeholders(self, chat_id=None):
        return {r[0] for r in self.db.execute("SELECT assistant_id FROM receipts WHERE event_id IS NOT NULL")}

    def close(self):
        self.db.close()


def attach_message(chat, instance_id, message, placeholders=()):
    """Return new chat and affected node; caller commits JSON and normalized row together.

    Stable outgoing IDs make a crash between destination commit and cursor commit safe.
    Reuse only an empty, completed transport placeholder at the selected leaf.
    """
    result = copy.deepcopy(chat)
    history = result.setdefault("history", {"messages": {}, "currentId": None})
    nodes = history["messages"]
    key = f"{instance_id}:{message['id']}"
    for node in nodes.values():
        if node.get("meta", {}).get("dmn_delivery") == key:
            if node.get("content") != message["content"]:
                raise ValueError("A previously delivered DMN message was edited")
            return result, node, False
    parent = history.get("currentId")
    if parent is not None and parent not in nodes:
        raise ValueError("Broken Open WebUI selected branch")
    prior = nodes.get(parent, {})
    reusable = parent in placeholders and not prior.get("content") and not prior.get("output") and prior.get("done") is True
    node_id = parent if reusable else str(uuid.uuid5(uuid.NAMESPACE_URL, "dmn:" + key))
    if node_id in nodes and not reusable:
        raise ValueError("DMN outgoing message ID collision")
    node = {"id": node_id, "role": "assistant", "content": message["content"],
            "parentId": prior.get("parentId") if reusable else parent,
            "childrenIds": [], "model": MODEL_ID, "modelName": "DMN", "done": True,
            "timestamp": message["created"], "meta": {"dmn_delivery": key}}
    if "conversation_id" in message:
        node["meta"].update(dmn_conversation_id=message["conversation_id"], dmn_in_reply_to=message.get("in_reply_to"))
    nodes[node_id] = node
    if not reusable and parent:
        children = nodes[parent].setdefault("childrenIds", [])
        if node_id not in children:
            children.append(node_id)
    history["currentId"] = node_id
    # Open WebUI reconstructs normalized rows on read, but keep the legacy projection consistent.
    branch, current, seen = [], node_id, set()
    while current is not None:
        if current in seen or current not in nodes:
            raise ValueError("Broken or cyclic Open WebUI ancestry")
        seen.add(current)
        branch.append(nodes[current])
        current = nodes[current].get("parentId")
    result["messages"] = list(reversed(branch))
    return result, node, True
