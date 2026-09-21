"""Experimental authenticated multi-chat adapter, pinned to Open WebUI 0.11.0."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import weakref
from pathlib import Path

from .conversation_bridge import ConversationClient, conversation_id, participant_id
from .conversations import identifier
from .multi_bridge_ledger import MultiBridgeLedger
from .openwebui import OpenWebUIBridge, validate_input
from .storage import InstanceLock

log = logging.getLogger(__name__)


class MultiUserOpenWebUIBridge(OpenWebUIBridge):
    def __init__(self, app):
        from open_webui.env import DATA_DIR, VERSION, DATABASE_URL, WEBUI_AUTH
        if VERSION != "0.11.0" or not DATABASE_URL.startswith("sqlite:") or not WEBUI_AUTH:
            raise ValueError("multi-user adapter requires authenticated Open WebUI 0.11.0 with SQLite")
        manifest = json.loads(Path(os.environ["DMN_MULTI_USER_BRIDGE_CONFIG"]).read_text(encoding="utf-8"))
        self.client = ConversationClient(manifest["url"], manifest["instance_id"],
                                         Path(manifest["token_file"]).read_text(encoding="utf-8").strip(), manifest["namespace"])
        self.client.status()
        root = DATA_DIR / "dmn-bridge"
        if (root / "adoption.json").exists() or (root / "relay.sqlite3").exists():
            raise ValueError("multi-user experiments require a fresh bridge directory, without single-user adoption")
        self.owner_lock = InstanceLock(root)
        try:
            self.ledger = MultiBridgeLedger(root / "multi-relay.sqlite3", self.client.instance_id, self.client.namespace)
        except BaseException:
            self.owner_lock.close()
            raise
        self.app, self.adoption = app, None
        self.mutex = asyncio.Lock()
        self.chat_locks = weakref.WeakValueDictionary()
        self.closed, self.task, self.original_payload, self.failure = False, None, None, None
        self.route_hooks = []
        self.relay_slots = asyncio.Semaphore(8)
        self.failures = {}

    def lock_for(self, chat_id):
        # Active operations retain their lock. Idle locks can be collected, so
        # probing arbitrary IDs cannot grow this map indefinitely.
        if isinstance(chat_id, str) and 1 <= len(chat_id) <= 80:
            return self.chat_locks.setdefault(chat_id, asyncio.Lock())
        return self.mutex

    async def authenticate(self, chat_id, session_id, user_id, allow_new=False):
        from open_webui.models.chats import Chats
        from open_webui.models.users import Users
        from open_webui.socket.main import get_user_id_from_session_pool
        if not user_id or not session_id or get_user_id_from_session_pool(session_id) != user_id:
            raise ValueError("a connected socket belonging to the authenticated sender is required")
        chat = await Chats.get_chat_by_id(chat_id)
        user = await Users.get_user_by_id(user_id)
        if chat_id is None and allow_new and user:
            # Upstream creates the new chat and supplies its ID before the Pipe.
            return None, user
        if not chat or not user or chat.user_id != user_id:
            raise ValueError("only the saved chat's owner can send DMN input")
        return chat, user

    async def authorize_completion(self, form, user):
        from fastapi import HTTPException
        from open_webui.models.models import Models
        from open_webui.utils.access_control import check_model_access
        await check_model_access(user, await Models.get_model_by_id("dmn"))
        try:
            message = form.get("user_message") or form.get("parent_message") or {}
            new_chat = form.get("chat_id") is None and "parent_id" in form and form["parent_id"] is None
            validate_input({**form, "chat_id": "new-chat-preflight" if new_chat else form.get("chat_id"),
                            "message_id": form.get("id") or "preflight", "user_message": message})
            chat, person = await self.authenticate(form.get("chat_id"), form.get("session_id"), user.id, allow_new=True)
            if chat is None:
                policy = await asyncio.to_thread(self.client.participant, person.id)
                if policy.get("blocked") or policy.get("contact_state") in {"pending", "deferred", "declined"}:
                    raise ValueError("DMN has not permitted a new conversation with this participant; no message was delivered")
            else:
                await self.bind_chat(chat, person, message)
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(403, str(exc)) from exc

    async def bind_chat(self, chat, person, message):
        if not self.ledger.binding(chat.id):
            from open_webui.models.chats import Chats
            nodes = await Chats.get_messages_map_by_chat_id(chat.id)
            if any(n.get("role") == "user" and n.get("id") != message["id"] for n in nodes.values()):
                raise ValueError("start a new chat; multi-user history adoption is not supported")
        remote = await asyncio.to_thread(self.client.bind, chat.id, person.id, person.name)
        self.validate_binding(remote, chat.id, person.id)
        if remote["closed"] or remote["blocked"]:
            raise ValueError("DMN has closed this conversation or blocked this participant")
        if remote["contact_state"] == "declined":
            raise ValueError("DMN has declined contact; your message has not been delivered")
        self.ledger.bind(chat.id, person.id, remote["conversation_id"], remote["participant_id"])
        return self.ledger.binding(chat.id)

    def validate_binding(self, remote, chat_id, user_id):
        if (remote["conversation_id"] != conversation_id(self.client.namespace, chat_id)
                or remote["participant_id"] != participant_id(self.client.namespace, user_id)):
            raise ValueError("bridge returned an unexpected identity mapping")

    async def prepare_completion(self, form, message, assistant_id):
        from fastapi import HTTPException
        from open_webui.models.chats import Chats
        try:
            identifier(message["id"], "message_id")
            identifier(assistant_id, "assistant_id")
            if message["id"] == assistant_id:
                raise ValueError("user and assistant message IDs must differ")
            chat = await Chats.get_chat_by_id(form["chat_id"]) if form.get("chat_id") else None
            history = (chat.chat.get("history") or {}) if chat else {}
            nodes = await Chats.get_messages_map_by_chat_id(chat.id) if chat else {}
            existing = nodes.get(message["id"])
            if existing and (existing.get("role") != "user" or existing.get("content") != message["content"]):
                raise ValueError("new input cannot replace an existing message")
            placeholder = nodes.get(assistant_id)
            if placeholder and (placeholder.get("role") != "assistant" or placeholder.get("content") or placeholder.get("output")
                                or placeholder.get("meta", {}).get("dmn_delivery") or placeholder.get("parentId") != message["id"]):
                raise ValueError("assistant placeholder cannot replace an existing message")
            used = self.ledger.db.execute("SELECT message_id FROM receipts WHERE chat_id=? AND assistant_id=?",
                                          (form.get("chat_id"), assistant_id)).fetchone()
            if used and used[0] != message["id"]:
                raise ValueError("assistant placeholder belongs to another input")
            # Append new input to the authoritative leaf, including any spontaneous
            # reply saved since the browser last refreshed. Never accept frontend
            # delivery metadata, summaries, output records or branch rewrites.
            parent = existing.get("parentId") if existing else history.get("currentId")
            form["user_message"] = {"id": message["id"], "role": "user", "content": message["content"],
                                    "parentId": parent, "childrenIds": [], "timestamp": int(time.time()), "meta": {}}
            form["parent_id"] = parent
            form.pop("parent_message", None)
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(409, str(exc)) from exc

    async def enqueue_bound(self, binding, message_id, content):
        return await asyncio.to_thread(self.client.enqueue, binding, message_id, content)

    async def submit(self, metadata, user=None):
        message = validate_input(metadata)
        user_id = (user or {}).get("id")
        if user_id != metadata.get("user_id"):
            raise ValueError("authenticated Pipe identity and server metadata disagree")
        async with self.lock_for(metadata["chat_id"]):
            if self.closed:
                raise ValueError("DMN relay is disabled")
            chat, person = await self.authenticate(metadata["chat_id"], metadata["session_id"], user_id)
            binding = await self.bind_chat(chat, person, message)
            if not binding or binding["user_id"] != person.id:
                raise ValueError("DMN chat requires authenticated completion preflight")
            self.ledger.receipt(chat.id, message["id"], message["content"], metadata["message_id"])
            # Even a known retry must obey a block or closure committed since acceptance.
            reply = await self.enqueue_bound(binding, message["id"], message["content"])
            self.ledger.accepted(message["id"], reply["event_id"], chat_id=chat.id)
            return reply

    async def update_contact_status(self, binding):
        from open_webui.models.chats import Chats
        contact = await asyncio.to_thread(self.client.contact, binding)
        if not contact["contact_request_event_id"]:
            return
        status = json.dumps([contact["contact_state"], contact["contact_reason"]])
        if self.ledger.contact_status(binding["chat_id"]) == status:
            return
        descriptions = {"pending": "Waiting for DMN's consent. Your first message is held outside its context.",
                        "deferred": "DMN deferred contact. Your first message remains withheld.",
                        "declined": "DMN declined contact. Your first message was not delivered.",
                        "accepted": "DMN accepted contact. Eligible held input is now queued."}
        description = descriptions.get(contact["contact_state"], "Contact has not been requested.")
        if contact["contact_reason"]:
            description += " DMN's reason: " + contact["contact_reason"]
        chat = await Chats.get_chat_by_id(binding["chat_id"])
        if not chat or chat.user_id != binding["user_id"]:
            return
        nodes = await Chats.get_messages_map_by_chat_id(chat.id)
        for node_id in self.ledger.placeholders(chat.id):
            node = nodes.get(node_id)
            if not node or node.get("content") or node.get("meta", {}).get("dmn_delivery"):
                continue
            await Chats.upsert_message_to_chat_by_id_and_message_id(chat.id, node_id, {
                "statusHistory": [{"action": "contact_consent", "description": description, "done": True}],
                "meta": {**(node.get("meta") or {}), "dmn_contact_state": contact["contact_state"]}})
            await self.notify(binding, node_id)
        self.ledger.save_contact_status(chat.id, status)

    async def persist_message(self, binding, outgoing):
        if (outgoing.get("conversation_id") != binding["conversation_id"]
                or outgoing.get("participant_id") != binding["participant_id"]):
            raise ValueError("outgoing message is addressed to another destination")
        return await super().persist_message(binding, outgoing)

    async def presence(self, user_id):
        from open_webui.socket.main import get_session_ids_from_room, SESSION_POOL, SESSION_POOL_TIMEOUT
        try:
            sessions = get_session_ids_from_room(f"user:{user_id}")
            for sid in sessions:
                entry = SESSION_POOL.get(sid)
                if entry and entry.get("id") == user_id and time.time() - entry.get("last_seen_at", 0) <= SESSION_POOL_TIMEOUT:
                    return "connected"
            return "disconnected"
        except Exception:
            return "unknown"

    async def relay_binding(self, chat_id):
        from open_webui.tasks import has_active_tasks
        async with self.relay_slots, self.lock_for(chat_id):
            if self.closed:
                return
            binding = self.ledger.binding(chat_id)
            if await has_active_tasks(self.app.state.redis, chat_id):
                return
            await self.update_contact_status(binding)
            for message in await asyncio.to_thread(self.client.messages, binding):
                try:
                    node_id = await self.persist_message(binding, message)
                except Exception:
                    await asyncio.to_thread(self.client.delivery, binding, message["id"], "failed", await self.presence(binding["user_id"]))
                    raise
                # Persistence and its durable report precede cursor advancement.
                # Notifications are best effort and cannot undo a committed save.
                await asyncio.to_thread(self.client.delivery, binding, message["id"], "persisted", await self.presence(binding["user_id"]))
                self.ledger.advance(chat_id, message["id"])
                try:
                    await self.notify(binding, node_id)
                except Exception:
                    log.warning("DMN reply saved; socket notification unavailable")

    async def relay_once(self):
        bindings = self.ledger.bindings()
        results = await asyncio.gather(*(self.relay_binding(b["chat_id"]) for b in bindings), return_exceptions=True)
        for binding, result in zip(bindings, results):
            chat_id = binding["chat_id"]
            if isinstance(result, BaseException):
                if self.failures.get(chat_id) != str(result):
                    log.warning("DMN destination delivery pending: %s", type(result).__name__)
                self.failures[chat_id] = str(result)
            else:
                self.failures.pop(chat_id, None)

    async def close(self):
        # Super restores hooks and waits on the initial-binding mutex. Drain
        # established-chat operations too before closing their shared ledger.
        self.closed = True
        acquired = []
        try:
            for lock in list(self.chat_locks.values()):
                await lock.acquire()
                acquired.append(lock)
            await super().close()
        finally:
            for lock in acquired:
                lock.release()
