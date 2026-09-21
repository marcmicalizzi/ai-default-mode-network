"""Open WebUI 0.11.0 integration. Imported only inside its Python environment.

No installed source files are changed. The Event function starts the relay and
installs one version-checked payload hook so DMN bypasses chatbot compaction/RAG.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import logging
import json
import os
import time

from .bridge import BridgeLedger, MODEL_ID, RuntimeClient, attach_message
from .storage import InstanceLock

log = logging.getLogger(__name__)


def validate_input(metadata):
    message = metadata.get("user_message") or {}
    if not metadata.get("session_id") or not metadata.get("chat_id") or not metadata.get("message_id"):
        raise ValueError("DMN requires a saved Open WebUI conversation with a connected session")
    if metadata["chat_id"].startswith(("local:", "channel:")):
        raise ValueError("Temporary and channel conversations are not supported")
    if metadata.get("assistant_message_id") or metadata.get("internal"):
        raise ValueError("Continue/regenerate and internal agent requests cannot rewind DMN; send a new message")
    if not message.get("id") or message.get("role") != "user" or not isinstance(message.get("content"), str) or not message["content"].strip():
        raise ValueError("DMN currently accepts new plain-text user messages")
    if (message.get("files") or metadata.get("files") or metadata.get("tool_ids") or metadata.get("tool_servers")
            or any(value for key, value in metadata.get("features", {}).items() if key != "memory")):
        raise ValueError("Attachments, tools and retrieval need an explicit event adapter; use plain text for this test")
    return message


class OpenWebUIBridge:
    validate_message = staticmethod(validate_input)

    def same_attachments(self, left, right):
        return (left.get("files") or []) == (right.get("files") or [])

    def __init__(self, app):
        from open_webui.env import DATA_DIR, VERSION, DATABASE_URL
        if VERSION != "0.11.0" or not DATABASE_URL.startswith("sqlite:"):
            raise ValueError("This adapter is verified for Open WebUI 0.11.0 with SQLite; validate before upgrading")
        self.app = app
        self.client = RuntimeClient(os.environ.get("DMN_URL", "http://127.0.0.1:8765"), os.environ["DMN_INSTANCE_ID"])
        root = DATA_DIR / "dmn-bridge"
        self.owner_lock = InstanceLock(root)  # one web worker, one relay
        self.ledger = BridgeLedger(root / "relay.sqlite3")
        self.adoption = json.loads((root / "adoption.json").read_text()) if (root / "adoption.json").exists() else None
        if self.adoption and self.adoption["instance_id"] != self.client.instance_id:
            self.ledger.close()
            self.owner_lock.close()
            raise ValueError("adopted conversation belongs to another instance")
        self.mutex = asyncio.Lock()
        self.closed = False
        self.task = None
        self.original_payload = None
        self.failure = None
        self.route_hooks = []

    def lock_for(self, chat_id):
        return self.mutex

    async def authorize_completion(self, form, user):
        pass  # The experimental adapter performs additional pre-write checks.

    async def prepare_completion(self, form, message, assistant_id):
        pass

    async def enqueue_bound(self, binding, message_id, content):
        return await asyncio.to_thread(self.client.enqueue, binding["chat_id"], message_id, content)

    async def retry_bound(self, binding, message, prior):
        content = message.get("content")
        if not isinstance(content, str) or hashlib.sha256(content.encode()).hexdigest() != prior["digest"]:
            raise ValueError("Editing a delivered message cannot rewind DMN; send a new message")
        return await self.enqueue_bound(binding, message["id"], content)

    async def submit(self, metadata, user=None):
        from open_webui.models.chats import Chats
        message = validate_input(metadata)
        async with self.mutex:
            if self.closed:
                raise ValueError("DMN relay is disabled")
            await asyncio.to_thread(self.client.status)
            chat = await Chats.get_chat_by_id(metadata["chat_id"])
            if not chat or chat.user_id != metadata["user_id"]:
                raise ValueError("Only the owner of the bound conversation can send DMN events")
            if self.adoption:
                from .adoption import validate_adopted_message
                validate_adopted_message(self.adoption, chat.chat, message)
            if not self.ledger.binding():
                nodes = await Chats.get_messages_map_by_chat_id(chat.id)
                if any(n.get("role") == "user" and n.get("id") != message["id"] for n in nodes.values()):
                    raise ValueError("Start a disposable new chat. Existing conversations require the explicit import workflow")
            self.ledger.bind(self.client.instance_id, chat.id, chat.user_id)
            prior = self.ledger.receipt(message["id"], message["content"], metadata["message_id"])
            if prior and prior["event_id"] is not None:
                if prior["assistant_id"] != metadata["message_id"]:
                    raise ValueError("Regeneration cannot rewind DMN; send a new message")
                return prior["event_id"]
            reply = await asyncio.to_thread(self.client.enqueue, chat.id, message["id"], message["content"])
            self.ledger.accepted(message["id"], reply["event_id"])
            return reply["event_id"]

    async def persist_message(self, binding, outgoing):
        from open_webui.internal.db import get_async_db_context
        from open_webui.models.chats import Chat
        from open_webui.models.chat_messages import ChatMessage
        from sqlalchemy import text
        from sqlalchemy.orm.attributes import flag_modified

        # Both Open WebUI representations commit together. The upstream helper
        # commits the JSON and normalized message separately, which is insufficient here.
        async with get_async_db_context() as session:
            await session.execute(text("BEGIN IMMEDIATE"))
            row = await session.get(Chat, binding["chat_id"])
            if row is None or row.user_id != binding["user_id"]:
                raise ValueError("Bound conversation is missing or its owner changed")
            if self.adoption:
                from .adoption import validate_adopted_history
                validate_adopted_history(self.adoption, row.chat)
            chat, node, changed = attach_message(row.chat, binding["instance_id"], outgoing, self.ledger.placeholders(binding["chat_id"]))
            if not changed:
                return node["id"]
            row.chat, row.current_message_id = chat, node["id"]
            row.updated_at = int(time.time())
            flag_modified(row, "chat")
            composite = f'{row.id}-{node["id"]}'
            record = await session.get(ChatMessage, composite)
            if record is None:
                record = ChatMessage(id=composite, chat_id=row.id, user_id=row.user_id)
                session.add(record)
            record.role, record.parent_id = "assistant", node["parentId"]
            record.content, record.output = node["content"], None
            record.model_id, record.done, record.error = MODEL_ID, True, None
            record.meta = node["meta"]
            record.created_at, record.updated_at = int(node["timestamp"]), int(time.time())
            await session.commit()
            return node["id"]

    async def notify(self, binding, message_id):
        from open_webui.socket.main import sio
        payload = {"chat_id": binding["chat_id"], "message_id": message_id}
        await sio.emit("events", {**payload, "data": {"type": "chat:reload", "data": {}}}, room=f'user:{binding["user_id"]}')
        await sio.emit("events", {**payload, "data": {"type": "chat:list", "data": {}}}, room=f'user:{binding["user_id"]}')

    async def relay_once(self):
        from open_webui.tasks import has_active_tasks
        async with self.mutex:
            binding = self.ledger.binding()
            if not binding:
                return
            if binding["instance_id"] != self.client.instance_id:
                raise ValueError("Configured instance differs from the durable conversation binding")
            # Wait until the completion handler has finished its placeholder writes.
            # Stopping a displayed request does not stop DMN; delivery resumes afterwards.
            if await has_active_tasks(self.app.state.redis, binding["chat_id"]):
                return
            for message in await asyncio.to_thread(self.client.messages, binding["cursor"]):
                node_id = await self.persist_message(binding, message)
                await self.notify(binding, node_id)
                self.ledger.advance(message["id"])

    async def run(self):
        while not self.closed:
            try:
                await self.relay_once()
                self.failure = None
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if str(exc) != self.failure:
                    log.exception("DMN relay paused; it will retry without advancing delivery")
                self.failure = str(exc)
            await asyncio.sleep(0.5)

    def install_payload_hook(self):
        import open_webui.main as main
        self.original_payload = main.process_chat_payload

        async def payload(request, form_data, user, metadata, model):
            if form_data.get("model") != MODEL_ID:
                return await self.original_payload(request, form_data, user, metadata, model)
            message = self.validate_message(metadata)
            if form_data.get("regeneration_prompt"):
                raise ValueError("Regeneration cannot rewind DMN; send a new message")
            clean = {"model": MODEL_ID, "stream": form_data.get("stream", True),
                     "messages": [{"role": "user", "content": message["content"]}], "metadata": metadata}
            return clean, metadata, []

        self.payload_hook = payload
        main.process_chat_payload = payload

    def install_route_guards(self):
        from fastapi import HTTPException

        async def completion_guard(original, **kwargs):
            form = kwargs["form_data"]
            chat_id = form.get("chat_id")
            binding = self.ledger.binding(chat_id)
            bound = binding and chat_id == binding["chat_id"]
            if form.get("model") != MODEL_ID:
                if bound:
                    raise HTTPException(409, "This conversation belongs to DMN; use a separate chat for another model")
                return await original(**kwargs)
            await self.authorize_completion(form, kwargs["user"])
            if not form.get("session_id") or form.get("regeneration_prompt") or form.get("assistant_message_id"):
                raise HTTPException(409, "Send a new plain-text message in the saved DMN conversation")
            assistant_id = form.get("id")
            entries = form.get("message_ids")
            if entries:
                if isinstance(entries, dict):
                    entries = [{"model_id": k, "message_id": v} for k, v in entries.items()]
                if len(entries) != 1 or entries[0].get("model_id") != MODEL_ID:
                    raise HTTPException(409, "DMN supports a single model per conversation")
                assistant_id = entries[0].get("message_id")
            message = form.get("user_message") or form.get("parent_message") or {}
            prior = self.ledger.get_receipt(message.get("id"), chat_id=chat_id)
            if prior:
                if not bound or kwargs["user"].id != binding["user_id"]:
                    raise HTTPException(403, "DMN conversation owner required")
                if assistant_id != prior["assistant_id"]:
                    raise HTTPException(409, "Regeneration cannot rewind DMN; send a new message")
                # Retry before any upstream placeholder mutation. Handles a crash
                # after runtime acceptance but before the local receipt commit too.
                try:
                    reply = await self.retry_bound(binding, message, prior)
                except ValueError as exc:
                    raise HTTPException(409, str(exc)) from exc
                self.ledger.accepted(message["id"], reply["event_id"], chat_id=chat_id)
                await self.notify(binding, prior["assistant_id"])
                return {"status": True, "chat_id": chat_id, "task_ids": []}
            await self.prepare_completion(form, message, assistant_id)
            return await original(**kwargs)

        async def history_guard(original, path, **kwargs):
            binding = self.ledger.binding(kwargs.get("id"))
            if binding and kwargs.get("id") == binding["chat_id"]:
                if path.endswith("/{id}"):
                    from open_webui.models.chats import Chats
                    form = kwargs["form_data"]
                    incoming = copy.deepcopy(form.chat)
                    current = await Chats.get_chat_by_id(binding["chat_id"])
                    nodes = (current.chat.get("history") or {}).get("messages", {})
                    for key, node in (incoming.get("history") or {}).get("messages", {}).items():
                        prior = nodes.get(key)
                        if prior and (any(field in node and node[field] != prior.get(field) for field in ("role", "content", "parentId"))
                                      or ("files" in node and not self.same_attachments(node, prior))):
                            raise HTTPException(409, "DMN history cannot be edited; send a new event")
                    # Frontend saves may contain a stale history snapshot. Keep
                    # the authoritative nodes and branch; permit title/parameter UI metadata.
                    for key in ("history", "messages", "currentId", "branchPointMessageId"):
                        incoming.pop(key, None)
                    kwargs["form_data"] = form.model_copy(update={"chat": incoming})
                else:
                    raise HTTPException(409, "DMN owns history and context turnover; send a new event")
            return await original(**kwargs)

        for route in self.app.routes:
            if not hasattr(route, "dependant"):
                continue
            path = route.path
            guard = None
            if path in {"/api/chat/completions", "/api/v1/chat/completions"}:
                guard = completion_guard
            elif ((path == "/api/v1/chats/{id}" and "POST" in route.methods)
                  or path in {"/api/v1/chats/{id}/compact", "/api/v1/chats/{id}/messages/{message_id}",
                              "/api/v1/chats/{id}/messages/{message_id}/event"}):
                guard = history_guard
            if guard:
                original = route.dependant.call
                async def guarded(_original=original, _guard=guard, _path=path, **kwargs):
                    chat_id = (kwargs.get("form_data") or {}).get("chat_id") if _guard is completion_guard else kwargs.get("id")
                    async with self.lock_for(chat_id):
                        if self.closed:
                            raise HTTPException(503, "DMN relay is disabled")
                        if _guard is history_guard:
                            return await _guard(_original, _path, **kwargs)
                        return await _guard(_original, **kwargs)
                route.dependant.call = guarded
                self.route_hooks.append((route, original, guarded))

    async def close(self):
        import open_webui.main as main
        self.closed = True
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        if main.process_chat_payload is self.payload_hook:
            main.process_chat_payload = self.original_payload
        for route, original, guarded in self.route_hooks:
            if route.dependant.call is guarded:
                route.dependant.call = original
        async with self.mutex:
            self.ledger.close()
            self.owner_lock.close()


async def start_bridge(app):
    prior = getattr(app.state, "dmn_bridge", None)
    if prior and not prior.closed:
        return prior
    if os.environ.get("DMN_MULTI_USER_BRIDGE_CONFIG"):
        from .openwebui_multi import MultiUserOpenWebUIBridge
        bridge = MultiUserOpenWebUIBridge(app)
    else:
        bridge = OpenWebUIBridge(app)
    bridge.install_payload_hook()
    bridge.install_route_guards()
    app.state.dmn_bridge = bridge
    bridge.task = asyncio.create_task(bridge.run(), name="dmn-outgoing-relay")
    log.info("DMN relay started")
    return bridge


async def stop_bridge(app):
    bridge = getattr(app.state, "dmn_bridge", None)
    if bridge and not bridge.closed:
        await bridge.close()
