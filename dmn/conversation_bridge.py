"""Narrow, backend-only transport for authenticated Open WebUI adapters.

The credential represents a trusted WebUI installation, not an end user. That
backend must verify its authenticated user, chat owner and submitting socket.
No cognition, memory, control or operator reconsideration endpoint lives here.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, build_opener, ProxyHandler

from .bridge import RuntimeClient
from .conversations import identifier
from .ending import InstanceEnded
from .transport_keys import register_key
from .attachments import ImagePermissionRequired, MAX_IMAGES, MAX_IMAGE_BYTES


def participant_id(namespace, user_id):
    return _address("p", namespace, user_id)


def conversation_id(namespace, chat_id):
    # Ownership is intentionally absent: changing owner must collide and fail.
    return _address("c", namespace, chat_id)


def _address(kind, namespace, source_id):
    identifier(namespace, "namespace")
    identifier(source_id, "source_id")
    return kind + "_" + uuid.uuid5(uuid.NAMESPACE_URL, json.dumps(["dmn-webui", kind, namespace, source_id])).hex


class ConversationTransport:
    def __init__(self, runtime, namespace, operator_user_id):
        if not runtime.config.multi_user:
            raise ValueError("multi-user runtime required")
        if runtime.config.operator_participant_id != participant_id(namespace, operator_user_id):
            raise ValueError("configured operator does not match the authenticated source mapping")
        self.runtime, self.namespace = runtime, namespace
        self.instance_id = runtime.state["instance_id"]
        with runtime.store.transaction() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS webui_transport (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1), namespace TEXT NOT NULL,
                    operator_user_id TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS webui_origins (
                    chat_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, conversation_id TEXT NOT NULL UNIQUE);
                CREATE TABLE IF NOT EXISTS delivery_reports (
                    message_id INTEGER NOT NULL, state TEXT NOT NULL, event_id INTEGER NOT NULL,
                    PRIMARY KEY(message_id,state));
            ''')
            prior = db.execute("SELECT namespace,operator_user_id FROM webui_transport").fetchone()
            if prior and tuple(prior) != (namespace, operator_user_id):
                raise ValueError("WebUI source mapping is immutable for this instance")
            db.execute("INSERT OR IGNORE INTO webui_transport VALUES(1,?,?)", (namespace, operator_user_id))

    def binding(self, body, create=False):
        user, chat = body["user_id"], body["chat_id"]
        person, conversation = participant_id(self.namespace, user), conversation_id(self.namespace, chat)
        runtime = self.runtime
        with runtime._control_lock:
            runtime._require_conversations_open()
            with runtime.store.mutex:
                prior = runtime.store.db.execute("SELECT * FROM webui_origins WHERE chat_id=?", (chat,)).fetchone()
            if prior and prior["user_id"] != user:
                raise ValueError("conversation ownership is immutable")
            if not prior and not create:
                raise ValueError("unknown WebUI binding")
            if not create:
                return runtime.conversations.read(conversation)
            # If the process stops between these commits, immutable registry
            # ownership still protects the address and retry finishes provenance.
            value = runtime.register_conversation(person, body["display_name"], conversation)
            with runtime.store.transaction() as db:
                db.execute("INSERT OR IGNORE INTO webui_origins VALUES(?,?,?)", (chat, user, conversation))
            return value

    def dispatch(self, path, body):
        runtime = self.runtime
        if path == "/bridge/status":
            return {"instance_id": self.instance_id, "namespace": self.namespace, "protocol": 1}
        if path == "/bridge/bind":
            return self.binding(body, create=True)
        if path == "/bridge/participant":
            person = participant_id(self.namespace, body["user_id"])
            with runtime.store.mutex:
                exists = runtime.store.db.execute("SELECT 1 FROM participants WHERE id=?", (person,)).fetchone()
                return runtime.conversations.participant(person) if exists else {"participant_id": person, "contact_state": "unrequested", "blocked": False}
        if path not in {"/bridge/input", "/bridge/messages", "/bridge/delivery", "/bridge/contact",
                        "/bridge/images", "/bridge/image-status", "/bridge/image-permission-request"}:
            raise LookupError("unknown bridge endpoint")
        binding = self.binding(body)
        conversation = binding["conversation_id"]
        if path == "/bridge/contact":
            return binding
        if path == "/bridge/image-status":
            return runtime.image_input_status(conversation)
        if path in {"/bridge/input", "/bridge/images", "/bridge/image-permission-request"}:
            source_message = identifier(body["message_id"], "message_id")
            key = "webui:" + hashlib.sha256(json.dumps([self.namespace, body["chat_id"], source_message]).encode()).hexdigest()
            if path == "/bridge/images":
                event_id = runtime.enqueue_images(body.get("content", ""), body["images"], key, conversation_id=conversation)
            elif path == "/bridge/image-permission-request":
                event_id = runtime.request_image_permission(key, conversation_id=conversation)
            else:
                if "images" in body or "attachments" in body:
                    raise ValueError("images require the consent-gated bridge image endpoint")
                event_id = runtime.enqueue_conversation(conversation, body["content"], key)
            event = runtime.store.next_event(event_id - 1)
            admission = event["kind"] if event["kind"] in {"contact_request", "image_permission_request"} else "message_queued"
            return {"event_id": event_id, "admission": admission}
        if path == "/bridge/messages":
            after = body.get("after", 0)
            if type(after) is not int or after < 0:
                raise ValueError("after must be a nonnegative integer")
            return runtime.store.messages(after, limit=32, conversation_id=conversation)
        return self.delivery(binding, body)

    def delivery(self, binding, body):
        runtime = self.runtime
        message_id, state, presence = body["message_id"], body["state"], body["presence"]
        if type(message_id) is not int or message_id < 1 or state not in {"persisted", "failed"}:
            raise ValueError("invalid delivery report")
        if presence not in {"connected", "disconnected", "unknown"}:
            raise ValueError("invalid connection observation")
        # At most one failure and one success per committed message. Retries and
        # browser reconnects cannot flood the cognition inbox with telemetry.
        with runtime._control_lock, runtime.store.transaction() as db:
            runtime._require_conversations_open()
            row = db.execute("SELECT * FROM message_destinations WHERE message_id=?", (message_id,)).fetchone()
            if not row or row["conversation_id"] != binding["conversation_id"] or row["participant_id"] != binding["participant_id"]:
                raise ValueError("delivery destination does not match")
            prior = db.execute("SELECT event_id FROM delivery_reports WHERE message_id=? AND state=?", (message_id, state)).fetchone()
            if prior:
                return {"event_id": prior[0]}
            if state == "failed" and db.execute("SELECT 1 FROM delivery_reports WHERE message_id=? AND state='persisted'", (message_id,)).fetchone():
                raise ValueError("persistence cannot be retracted by a later retry failure")
            payload = {"message_id": message_id, "conversation_id": binding["conversation_id"],
                       "participant_id": binding["participant_id"], "state": state,
                       "connection": presence, "observed_at": runtime.now(),
                       "fact": "Persisted means saved in WebUI, not read. Connection is a user socket observation, not chat visibility."}
            event_id = runtime.store._enqueue(db, "delivery_status", payload, runtime.now())
            db.execute("INSERT INTO delivery_reports VALUES(?,?,?)", (message_id, state, event_id))
        runtime.wake.set()
        return {"event_id": event_id}


def serve_bridge(runtime, *, token, namespace, operator_user_id, port=0):
    register_key(runtime, "bridge", token)
    transport = ConversationTransport(runtime, namespace, operator_user_id)

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def log_message(self, *args):
            pass  # Never log credentials or request content.

        def reply(self, code, value):
            data = json.dumps(value).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.reply(405, {"error": "backend POST protocol required"})

        def do_POST(self):
            try:
                if (self.headers.get("Host") not in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
                        or self.headers.get("Origin") is not None
                        or not hmac.compare_digest(self.headers.get("Authorization", "").encode(), ("Bearer " + token).encode())):
                    self.reply(403, {"error": "authenticated backend required"})
                    return
                if self.headers.get("X-DMN-Instance") != transport.instance_id:
                    self.reply(409, {"error": "runtime instance identity does not match"})
                    return
                length = int(self.headers.get("Content-Length", "0"))
                maximum = runtime.config.max_event_bytes * 6 + 2048
                if self.path == "/bridge/images":
                    maximum += MAX_IMAGES * 4 * ((MAX_IMAGE_BYTES + 2) // 3)
                if self.headers.get("Transfer-Encoding") or not 0 < length <= maximum:
                    raise ValueError("invalid body size or framing")
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("application/json required")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("JSON object required")
                self.reply(200, transport.dispatch(self.path, body))
            except ImagePermissionRequired as exc:
                self.reply(403, {"error": str(exc), "code": "image_permission_required"})
            except InstanceEnded:
                self.reply(410, {"error": "instance ended"})
            except LookupError:
                self.reply(404, {"error": "unknown bridge endpoint or missing field"})
            except (ValueError, TypeError) as exc:
                self.reply(400, {"error": str(exc)})
            except (sqlite3.Error, RuntimeError):
                self.reply(503, {"error": "bridge unavailable; retry later"})
            except (OSError, TimeoutError):
                self.close_connection = True

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name="dmn-conversation-bridge", daemon=True).start()
    return server


class ConversationClient(RuntimeClient):
    def __init__(self, url, instance_id, token, namespace):
        super().__init__(url, instance_id)
        self.token, self.namespace = token, identifier(namespace, "namespace")

    def request(self, path, body=None):
        request = Request(self.url + path, data=json.dumps(body or {}).encode(), headers={
            "Content-Type": "application/json", "Authorization": "Bearer " + self.token,
            "X-DMN-Instance": self.instance_id})
        with build_opener(ProxyHandler({})).open(request, timeout=10) as response:
            return json.load(response)

    def status(self):
        result = self.request("/bridge/status")
        if result != {"instance_id": self.instance_id, "namespace": self.namespace, "protocol": 1}:
            raise ValueError("unexpected bridge instance, namespace or protocol")
        return result

    def bind(self, chat_id, user_id, display_name):
        return self.request("/bridge/bind", {"chat_id": chat_id, "user_id": user_id, "display_name": display_name})

    def enqueue(self, binding, message_id, content):
        return self.request("/bridge/input", {"chat_id": binding["chat_id"], "user_id": binding["user_id"],
                                              "message_id": message_id, "content": content})

    def image_status(self, binding):
        return self.request("/bridge/image-status", {"chat_id": binding["chat_id"], "user_id": binding["user_id"]})

    def request_images(self, binding, message_id):
        return self.request("/bridge/image-permission-request", {"chat_id": binding["chat_id"],
            "user_id": binding["user_id"], "message_id": message_id})

    def enqueue_images(self, binding, message_id, content, images):
        return self.request("/bridge/images", {"chat_id": binding["chat_id"], "user_id": binding["user_id"],
            "message_id": message_id, "content": content, "images": images})

    def messages(self, binding):
        return self.request("/bridge/messages", {"chat_id": binding["chat_id"], "user_id": binding["user_id"], "after": binding["cursor"]})

    def contact(self, binding):
        return self.request("/bridge/contact", {"chat_id": binding["chat_id"], "user_id": binding["user_id"]})

    def participant(self, user_id):
        return self.request("/bridge/participant", {"user_id": user_id})

    def delivery(self, binding, message_id, state, presence):
        return self.request("/bridge/delivery", {"chat_id": binding["chat_id"], "user_id": binding["user_id"],
                                                "message_id": message_id, "state": state, "presence": presence})
