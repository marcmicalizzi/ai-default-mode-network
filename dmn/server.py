from __future__ import annotations

import json
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .storage import json_text
from .ending import InstanceEnded
from .attachments import ImagePermissionRequired, MAX_IMAGES, MAX_IMAGE_BYTES


def serve(runtime, port=8765):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def allowed_host(self):
            return self.headers.get("Host") in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}

        def reply(self, code, data, content_type="application/json; charset=utf-8"):
            raw = data if isinstance(data, bytes) else json_text(data).encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if not self.allowed_host():
                self.reply(403, {"error": "loopback host required"})
                return
            parts = urlsplit(self.path)
            query = parse_qs(parts.query)
            try:
                if ((runtime.status().get("ending") or {}).get("mode") == "erase" and
                        parts.path not in {"/", "/api/status", "/api/stream"}):
                    raise InstanceEnded("This instance ended with erasure; its records are unavailable")
                if parts.path == "/":
                    self.reply(200, (Path(__file__).parent / "web" / "index.html").read_bytes(), "text/html; charset=utf-8")
                elif parts.path == "/api/status":
                    self.reply(200, runtime.status())
                elif parts.path == "/api/messages":
                    expected = query.get("instance_id", [None])[0]
                    if expected is not None and expected != runtime.status()["instance_id"]:
                        raise ValueError("runtime instance identity does not match")
                    self.reply(200, runtime.store.messages(int(query.get("after", [0])[0])))
                elif parts.path == "/api/prompts":
                    self.reply(200, runtime.prompt_status())
                elif parts.path == "/api/memories":
                    if "path" in query:
                        self.reply(200, {"path": query["path"][0], "content": runtime.store.memory_read(query["path"][0])})
                    else:
                        self.reply(200, runtime.store.memory_list(offset=int(query.get("offset", [0])[0]), limit=100))
                elif parts.path == "/api/events":
                    after = int(query.get("after", [0])[0])
                    with runtime.store.mutex:
                        rows = runtime.store.db.execute("SELECT * FROM events WHERE id>? ORDER BY id LIMIT 200", (after,)).fetchall()
                    self.reply(200, [{**dict(row), "payload": json.loads(row["payload"])} for row in rows])
                elif parts.path == "/api/stream":
                    cursor = max(int(query.get("after", [0])[0]), int(self.headers.get("Last-Event-ID", "0")))
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    # Short-lived connections allow shutdown; EventSource reconnects
                    # with the last durable message ID. Idle ticks never publish thoughts.
                    for _ in range(30):
                        try:
                            messages = [] if runtime._end_requested else runtime.store.messages(cursor)
                        except sqlite3.ProgrammingError:
                            if not runtime._end_requested and not runtime.stopped.is_set():
                                raise
                            messages = []
                        for message in messages:
                            self.wfile.write(f'id: {message["id"]}\nevent: message\ndata: {json_text(message)}\n\n'.encode())
                            cursor = message["id"]
                        self.wfile.write(f'event: status\ndata: {json_text(runtime.status())}\n\n'.encode())
                        self.wfile.flush()
                        if runtime.stopped.wait(1):
                            self.wfile.write(f'event: status\ndata: {json_text(runtime.status())}\n\n'.encode())
                            self.wfile.flush()
                            break
                    self.close_connection = True
                else:
                    self.reply(404, {"error": "not found"})
            except InstanceEnded as exc:
                self.reply(410, {"error": str(exc)})
            except sqlite3.ProgrammingError:
                if not runtime._end_requested and not runtime.stopped.is_set():
                    raise
                self.reply(410, {"error": "Runtime stopped; records are unavailable"})
            except (ValueError, KeyError) as exc:
                self.reply(400, {"error": str(exc)})
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                self.close_connection = True

        def do_POST(self):
            origin = self.headers.get("Origin")
            allowed_origins = {f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"}
            if not self.allowed_host() or (origin and origin not in allowed_origins) or self.headers.get("X-DMN-Request") != "1":
                self.reply(403, {"error": "same-origin request with X-DMN-Request: 1 required"})
                self.close_connection = True
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                maximum = runtime.config.max_event_bytes * 6 + 1024
                if self.path == "/api/images":
                    # Reject before reading an image body when consent is absent.
                    with runtime._control_lock:
                        runtime._require_image_input_open()
                        runtime.image_permissions.ticket("local-user")
                    maximum += MAX_IMAGES * 4 * ((MAX_IMAGE_BYTES + 2) // 3)
                if not 0 < length <= maximum:
                    raise ValueError("request body missing or too large")
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("application/json required")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("request must be a JSON object")
                if self.path in {"/api/events", "/api/images", "/api/image-permission-request"}:
                    expected = body.get("instance_id")
                    if expected is not None and expected != runtime.status()["instance_id"]:
                        raise ValueError("runtime instance identity does not match")
                    if self.path == "/api/images":
                        event_id = runtime.enqueue_images(body.get("content", ""), body["images"], body.get("idempotency_key"))
                    elif self.path == "/api/image-permission-request":
                        event_id = runtime.request_image_permission(body.get("idempotency_key"))
                    else:
                        if "images" in body or "attachments" in body:
                            raise ValueError("image uploads require the consent-gated /api/images endpoint")
                        event_id = runtime.enqueue(body["content"], body.get("idempotency_key"))
                    self.reply(202, {"event_id": event_id})
                elif self.path == "/api/control":
                    result = runtime.control(body["action"], preparation_seconds=body.get("preparation_seconds"), reason=body.get("reason"))
                    self.reply(202, {"accepted": body["action"], **result})
                elif self.path == "/api/prompts":
                    self.reply(202, runtime.propose_prompt(body["text"], body["base_revision"]))
                else:
                    self.reply(404, {"error": "not found"})
            except ImagePermissionRequired as exc:
                self.reply(403, {"error": str(exc), "code": "image_permission_required"})
                self.close_connection = True
            except InstanceEnded as exc:
                self.reply(410, {"error": str(exc)})
                self.close_connection = True
            except (ValueError, KeyError, TypeError) as exc:
                self.reply(400, {"error": str(exc)})
                self.close_connection = True

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    server_thread = threading.Thread(target=server.serve_forever, name="dmn-http", daemon=True)
    server_thread.start()
    return server
