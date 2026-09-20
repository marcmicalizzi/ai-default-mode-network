"""A local, capture-only OpenAI endpoint for a disposable Open WebUI copy.

It records the exact final provider body and returns an explicit error. It never
loads a model, generates a reply, forwards traffic, or captures authentication.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .backend import sha256_file
from .storage import write_durable


def serve_capture(output: Path, model: str, port=9932):
    output.mkdir(parents=True, exist_ok=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def allowed(self):
            hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            origin = self.headers.get("Origin")
            return self.headers.get("Host") in hosts and (not origin or origin in {"http://" + h for h in hosts})

        def reply(self, status, data):
            raw = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if not self.allowed():
                return self.reply(403, {"error": "loopback origin required"})
            if self.path in {"/models", "/v1/models"}:
                return self.reply(200, {"object": "list", "data": [{"id": model, "object": "model", "owned_by": "dmn-capture-only"}]})
            self.reply(404, {"error": "capture-only endpoint"})

        def do_POST(self):
            if not self.allowed():
                return self.reply(403, {"error": "loopback origin required"})
            if self.path not in {"/chat/completions", "/v1/chat/completions", "/responses", "/v1/responses"}:
                return self.reply(404, {"error": "capture-only endpoint"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16 * 1024 * 1024 or self.headers.get_content_type() != "application/json":
                    raise ValueError("expected JSON provider request, maximum 16 MiB")
                raw = self.rfile.read(length)
                body = json.loads(raw)
                if not isinstance(body, dict):
                    raise ValueError("provider request must be a JSON object")
                responses = self.path.endswith("/responses")
                source = body.get("input") if responses else body.get("messages")
                if not isinstance(body, dict) or body.get("model") != model or not isinstance(source, (str, list) if responses else list) or not source:
                    raise ValueError("model must match and input/messages must be nonempty")
                request_id = str(uuid.uuid4())
                directory = output / request_id
                directory.mkdir()
                path = directory / "provider-request.json"
                with path.open("wb") as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                write_durable(directory / "capture.json", {"schema": 1, "captured_at": time.time(),
                    "request_id": request_id, "provider_request_sha256": sha256_file(path),
                    "model": model, "inference_performed": False, "endpoint": self.path,
                    "api": "responses" if responses else "chat_completions",
                    "limitation": "Final HTTP request only; native template rendering, token IDs and server defaults still require capture."})
                self.reply(409, {"error": {"type": "dmn_capture_complete", "message":
                    f"DMN captured provider request {request_id}. No inference was performed.", "request_id": request_id}})
            except (ValueError, TypeError, KeyError) as exc:
                self.reply(400, {"error": {"message": str(exc)}})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True, name="provider-capture").start()
    return server
