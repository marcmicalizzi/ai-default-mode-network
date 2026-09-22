"""Authenticated operational dashboard, prompt proposals and contact requests.

This credential never travels through WebUI. Mutations request maintenance or
reconsideration or prompt review; no endpoint approves behavioral text or contact.
"""
from __future__ import annotations

import hmac
import json
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .ending import InstanceEnded
from .transport_keys import register_key


def operator_status(runtime):
    value = runtime.status()
    fields = ('instance_id', 'mode', 'active_tokens', 'context_capacity', 'generated_tokens',
              'checkpoint_at', 'checkpoint_reason', 'context_retirement', 'activity', 'storage',
              'sleep_service', 'maintenance', 'action_diagnostics', 'checkpoint', 'continuity',
              'native_context_retirement_supported')
    result = {key: value[key] for key in fields if key in value}
    result['operator_ui_version'] = 2
    with runtime.store.mutex:
        result['published_messages'] = runtime.store.db.execute('SELECT COUNT(*) FROM messages').fetchone()[0]
        result['learning'] = {table: dict(runtime.store.db.execute(f'SELECT {column},COUNT(*) FROM {table} GROUP BY {column}'))
                              for table, column in (('learning_plans', 'status'), ('sleep_executions', 'status'), ('sleep_runs', 'phase'))}
    return result


def operator_prompts(runtime):
    if runtime._end_requested:
        raise InstanceEnded('This instance ended; prompt records are unavailable here')
    value = runtime.prompt_status()
    with runtime._control_lock, runtime.store.mutex:
        suppressed = {row[0] for row in runtime.store.db.execute('SELECT event_id FROM suppressed_events')}
        for item in value['proposals']:
            if item['event_id'] in suppressed:
                item['status'] = 'suppressed_before_delivery'
        directory = runtime.conversations.operator_directory()
        person = next((p for p in directory if p['is_operator']), None)
        chats = [c['conversation_id'] for c in (person or {}).get('conversations', []) if not c['closed']]
        reason = ('Wait for the promised first contact in Open WebUI.' if runtime.state.get('first_contact_gate') else
                  'The instance is on hold.' if runtime.state.get('hold') else
                  'The instance must accept operator contact first.' if not person or person['contact_state'] != 'accepted' else
                  'The operator is blocked; use reconsideration to request contact.' if person['blocked'] else
                  'An open operator conversation is required.' if not chats else '')
    value['submission'] = {'allowed': not reason, 'reason': reason, 'conversations': chats,
                           'max_text_bytes': runtime.config.max_event_bytes}
    return value


def serve_operator(runtime, *, token, port=0):
    if not runtime.config.multi_user:
        raise ValueError("multi-user runtime required")
    register_key(runtime, "operator", token)
    instance_id = runtime.state["instance_id"]
    web = Path(__file__).parent / "web"

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def log_message(self, *args):
            pass

        def reply(self, code, data, content_type="application/json; charset=utf-8"):
            raw = data if isinstance(data, bytes) else json.dumps(data).encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
            self.end_headers()
            self.wfile.write(raw)

        def allowed(self, authenticated=True):
            hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            if self.headers.get("Host") not in hosts or self.headers.get("Origin") not in {None, *("http://" + h for h in hosts)}:
                self.reply(403, {"error": "same-origin loopback request required"})
                return False
            if authenticated and (self.headers.get("X-DMN-Request") != "1" or not hmac.compare_digest(
                    self.headers.get("Authorization", "").encode(), ("Bearer " + token).encode())):
                self.reply(403, {"error": "operator access key required"})
                return False
            return True

        def do_GET(self):
            public = self.path in {"/", "/operator.js"}
            if not self.allowed(authenticated=not public):
                return
            try:
                if self.path == "/":
                    self.reply(200, (web / "operator.html").read_bytes(), "text/html; charset=utf-8")
                elif self.path == "/operator.js":
                    self.reply(200, (web / "operator.js").read_bytes(), "text/javascript; charset=utf-8")
                elif self.path == "/api/operator/contacts":
                    self.reply(200, {"instance_id": instance_id, "participants": runtime.conversations.operator_directory()})
                elif self.path == '/api/operator/status':
                    self.reply(200, operator_status(runtime))
                elif self.path == '/api/operator/prompts':
                    self.reply(200, operator_prompts(runtime))
                else:
                    self.reply(404, {"error": "unknown operator endpoint"})
            except InstanceEnded as exc:
                self.reply(410, {'error': str(exc)})
            except (sqlite3.Error, RuntimeError):
                self.reply(503, {"error": "operator data unavailable"})
            except OSError:
                self.close_connection = True

        def do_POST(self):
            if not self.allowed():
                return
            try:
                if self.path not in {'/api/operator/unblock-requests', '/api/operator/maintenance', '/api/operator/prompts'}:
                    self.reply(404, {"error": "unknown operator endpoint"})
                    return
                length = int(self.headers.get("Content-Length", "0"))
                maximum = runtime.config.max_event_bytes * 6 + 2048 if self.path == '/api/operator/prompts' else 32768
                if self.headers.get("Transfer-Encoding") or not 0 < length <= maximum or self.headers.get_content_type() != "application/json":
                    raise ValueError("bounded application/json body required")
                body = json.loads(self.rfile.read(length))
                if self.path == '/api/operator/prompts':
                    if not isinstance(body, dict) or set(body) != {'instance_id', 'text', 'base_revision', 'conversation_id'}:
                        raise ValueError('instance, proposal text, base revision and operator conversation are required')
                    if body['instance_id'] != instance_id:
                        self.reply(409, {'error': 'runtime instance identity does not match'})
                        return
                    self.reply(202, runtime.propose_prompt(body['text'], body['base_revision'], body['conversation_id']))
                    return
                if self.path == '/api/operator/maintenance':
                    if (not isinstance(body, dict) or set(body) != {'instance_id', 'action', 'reason'} or
                            body['instance_id'] != instance_id or body['action'] not in {'shutdown', 'suspend'} or
                            not isinstance(body['reason'], str) or not 1 <= len(body['reason']) <= 4000):
                        raise ValueError('instance, shutdown/suspend action and a bounded reason are required')
                    self.reply(202, runtime.control(body['action'], reason=body['reason']))
                    return
                if not isinstance(body, dict) or set(body) != {"instance_id", "participant_id", "expected_block_revision", "reason"}:
                    raise ValueError("instance, participant, block revision and reasoning are required")
                if body["instance_id"] != instance_id:
                    self.reply(409, {"error": "runtime instance identity does not match"})
                    return
                event_id = runtime.request_unblock(body["participant_id"], body["expected_block_revision"], body["reason"])
                self.reply(202, {"event_id": event_id, "fact": "Request queued. The block changes only through the model's explicit choice."})
            except InstanceEnded:
                self.reply(410, {"error": "instance ended; this request cannot restart it"})
            except (ValueError, KeyError, TypeError) as exc:
                self.reply(400, {"error": str(exc)})
            except (sqlite3.Error, RuntimeError):
                self.reply(503, {"error": "request unavailable; retry later"})
            except OSError:
                self.close_connection = True

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name="dmn-operator-contacts", daemon=True).start()
    return server
