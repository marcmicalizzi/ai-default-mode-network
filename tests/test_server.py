import http.client
import json
import tempfile
import unittest
from pathlib import Path

from dmn.backend import DemoBackend
from dmn.config import Config
from dmn.runtime import Runtime
from dmn.server import serve


class ServerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.config = Config(backend="demo", clock_interval_seconds=0)
        self.runtime = Runtime(Path(self.temp.name), self.config, DemoBackend(self.config))
        self.server = serve(self.runtime, 0)
        self.port = self.server.server_port

    def tearDown(self):
        self.runtime.exit_requested.set()
        self.server.shutdown()
        self.server.server_close()
        self.runtime.close()
        self.temp.cleanup()

    def request(self, method, path, payload=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        body = json.dumps(payload) if payload is not None else None
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        data = response.read()
        status = response.status
        connection.close()
        return status, json.loads(data)

    def test_event_acceptance_is_durable_and_does_not_rebuild_or_generate(self):
        count = self.runtime.backend.decoded_tokens
        code, body = self.request("POST", "/api/events", {"content": "arrived from UI"},
                                  {"Content-Type": "application/json", "X-DMN-Request": "1"})
        self.assertEqual(code, 202)
        self.assertEqual(body["event_id"], 1)
        self.assertEqual(self.runtime.backend.decoded_tokens, count)
        self.assertEqual(self.runtime.store.next_event(0)["payload"]["content"], "arrived from UI")
        code, body = self.request("GET", "/api/messages")
        self.assertEqual((code, body), (200, []))

    def test_foreign_origin_and_host_cannot_submit_events(self):
        for headers in ({"Origin": "https://example.com"}, {"Host": "example.com"}, {}):
            supplied = {"Content-Type": "application/json", **headers}
            if headers:
                supplied["X-DMN-Request"] = "1"
            code, _ = self.request("POST", "/api/events", {"content": "not accepted"}, supplied)
            self.assertEqual(code, 403)
        self.assertIsNone(self.runtime.store.next_event(0))

    def test_shutdown_preparation_override_is_validated_and_applied(self):
        headers = {"Content-Type": "application/json", "X-DMN-Request": "1"}
        for seconds in (-1, True, "0", float("inf")):
            code, _ = self.request("POST", "/api/control", {"action": "shutdown", "preparation_seconds": seconds}, headers)
            self.assertEqual(code, 400)
        self.assertFalse(self.runtime.exit_requested.is_set())
        code, _ = self.request("POST", "/api/control", {"action": "shutdown", "preparation_seconds": 0}, headers)
        self.assertEqual(code, 202)
        self.runtime.tick()
        self.assertEqual(self.runtime.state["mode"], "suspended")
        self.assertEqual(self.runtime.state["last_suspension"]["preparation_tokens_used"], 0)

    def test_bridge_identity_and_retry_contract(self):
        headers = {"Content-Type": "application/json", "X-DMN-Request": "1"}
        body = {"content": "one input", "instance_id": self.runtime.state["instance_id"], "idempotency_key": "webui:chat:msg"}
        first = self.request("POST", "/api/events", body, headers)
        self.assertEqual(first, self.request("POST", "/api/events", body, headers))
        self.assertEqual(first[0], 202)
        for change in ({"instance_id": "wrong"}, {"content": "changed"}):
            self.assertEqual(self.request("POST", "/api/events", {**body, **change}, headers)[0], 400)
        self.assertIsNone(self.runtime.store.next_event(first[1]["event_id"]))
        self.assertEqual(self.request("GET", "/api/messages?instance_id=wrong")[0], 400)

    def test_storage_retry_control_and_status(self):
        headers = {"Content-Type": "application/json", "X-DMN-Request": "1"}
        code, _ = self.request("POST", "/api/control", {"action": "retry_checkpoint"}, headers)
        self.assertEqual(code, 202)
        self.assertTrue(self.runtime._storage_retry.is_set())
        code, body = self.request("GET", "/api/status")
        self.assertEqual(body["storage"]["reserve_bytes"], 256 * 1024 * 1024)
        self.assertIsNone(body["storage"]["blocked"])
        code, _ = self.request("POST", "/api/control", {"action": "retry_checkpoint", "preparation_seconds": 0}, headers)
        self.assertEqual(code, 400)

    def test_sse_reconnect_uses_durable_cursor(self):
        for _ in range(205):
            self.runtime.tick()
        self.assertEqual(len(self.runtime.store.messages()), 1)
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        connection.request("GET", "/api/stream", headers={"Last-Event-ID": "1"})
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        self.assertEqual(response.readline(), b"event: status\n")
        self.runtime.exit_requested.set()
        connection.close()
