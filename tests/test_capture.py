import http.client
import json
import tempfile
import unittest
from pathlib import Path

from dmn.capture import serve_capture


class CaptureTest(unittest.TestCase):
    def test_responses_body_is_preserved_without_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            server = serve_capture(Path(folder), "gemma", 0)
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
            try:
                raw = b'{"model":"gemma","input":[{"role":"user","content":"original"}],"instructions":"system","temperature":0.7}'
                connection.request("POST", "/v1/responses", body=raw, headers={"Content-Type": "application/json"})
                response = connection.getresponse()
                self.assertEqual(response.status, 409)
                directory = Path(folder) / json.loads(response.read())["error"]["request_id"]
                self.assertEqual((directory / "provider-request.json").read_bytes(), raw)
                evidence = json.loads((directory / "capture.json").read_text())
                self.assertEqual(evidence["api"], "responses")
                self.assertFalse(evidence["inference_performed"])
                connection.request("POST", "/v1/responses", body="[]", headers={"Content-Type": "application/json"})
                response = connection.getresponse()
                self.assertEqual(response.status, 400)
                response.read()
            finally:
                connection.close()
                server.shutdown()
                server.server_close()

    def test_exact_final_request_is_archived_without_fabricated_reply(self):
        with tempfile.TemporaryDirectory() as folder:
            server = serve_capture(Path(folder), "original-model", 0)
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
            try:
                raw = b'{"model": "original-model", "messages": [{"role":"system","content":"summary"},{"role":"user","content":"retained"}], "temperature":1.0}'
                connection.request("POST", "/v1/chat/completions", body=raw, headers={"Content-Type": "application/json", "Authorization": "Bearer secret-do-not-store"})
                response = connection.getresponse()
                self.assertEqual(response.status, 409)
                error = json.loads(response.read())["error"]
                directory = Path(folder) / error["request_id"]
                self.assertEqual((directory / "provider-request.json").read_bytes(), raw)
                evidence = json.loads((directory / "capture.json").read_text())
                self.assertFalse(evidence["inference_performed"])
                self.assertNotIn("secret-do-not-store", (directory / "capture.json").read_text())
            finally:
                connection.close()
                server.shutdown()
                server.server_close()
