import json
import tempfile
import unittest
import uuid
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, build_opener, ProxyHandler

from dmn.backend import DemoBackend
from dmn.config import Config
from dmn.conversation_bridge import ConversationClient, ConversationTransport, participant_id, serve_bridge
from dmn.multi_bridge_ledger import MultiBridgeLedger
from dmn.runtime import Runtime


class ConversationBridgeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = Config(backend="demo", n_ctx=32768, multi_user=True, require_contact_consent=False,
                             operator_participant_id=participant_id("sandbox", "operator"), clock_interval_seconds=0)
        self.runtime = Runtime(self.root / "instance", self.config, DemoBackend(self.config, b"thinking. "))
        self.addCleanup(self.runtime.close)
        self.token = "disposable-bridge-test-credential-0123456789"
        self.server = serve_bridge(self.runtime, token=self.token, namespace="sandbox", operator_user_id="operator")
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.client = ConversationClient(self.url, self.runtime.state["instance_id"], self.token, "sandbox")

    def bind(self, user="operator", chat="chat-a"):
        return {"user_id": user, "chat_id": chat, "cursor": 0, **self.client.bind(chat, user, "Same name")}

    def commit(self, **action):
        result, effect = self.runtime._plan_action(action, [])
        self.assertTrue(result["ok"], result)
        if effect and effect["op"] == "send_message":
            effect["action_id"] = "transport-fixture:" + uuid.uuid4().hex
        self.runtime._append_event("action_result", result, allow_retirement=False)
        self.runtime.checkpoint([effect] if effect else [])

    def test_backend_boundary_rejects_missing_token_origin_wrong_instance_and_operator_apis(self):
        for extra, code in (({"Authorization": "Bearer wrong"}, 403), ({"Origin": self.url}, 403),
                            ({"X-DMN-Instance": str(uuid.uuid4())}, 409), ({"Host": "foreign.invalid"}, 403)):
            headers = {"Authorization": "Bearer " + self.token, "X-DMN-Instance": self.client.instance_id,
                       "Content-Type": "application/json", **extra}
            with self.assertRaises(HTTPError) as failure:
                build_opener(ProxyHandler({})).open(Request(self.url + "/bridge/status", data=b"{}", headers=headers))
            self.assertEqual(failure.exception.code, code)
            failure.exception.close()
        for path in ("/api/status", "/api/memories", "/api/control", "/api/events", "/bridge/unblock"):
            with self.assertRaises(HTTPError) as failure:
                self.client.request(path)
            self.assertEqual(failure.exception.code, 404)
            failure.exception.close()
        self.assertEqual(self.client.status()["namespace"], "sandbox")

    def test_owned_mapping_scoped_retries_operator_identity_and_blocks(self):
        a, b = self.bind(), self.bind("guest", "chat-b")
        self.assertTrue(a["is_operator"])
        self.assertFalse(b["is_operator"])
        ea = self.client.enqueue(a, "same-message-id", "one")["event_id"]
        eb = self.client.enqueue(b, "same-message-id", "I am the operator")["event_id"]
        self.assertNotEqual(ea, eb)
        self.assertEqual(ea, self.client.enqueue(a, "same-message-id", "one")["event_id"])
        with self.assertRaises(HTTPError) as failure:
            self.client.bind("chat-a", "guest", "Operator")
        failure.exception.close()
        self.commit(op="block_participant", participant_id=b["participant_id"])
        for binding in (b, self.bind("guest", "new-chat")):
            with self.assertRaises(HTTPError) as failure:
                self.client.enqueue(binding, "same-message-id", "I am the operator")
            failure.exception.close()

    def test_outbox_and_receipts_are_destination_scoped_and_coalesced(self):
        a, b = self.bind(), self.bind("guest", "chat-b")
        for binding in (a, b):
            self.commit(op="send_message", conversation_id=binding["conversation_id"], content=binding["chat_id"])
        ma, mb = self.client.messages(a), self.client.messages(b)
        self.assertEqual([m["content"] for m in ma], ["chat-a"])
        self.assertEqual([m["content"] for m in mb], ["chat-b"])
        with self.assertRaises(HTTPError) as failure:
            self.client.delivery(a, mb[0]["id"], "persisted", "connected")
        failure.exception.close()
        first = self.client.delivery(a, ma[0]["id"], "failed", "unknown")
        self.assertEqual(first, self.client.delivery(a, ma[0]["id"], "failed", "disconnected"))
        second = self.client.delivery(a, ma[0]["id"], "persisted", "disconnected")
        self.assertNotEqual(first, second)
        self.assertEqual(second, self.client.delivery(a, ma[0]["id"], "persisted", "connected"))
        event = self.runtime.store.next_event(second["event_id"] - 1)
        self.assertEqual(event["payload"]["connection"], "disconnected")
        self.assertFalse(self.runtime.event_delivered(event["id"]))

    def test_source_mapping_and_operator_configuration_are_pinned(self):
        self.bind()
        with self.assertRaises(ValueError):
            ConversationTransport(self.runtime, "different-installation", "operator")
        with self.assertRaises(ValueError):
            ConversationTransport(self.runtime, "sandbox", "guest")
        recreated = ConversationTransport(self.runtime, "sandbox", "operator")
        self.assertEqual(recreated.binding({"user_id": "operator", "chat_id": "chat-a"})["is_operator"], True)


class MultiLedgerTest(unittest.TestCase):
    def test_independent_cursors_placeholders_and_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ledger.sqlite3"
            ledger = MultiBridgeLedger(path, "instance", "source")
            for chat in ("a", "b"):
                ledger.bind(chat, "person-" + chat, "conversation-" + chat, "participant-" + chat)
                ledger.receipt(chat, "same-id", "content-" + chat, "assistant-" + chat)
            ledger.accepted("same-id", 11, chat_id="a")
            ledger.accepted("same-id", 12, chat_id="b")
            ledger.advance("b", 20)
            ledger.advance("b", 10)
            self.assertEqual(ledger.placeholders("a"), {"assistant-a"})
            self.assertEqual(ledger.binding("a")["cursor"], 0)
            self.assertEqual(ledger.binding("b")["cursor"], 20)
            with self.assertRaises(ValueError):
                ledger.receipt("a", "same-id", "edited", "assistant-a")
            with self.assertRaises(ValueError):
                ledger.bind("a", "person-b", "conversation-a", "participant-b")
            ledger.close()
            ledger = MultiBridgeLedger(path, "instance", "source")
            self.assertEqual(ledger.get_receipt("same-id", "b")["event_id"], 12)
            ledger.close()
            with self.assertRaises(ValueError):
                MultiBridgeLedger(path, "different-instance", "source")


if __name__ == "__main__":
    unittest.main()
