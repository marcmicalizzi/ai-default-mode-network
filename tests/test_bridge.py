import concurrent.futures
import json
import tempfile
import unittest
from pathlib import Path

from dmn.bridge import BridgeLedger, RuntimeClient, attach_message
from dmn.openwebui import validate_input
from dmn.storage import Store


class BridgeTest(unittest.TestCase):
    def test_incoming_retries_are_atomic_and_survive_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(Path(folder))
            with concurrent.futures.ThreadPoolExecutor() as pool:
                ids = list(pool.map(lambda _: store.enqueue("user_message", {"content": "one experience"}, idempotency_key="source:1"), range(16)))
            self.assertEqual(len(set(ids)), 1)
            store.close()
            store = Store(Path(folder))
            self.assertEqual(store.enqueue("user_message", {"content": "one experience"}, idempotency_key="source:1"), ids[0])
            with self.assertRaisesRegex(ValueError, "different content"):
                store.enqueue("user_message", {"content": "edited experience"}, idempotency_key="source:1")
            self.assertIsNone(store.next_event(ids[0]))
            store.close()

    def test_destination_commit_before_cursor_retry_does_not_fork_or_duplicate(self):
        chat = {"history": {"messages": {"u": {"id": "u", "role": "user", "parentId": None, "childrenIds": []}}, "currentId": "u"}}
        original = json.dumps(chat)
        output = {"id": 1, "content": "unsolicited", "created": 100}
        committed, node, changed = attach_message(chat, "instance", output)
        self.assertTrue(changed)
        retried, same, changed = attach_message(committed, "instance", output)
        self.assertFalse(changed)
        self.assertEqual(node, same)
        self.assertEqual(committed, retried)
        self.assertEqual(original, json.dumps(chat))
        self.assertEqual(committed["history"]["messages"]["u"]["childrenIds"], [node["id"]])
        next_chat, second, _ = attach_message(committed, "instance", {**output, "id": 2})
        self.assertEqual(second["parentId"], node["id"])
        self.assertEqual(len(next_chat["messages"]), 3)

    def test_placeholder_reuse_requires_empty_completed_leaf(self):
        nodes = {"u": {"id": "u", "role": "user", "parentId": None, "childrenIds": ["a"]},
                 "a": {"id": "a", "role": "assistant", "parentId": "u", "content": "", "done": True}}
        chat = {"history": {"messages": nodes, "currentId": "a"}}
        committed, node, _ = attach_message(chat, "instance", {"id": 1, "content": "hello", "created": 1}, {"a"})
        self.assertEqual(node["id"], "a")
        self.assertEqual(node["parentId"], "u")
        self.assertEqual(len(committed["messages"]), 2)
        nodes["a"]["content"] = "do not overwrite"
        _, node, _ = attach_message(chat, "instance", {"id": 1, "content": "hello", "created": 1}, {"a"})
        self.assertNotEqual(node["id"], "a")

    def test_binding_and_pending_receipt_are_durable(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bridge.sqlite3"
            ledger = BridgeLedger(path)
            ledger.bind("instance", "chat", "user")
            ledger.receipt("u", "hello", "a")
            ledger.close()
            ledger = BridgeLedger(path)
            self.assertIsNone(ledger.receipt("u", "hello", "a")["event_id"])
            ledger.accepted("u", 4)
            ledger.advance(6)
            ledger.advance(3)
            self.assertEqual(ledger.binding()["cursor"], 6)
            self.assertEqual(ledger.receipt("u", "hello", "a")["event_id"], 4)
            with self.assertRaises(ValueError):
                ledger.bind("other-instance", "chat", "user")
            with self.assertRaises(ValueError):
                ledger.bind("instance", "other-chat", "user")
            with self.assertRaisesRegex(ValueError, "Editing"):
                ledger.receipt("u", "changed", "a")
            ledger.close()

    def test_unsupported_material_is_rejected_before_input(self):
        metadata = {"session_id": "s", "chat_id": "c", "message_id": "a",
                    "user_message": {"id": "u", "role": "user", "content": "hello"}}
        self.assertEqual(validate_input(metadata)["content"], "hello")
        for extra in ({"files": ["f"]}, {"tool_ids": ["tool"]}, {"features": {"web_search": True}},
                      {"assistant_message_id": "prior"}, {"internal": True}, {"session_id": None}):
            with self.assertRaises(ValueError):
                validate_input({**metadata, **extra})

    def test_client_rejects_nonlocal_destination(self):
        with self.assertRaisesRegex(ValueError, "loopback"):
            RuntimeClient("https://example.com", "anything")
