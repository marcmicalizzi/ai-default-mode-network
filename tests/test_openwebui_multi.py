"""Adapter concurrency and failure policy without importing WebUI or a model."""
import asyncio
import tempfile
import types
import unittest
import weakref
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from dmn.multi_bridge_ledger import MultiBridgeLedger
from dmn.openwebui_multi import MultiUserOpenWebUIBridge


class MultiRelayTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.bridge = object.__new__(MultiUserOpenWebUIBridge)
        bridge = self.bridge
        bridge.ledger = MultiBridgeLedger(Path(self.temp.name) / "relay.db", "instance", "source")
        bridge.mutex, bridge.chat_locks = asyncio.Lock(), weakref.WeakValueDictionary()
        bridge.closed, bridge.failures = False, {}
        bridge.relay_slots = asyncio.Semaphore(8)
        bridge.app = types.SimpleNamespace(state=types.SimpleNamespace(redis=None))
        bridge.client = Mock()
        bridge.client.messages.side_effect = lambda b: [{"id": 1 if b["chat_id"] == "a" else 2, "content": b["chat_id"]}]
        bridge.presence = AsyncMock(return_value="unknown")
        bridge.update_contact_status = AsyncMock()
        bridge.notify = AsyncMock()
        for chat in ("a", "b"):
            bridge.ledger.bind(chat, "user-" + chat, "conversation-" + chat, "person-" + chat)
        self.tasks = patch.dict("sys.modules", {"open_webui.tasks": types.SimpleNamespace(has_active_tasks=AsyncMock(return_value=False))})
        self.tasks.start()

    async def asyncTearDown(self):
        self.tasks.stop()
        self.bridge.ledger.close()
        self.temp.cleanup()

    async def test_slow_failed_destination_does_not_prevent_other_chat_delivery(self):
        release = asyncio.Event()
        other_saved = asyncio.Event()
        async def persist(binding, message):
            if binding["chat_id"] == "a":
                await release.wait()
                raise ValueError("missing chat")
            other_saved.set()
            return "saved-node"
        self.bridge.persist_message = persist
        task = asyncio.create_task(self.bridge.relay_once())
        try:
            await asyncio.wait_for(other_saved.wait(), 2)
            self.assertFalse(task.done())
        finally:
            release.set()
            await task
        self.assertEqual(self.bridge.ledger.binding("a")["cursor"], 0)
        self.assertEqual(self.bridge.ledger.binding("b")["cursor"], 2)
        self.assertEqual(self.bridge.client.delivery.call_count, 2)
        self.assertIn("a", self.bridge.failures)

    async def test_notification_failure_does_not_retract_durable_delivery(self):
        self.bridge.persist_message = AsyncMock(return_value="saved-node")
        self.bridge.notify.side_effect = OSError("socket unavailable")
        await self.bridge.relay_once()
        self.assertEqual(self.bridge.ledger.binding("a")["cursor"], 1)
        self.assertEqual(self.bridge.ledger.binding("b")["cursor"], 2)
        self.assertEqual(self.bridge.failures, {})
        self.assertEqual({c.args[2] for c in self.bridge.client.delivery.call_args_list}, {"persisted"})

    async def test_receipt_failure_leaves_cursor_for_stable_replay(self):
        self.bridge.persist_message = AsyncMock(return_value="saved-node")
        self.bridge.client.delivery.side_effect = OSError("bridge temporarily unavailable")
        await self.bridge.relay_once()
        self.assertEqual(self.bridge.ledger.binding("a")["cursor"], 0)
        self.assertEqual(self.bridge.ledger.binding("b")["cursor"], 0)
        self.bridge.notify.assert_not_awaited()

    async def test_wrong_destination_cannot_reach_persistence(self):
        binding = self.bridge.ledger.binding("a")
        with self.assertRaisesRegex(ValueError, "another destination"):
            await self.bridge.persist_message(binding, {"conversation_id": "conversation-b", "participant_id": "person-b"})


if __name__ == "__main__":
    unittest.main()
