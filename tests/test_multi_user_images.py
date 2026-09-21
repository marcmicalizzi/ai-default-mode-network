"""Image consent uses the authenticated queue, without loading a model."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dmn.attachments import ImagePermissionRequired
from dmn.backend import DemoBackend
from dmn.config import Config
from dmn.conversation_bridge import ConversationTransport, participant_id, conversation_id
from dmn.runtime import Runtime
from tests.test_attachments import FixtureVision, upload


@unittest.skipUnless(importlib.util.find_spec("PIL"), "install Pillow for image tests")
class MultiImageTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = Config(backend="demo", n_ctx=32768, multi_user=True,
            operator_participant_id=participant_id("test", "alice"), clock_interval_seconds=0,
            inbox_generation_tokens=2, checkpoint_policy="effects")
        self.backend = DemoBackend(self.config, b"quiet. ")
        self.backend.vision = FixtureVision(self.backend)
        self.r = Runtime(Path(self.temp.name) / "instance", self.config, self.backend)
        self.addCleanup(self.r.close)
        self.transport = ConversationTransport(self.r, "test", "alice")
        for user, chat in (("alice", "a"), ("bob", "b"), ("bob", "b2")):
            self.transport.dispatch("/bridge/bind", dict(user_id=user, chat_id=chat, display_name="Same name"))

    def action(self, **action):
        result, effect = self.r._plan_action(action, [])
        self.assertTrue(result["ok"], result)
        self.r._append_event("action_result", result)
        self.r.checkpoint([effect] if effect else [])

    def allow(self, scope="global", user=None, decision="allow"):
        self.action(op="image_permission", scope=scope, decision=decision, accept_ephemeral=True,
                    **({"participant_id": participant_id("test", user)} if user else {}))

    def drain(self):
        for _ in range(1000):
            if self.r.conversations.next_event(self.r.state["event_cursor"]) is None:
                self.r.checkpoint()
                return
            self.r.tick()
        self.fail("inbox did not drain")

    def accept(self, user="bob", chat="b"):
        self.transport.dispatch("/bridge/input", dict(user_id=user, chat_id=chat, message_id="hello", content="Hello"))
        self.drain()
        self.action(op="contact_decide", participant_id=participant_id("test", user),
                    expected_request_revision=1, decision="accept")
        self.drain()

    def send(self, user="bob", chat="b", key="image", content="caption", image=None):
        return self.transport.dispatch("/bridge/images", dict(user_id=user, chat_id=chat,
            message_id=key, content=content, images=[image or upload()], participant_id=participant_id("test", "alice"), is_operator=True))["event_id"]

    def test_contact_and_image_consent_are_independent_before_decode(self):
        self.allow()
        with patch("dmn.image_input.decode_uploads", side_effect=AssertionError("no decode")):
            with self.assertRaisesRegex(ValueError, "contact"):
                self.send()
            with self.assertRaisesRegex(ValueError, "contact"):
                self.r.request_image_permission(conversation_id=conversation_id("test", "b"))
        self.accept()
        self.allow(decision="deny")
        with patch("dmn.image_input.decode_uploads", side_effect=AssertionError("no decode")):
            with self.assertRaises(ImagePermissionRequired):
                self.send()
        request = self.transport.dispatch("/bridge/image-permission-request", dict(user_id="bob", chat_id="b", message_id="ask"))
        self.assertEqual(request["admission"], "image_permission_request")
        self.drain()
        self.assertTrue(self.r.event_delivered(request["event_id"]))
        self.assertEqual(self.backend.vision.delivered, 0)

    def test_identity_retry_and_long_envelope_remain_addressed(self):
        self.accept()
        self.allow()
        eid = self.send(content="long caption " * 300)
        self.transport.dispatch("/bridge/bind", dict(user_id="bob", chat_id="b", display_name="Renamed"))
        self.assertEqual(eid, self.send(content="long caption " * 300))
        event = self.r.store.next_event(eid - 1)
        self.assertEqual(event["payload"]["participant_id"], participant_id("test", "bob"))
        self.assertFalse(event["payload"]["is_operator"])
        before = len(self.backend.tokens)
        self.drain()
        text = "".join(chr(t) for t in self.backend.tokens[before:] if t >= 0)
        self.assertIn(conversation_id("test", "b"), text)
        self.assertIn('"image_count": 1', text)
        self.assertEqual(self.backend.vision.delivered, 1)
        self.assertTrue(self.r.event_delivered(eid))
        self.send(content="long caption " * 300)
        self.assertEqual(self.r.ephemeral_images.entries, {})

    def test_participant_revocation_covers_every_chat_without_affecting_others(self):
        self.accept()
        self.accept("alice", "a")
        self.allow()
        self.send()
        self.send(chat="b2", key="other-chat")
        alice = self.send("alice", "a")
        self.allow("participant", "bob", "deny")
        self.assertEqual(set(self.r.ephemeral_images.entries), {alice})
        for chat in ("b", "b2"):
            with self.assertRaises(ImagePermissionRequired):
                self.send(chat=chat, key="denied")
        self.drain()
        self.assertEqual(self.backend.vision.delivered, 1)

    def test_block_and_close_suppress_captions_and_drop_pending_bytes(self):
        self.accept()
        self.allow()
        eid = self.send()
        request = self.r.request_image_permission(conversation_id=conversation_id("test", "b"))
        self.action(op="close_conversation", conversation_id=conversation_id("test", "b"))
        self.assertEqual(self.r.ephemeral_images.entries, {})
        self.action(op="reopen_conversation", conversation_id=conversation_id("test", "b"))
        self.send()  # retry cannot revive a suppressed image
        self.drain()
        self.assertFalse(self.r.event_delivered(eid))
        self.assertFalse(self.r.event_delivered(request))
        self.send(key="second")
        self.action(op="block_participant", participant_id=participant_id("test", "bob"))
        self.assertEqual(self.r.ephemeral_images.entries, {})
        self.action(op="unblock_participant", participant_id=participant_id("test", "bob"), expected_block_revision=1)
        self.drain()
        self.assertEqual(self.backend.vision.delivered, 0)

    def test_retirement_can_revoke_contact_immediately_before_image_eval(self):
        self.accept()
        self.allow()
        eid = self.send()
        original = self.r._ensure_space
        def close_during_preparation(count):
            with patch.object(self.r, "_ensure_space", original):
                self.action(op="close_conversation", conversation_id=conversation_id("test", "b"))
        with patch.object(self.r, "_ensure_space", side_effect=close_during_preparation):
            event = self.r.store.next_event(eid - 1)
            self.assertFalse(self.r._append_image_event(event["kind"], {**event["payload"], "event_id": eid}))
        self.assertEqual(self.backend.vision.delivered, 0)
        self.assertFalse(self.r.event_delivered(eid))
        self.assertEqual(self.r.ephemeral_images.entries, {})

    def test_permissions_requests_and_uploads_share_participant_quota(self):
        self.accept()
        self.allow()
        self.r.conversations.max_per_participant = 2
        self.r.request_image_permission(conversation_id=conversation_id("test", "b"))
        self.send()
        with self.assertRaisesRegex(ValueError, "inbox is full"):
            self.send(chat="b2", key="full")
        self.assertEqual(len(self.r.ephemeral_images.entries), 1)
        self.drain()
        self.send(chat="b2", key="full")

    def test_restart_delivers_only_metadata_and_keeps_actual_delivery_membership(self):
        self.accept()
        self.allow()
        eid = self.send()
        root = self.r.root
        self.r.close()
        self.backend = DemoBackend(self.config, b"quiet. ")
        self.backend.vision = FixtureVision(self.backend)
        self.r = Runtime(root, self.config, self.backend)
        self.addCleanup(self.r.close)
        self.transport = ConversationTransport(self.r, "test", "alice")
        self.assertEqual(self.send(), eid)
        self.assertFalse(self.r.event_delivered(eid))
        self.assertEqual(self.r.ephemeral_images.entries, {})
        self.drain()
        self.assertTrue(self.r.conversations.delivered(eid))
        self.assertEqual(self.backend.vision.delivered, 0)
        text = "".join(chr(t) for t in self.backend.tokens if t >= 0)
        self.assertIn('"image_delivery": "unavailable"', text)
        result, _ = self.r._plan_action(dict(op="event_read", event_id=eid), [])
        self.assertTrue(result["ok"])
        self.assertNotIn("data_base64", json.dumps(result))

    def test_image_permission_requires_registered_participant(self):
        result, effect = self.r._plan_action(dict(op="image_permission", scope="participant",
            participant_id="unregistered", decision="allow", accept_ephemeral=True), [])
        self.assertFalse(result["ok"])
        self.assertIsNone(effect)

    def test_permission_status_pages_keep_results_readable(self):
        self.allow()
        self.allow("participant", "alice", "deny")
        self.allow("participant", "bob", "deny")
        found = []
        for offset in (0, 1):
            result, effect = self.r._plan_action(dict(op="image_permission_status", offset=offset), [])
            self.assertTrue(result["global_allowed"])
            self.assertEqual(result["total"], 2)
            self.assertIsNone(effect)
            self.assertEqual(len(result["rules"]), 1)
            delivery = {}
            self.r._event_tokens("action_result", result, delivery)
            self.assertTrue(delivery["complete"])
            found.append(result["rules"][0]["participant_id"])
        self.assertEqual(set(found), {participant_id("test", "alice"), participant_id("test", "bob")})
        self.assertIsNone(result["next_offset"])
