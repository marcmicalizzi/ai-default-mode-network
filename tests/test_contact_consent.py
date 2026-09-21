import json
import tempfile
import unittest
from pathlib import Path

from dmn.backend import DemoBackend
from dmn.config import Config
from dmn.runtime import Runtime


class ContactConsentTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "instance"
        self.config = Config(backend="demo", n_ctx=32768, multi_user=True, operator_participant_id="operator",
                             clock_interval_seconds=0, inbox_generation_tokens=4)
        self.runtime = Runtime(self.root, self.config, DemoBackend(self.config, b"Private fixture. "))
        self.addCleanup(lambda: self.runtime.close())
        for person in ("operator", "guest"):
            self.runtime.register_conversation(person, "Same name", "chat-" + person)

    def commit(self, **action):
        result, effect = self.runtime._plan_action(action, [])
        self.assertTrue(result["ok"], result)
        self.runtime._append_event("action_result", result, allow_retirement=False)
        self.runtime.checkpoint([effect] if effect else [])

    def pending(self, person="guest"):
        event_id = self.runtime.enqueue_conversation("chat-" + person, "WITHHELD_PRIVATE_FIRST_MESSAGE", person + ":first")
        for _ in range(100):
            if self.runtime.event_delivered(event_id):
                break
            self.runtime.tick()
        self.assertTrue(self.runtime.event_delivered(event_id))
        return event_id

    def decide(self, decision, person="guest", **extra):
        self.commit(op="contact_decide", participant_id=person, expected_request_revision=1, decision=decision, **extra)

    def input_events(self):
        return self.runtime.store.db.execute("SELECT * FROM events WHERE kind='user_message'").fetchall()

    def test_first_message_cannot_enter_sequence_or_event_read_before_explicit_acceptance(self):
        event_id = self.pending()
        event = self.runtime.store.next_event(event_id - 1)
        self.assertEqual(event["kind"], "contact_request")
        self.assertNotIn("WITHHELD_PRIVATE_FIRST_MESSAGE", json.dumps(event))
        result, _ = self.runtime._plan_action({"op": "event_read", "event_id": event_id}, [])
        self.assertNotIn("WITHHELD_PRIVATE_FIRST_MESSAGE", json.dumps(result))
        self.assertNotIn("WITHHELD_PRIVATE_FIRST_MESSAGE", "".join(map(chr, self.runtime.backend.tokens)))
        self.assertEqual(self.input_events(), [])
        result, _ = self.runtime._plan_action({"op": "send_message", "conversation_id": "chat-guest", "content": "too early"}, [])
        self.assertFalse(result["ok"])
        self.decide("accept")
        inputs = self.input_events()
        self.assertEqual(len(inputs), 1)
        self.assertEqual(json.loads(inputs[0]["payload"])["content"], "WITHHELD_PRIVATE_FIRST_MESSAGE")
        self.assertFalse(self.runtime.event_delivered(inputs[0]["id"]))
        self.assertEqual(self.runtime.enqueue_conversation("chat-guest", "WITHHELD_PRIVATE_FIRST_MESSAGE", "guest:first"), inputs[0]["id"])

    def test_operator_has_no_consent_exemption_and_silence_or_restart_does_not_accept(self):
        self.pending("operator")
        self.runtime.checkpoint()
        self.runtime.close()
        self.runtime = Runtime(self.root, self.config, DemoBackend(self.config, b"Private fixture. "))
        self.assertEqual(self.runtime.conversations.participant("operator")["contact_state"], "pending")
        self.assertEqual(self.input_events(), [])
        self.decide("defer", "operator", reason="I want more time")
        self.assertEqual(self.input_events(), [])
        self.assertEqual(self.runtime.conversations.participant("operator")["contact_reason"], "I want more time")
        self.decide("accept", "operator")
        self.assertEqual(len(self.input_events()), 1)

    def test_decline_discards_first_message_and_later_acceptance_does_not_replay_it(self):
        self.pending()
        self.decide("decline", reason="I do not want contact now")
        with self.assertRaisesRegex(ValueError, "additional messages"):
            self.runtime.enqueue_conversation("chat-guest", "Please accept", "guest:second")
        self.decide("accept")
        self.assertEqual(self.input_events(), [])
        with self.assertRaisesRegex(ValueError, "discarded"):
            self.runtime.enqueue_conversation("chat-guest", "WITHHELD_PRIVATE_FIRST_MESSAGE", "guest:first")
        self.runtime.enqueue_conversation("chat-guest", "New permitted message", "guest:new")
        self.assertEqual(len(self.input_events()), 1)

    def test_close_or_block_discards_held_message_and_accept_does_not_override(self):
        self.pending()
        self.commit(op="close_conversation", conversation_id="chat-guest")
        self.commit(op="block_participant", participant_id="guest")
        result, effect = self.runtime._plan_action({"op": "contact_decide", "participant_id": "guest", "expected_request_revision": 1, "decision": "accept"}, [])
        self.assertFalse(result["ok"])
        self.assertIsNone(effect)
        self.commit(op="unblock_participant", participant_id="guest", expected_block_revision=1)
        self.decide("accept")
        self.assertEqual(self.input_events(), [])
        with self.assertRaisesRegex(ValueError, "closed"):
            self.runtime.enqueue_conversation("chat-guest", "Still closed")

    def test_acceptance_and_release_roll_back_together_if_checkpoint_publication_fails(self):
        self.pending()
        result, effect = self.runtime._plan_action({"op": "contact_decide", "participant_id": "guest", "expected_request_revision": 1, "decision": "accept"}, [])
        self.assertTrue(result["ok"])
        with self.assertRaisesRegex(ValueError, "unknown staged effect"):
            self.runtime.checkpoint([effect, {"op": "invalid_effect"}])
        self.assertEqual(self.runtime.conversations.participant("guest")["contact_state"], "pending")
        self.assertEqual(self.input_events(), [])

    def test_pending_first_message_is_bounded_and_cannot_be_replaced_or_read_before_request(self):
        event_id = self.runtime.enqueue_conversation("chat-guest", "WITHHELD_PRIVATE_FIRST_MESSAGE", "guest:first")
        self.assertEqual(event_id, self.runtime.enqueue_conversation("chat-guest", "WITHHELD_PRIVATE_FIRST_MESSAGE", "guest:first"))
        self.runtime.register_conversation("guest", "Renamed", "second-chat")
        for chat, content in (("chat-guest", "Edited"), ("second-chat", "Another chat")):
            with self.assertRaisesRegex(ValueError, "additional messages"):
                self.runtime.enqueue_conversation(chat, content, "guest:second")
        result, effect = self.runtime._plan_action({"op": "contact_decide", "participant_id": "guest", "expected_request_revision": 1, "decision": "accept"}, [])
        self.assertFalse(result["ok"])
        self.assertIsNone(effect)
        self.assertEqual(self.input_events(), [])

    def test_old_multi_user_checkpoint_cannot_silently_gain_an_untaught_consent_contract(self):
        saved = self.runtime.store.latest() / "manifest.json"
        manifest = json.loads(saved.read_text())
        del manifest["fingerprint"]["config"]["require_contact_consent"]
        saved.write_text(json.dumps(manifest))
        self.runtime.close()
        for policy in ("strict", "fallback", "rebuild"):
            with self.subTest(policy=policy), self.assertRaisesRegex(ValueError, "no saved consent contract"):
                Runtime(self.root, self.config, DemoBackend(self.config, b"Private fixture. "), kv_recovery=policy)


if __name__ == "__main__":
    unittest.main()
