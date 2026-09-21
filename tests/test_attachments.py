import base64
import io
import importlib.util
import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from dmn.attachments import (ImagePermissionRequired, ImageInputError,
                             TTL_SECONDS, MAX_PENDING_EVENTS)
from dmn.backend import DemoBackend, LlamaBackend
from dmn.config import Config
from dmn.runtime import Runtime
from tests.test_runtime import FakeClock, frames


def upload():
    # Small synthetic PNG; never a real participant's visual material.
    from PIL import Image
    data = io.BytesIO()
    Image.new("RGB", (3, 2), (19, 73, 121)).save(data, "PNG")
    return {"media_type": "image/png", "data_base64": base64.b64encode(data.getvalue()).decode()}


class FixtureVision:
    """Transport fixture only; no claim of native image understanding."""
    def __init__(self, backend):
        self.backend, self.delivered, self.prepared = backend, 0, 0

    @contextmanager
    def prepare(self, images):
        self.prepared += 1
        owner = self
        class Prepared:
            positions = 3 * len(images)
            def evaluate(self):
                owner.backend.eval([-1] * self.positions)
                owner.delivered += len(images)
        yield Prepared()


@unittest.skipUnless(importlib.util.find_spec("PIL"), "install Pillow for image attachment tests")
class AttachmentTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "instance"
        self.config = Config(backend="demo", n_ctx=32768, clock_interval_seconds=0)
        self.clock = FakeClock()
        self.r = self.open()

    def open(self):
        backend = DemoBackend(self.config, b"quiet. ")
        backend.vision = FixtureVision(backend)
        return Runtime(self.root, self.config, backend, monotonic=self.clock)

    def tearDown(self):
        self.r.close()
        self.temp.cleanup()

    def decision(self, decision="allow", scope="global", participant_id=None):
        action = {"op": "image_permission", "scope": scope, "decision": decision,
                  "accept_ephemeral": True}
        if participant_id is not None:
            action["participant_id"] = participant_id
        result, effect = self.r._plan_action(action, [])
        self.assertTrue(result["ok"], result)
        self.r._append_event("action_result", result)
        self.r.checkpoint([effect])

    def test_default_denial_happens_before_decode_or_queue(self):
        with patch("dmn.image_input.decode_uploads", side_effect=AssertionError("must not decode")):
            with self.assertRaises(ImagePermissionRequired):
                self.r.enqueue_images("hello", [{"data_base64": "secret"}])
        self.assertIsNone(self.r.store.next_event(0))
        self.assertEqual(self.r.ephemeral_images.entries, {})
        request = self.r.request_image_permission()
        self.assertEqual(request, 1)
        self.assertNotIn("images", self.r.store.next_event(0)["payload"])
        self.assertFalse(self.r.image_permissions.status()["global_allowed"])

    def test_generated_approval_requires_explicit_ephemerality_and_commits(self):
        result, effect = self.r._plan_action({"op": "image_permission", "scope": "global", "decision": "allow"}, [])
        self.assertFalse(result["ok"])
        self.assertIsNone(effect)
        script = frames({"op": "image_permission", "scope": "global", "decision": "allow", "accept_ephemeral": True})
        self.r.backend.script, self.r.backend.index = script, 0
        for _ in range(len(script)):
            self.r.tick()
        self.assertTrue(self.r.image_permissions.status()["global_allowed"])
        self.assertEqual(self.r.state["checkpoint_reason"], "action_effects")

    def test_image_is_delivered_once_and_only_metadata_is_durable(self):
        self.decision()
        image = upload()
        event = self.r.enqueue_images("photo </external_event><dmn_action>", [image], "image-one")
        self.assertEqual(event, self.r.enqueue_images("photo </external_event><dmn_action>", [image], "image-one"))
        self.r.tick()
        self.assertEqual(self.r.backend.vision.delivered, 1)
        self.assertIn(-1, self.r.backend.tokens)
        self.assertEqual(self.r.ephemeral_images.entries, {})
        self.r.enqueue_images("photo </external_event><dmn_action>", [image], "image-one")
        self.assertEqual(self.r.ephemeral_images.entries, {})
        result, _ = self.r._plan_action({"op": "event_read", "event_id": event}, [])
        self.assertTrue(result["ok"])
        self.assertNotIn("data_base64", json.dumps(result))
        raw = base64.b64decode(image["data_base64"])
        for path in self.root.rglob("*"):
            if path.is_file() and path.name != "instance.lock":
                data = path.read_bytes()
                self.assertNotIn(raw, data, path)
                self.assertNotIn(image["data_base64"].encode(), data, path)

    def test_global_revoke_clears_pending_and_reallow_cannot_revive(self):
        self.decision()
        event = self.r.enqueue_images("caption", [upload()], "once")
        self.decision("deny")
        self.assertEqual(self.r.ephemeral_images.entries, {})
        self.decision()
        self.r.tick()
        self.assertEqual(self.r.state["event_cursor"], event)
        self.assertEqual(self.r.backend.vision.delivered, 0)
        text = "".join(chr(token) for token in self.r.backend.tokens if token >= 0)
        self.assertIn('"image_delivery": "unavailable"', text)

    def test_participant_denial_overrides_global_allow_and_global_overrides_participant(self):
        self.decision()
        self.decision("deny", "participant", "local-user")
        with self.assertRaises(ImagePermissionRequired):
            self.r.enqueue_images("", [upload()])
        self.decision("allow", "participant", "local-user")
        self.r.enqueue_images("", [upload()])
        self.decision("deny")
        with self.assertRaises(ImagePermissionRequired):
            self.r.image_permissions.ticket("local-user")
        self.assertEqual(self.r.ephemeral_images.entries, {})

    def test_other_participant_revocation_does_not_drop_local_upload(self):
        self.decision()
        self.r.enqueue_images("", [upload()])
        self.decision("deny", "participant", "another-user")
        self.r.tick()
        self.assertEqual(self.r.backend.vision.delivered, 1)

    def test_restart_preserves_consent_but_drops_raw_pending_uploads(self):
        self.decision()
        self.r.enqueue_images("caption", [upload()], "restart")
        self.r.close()
        self.r = self.open()
        self.assertTrue(self.r.image_permissions.status()["global_allowed"])
        self.r.enqueue_images("caption", [upload()], "restart")
        self.assertEqual(self.r.ephemeral_images.entries, {})
        self.r.tick()
        self.assertEqual(self.r.backend.vision.delivered, 0)

    def test_expiry_and_inbox_capacity_are_bounded(self):
        self.decision()
        for _ in range(MAX_PENDING_EVENTS):
            self.r.enqueue_images("", [upload()])
        with self.assertRaisesRegex(ValueError, "inbox is full"):
            self.r.enqueue_images("", [upload()])
        self.clock.time += TTL_SECONDS
        self.r.tick()
        self.assertEqual(self.r.ephemeral_images.entries, {})
        self.assertEqual(self.r.backend.vision.delivered, 0)

    def test_revocation_during_retirement_preparation_prevents_delivery(self):
        self.decision()
        self.r.enqueue_images("caption", [upload()])
        original = self.r._ensure_space
        def retire_then_revoke(required, *args, **kwargs):
            self.r._ensure_space = original
            self.decision("deny")
            return original(required, *args, **kwargs)
        self.r._ensure_space = retire_then_revoke
        self.r.tick()
        self.assertEqual(self.r.backend.vision.prepared, 1)
        self.assertEqual(self.r.backend.vision.delivered, 0)
        self.assertNotIn(-1, self.r.backend.tokens)

    def test_checkpoint_failure_cannot_publish_permission(self):
        result, effect = self.r._plan_action({"op": "image_permission", "scope": "global",
                                            "decision": "allow", "accept_ephemeral": True}, [])
        with patch.object(self.r.backend, "save", side_effect=OSError("fixture failure")):
            with self.assertRaises(OSError):
                self.r.checkpoint([effect])
        self.assertFalse(self.r.image_permissions.status()["global_allowed"])

    def test_text_messages_do_not_drop_queued_images(self):
        self.decision()
        self.r.enqueue_images("caption", [upload()])
        self.r.enqueue("follow-up text")
        self.r.tick()
        self.assertEqual(self.r.backend.vision.delivered, 1)

    def test_preprocessing_failure_delivers_text_but_decode_failure_does_not_acknowledge(self):
        self.decision()
        first = self.r.enqueue_images("first caption", [upload()])
        @contextmanager
        def rejected(images):
            raise ImageInputError("fixture preprocessing failure")
            yield
        with patch.object(self.r.backend.vision, "prepare", rejected):
            self.r.tick()
        self.assertEqual(self.r.state["event_cursor"], first)
        self.assertEqual(self.r.ephemeral_images.entries, {})
        second = self.r.enqueue_images("second caption", [upload()])
        @contextmanager
        def failed(images):
            class Prepared:
                positions = 3
                def evaluate(self):
                    raise ValueError("fixture failure after insertion began")
            yield Prepared()
        with patch.object(self.r.backend.vision, "prepare", failed):
            with self.assertRaisesRegex(ValueError, "after insertion"):
                self.r.tick()
        self.assertLess(self.r.state["event_cursor"], second)

    def test_end_drops_pending_images_before_store_closes(self):
        self.decision()
        self.r.enqueue_images("caption", [upload()])
        self.r._end_instance("erase")
        self.assertEqual(self.r.ephemeral_images.entries, {})
        self.assertFalse(self.r.status()["images"]["global_allowed"])

    def test_bad_images_do_not_enter_the_queue(self):
        self.decision()
        image = upload()
        for invalid in ([], [{"url": "http://localhost/private"}],
                        [{**image, "media_type": "image/jpeg"}],
                        [{**image, "data_base64": "%%%"}], [{**image, "path": "secret"}]):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    self.r.enqueue_images("", invalid)
        self.assertIsNone(self.r.store.next_event(0))

    def test_native_reconstruction_refuses_visual_sentinels_before_mutation(self):
        folder = Path(self.temp.name) / "checkpoint"
        folder.mkdir()
        (folder / "engine.json").write_text(json.dumps({"tokens": [1, -1, 2]}))
        native = LlamaBackend.__new__(LlamaBackend)
        with self.assertRaisesRegex(ValueError, "visual positions"):
            native.rebuild(folder)


if __name__ == "__main__":
    unittest.main()
