"""Synthetic integration checks for vision, idle admission and deferred sleep."""
import dataclasses
import importlib.util
import json
import unittest

from dmn.backend import DemoBackend
from dmn.runtime import Runtime
from tests.test_activity import ActivityFixture
from tests.test_attachments import FixtureVision, upload
from tests.test_runtime import frames


class VisionActivityTest(ActivityFixture):
    def setUp(self):
        super().setUp()
        self.vision_enabled = False
        self.config = dataclasses.replace(self.config, clock_interval_seconds=0,
                                         sleep_checkpoint_min_interval_seconds=30)

    def create(self, script, config=None):
        config = config or self.config
        backend = DemoBackend(config, script)
        if self.vision_enabled:
            backend.vision = FixtureVision(backend)
        r = Runtime(self.root, config, backend, now=self.wall, monotonic=self.mono)
        self.opened.append(r)
        return r

    def test_new_vision_notice_waits_for_recovered_sleep_to_end(self):
        script = frames({'op': 'sleep'}) + b'awake'
        r = self.create(script)
        for _ in range(len(script)):
            r.tick()
            if r.state['mode'] == 'sleeping':
                break
        self.assertIsNotNone(r.store.activity_intent())
        saved = json.loads((r.store.latest() / 'engine.json').read_text())
        self.vision_enabled = True
        r = self.reopen(r, script)
        retained = r.backend.tokens.copy()
        self.assertIn('pending_restore', r.state)
        self.assertNotIn('image_protocol', r.state)
        self.assertFalse(r.tick())
        self.assertEqual(r.backend.tokens, retained)
        self.assertEqual(retained, saved['tokens'])
        self.assertEqual(r.backend.decoded_tokens, saved['decoded_tokens'])
        r.enqueue('wake event')
        r.tick()
        self.assertEqual(r.state['image_protocol'], 'ephemeral_images_v1')
        self.assertNotIn('pending_restore', r.state)

    @unittest.skipUnless(importlib.util.find_spec('PIL'), 'image input requires Pillow')
    def test_image_resumes_idle_without_splitting_addressed_action(self):
        self.vision_enabled = True
        self.config = dataclasses.replace(self.config, multi_user=True,
            operator_participant_id='alice', require_contact_consent=False)
        script = frames({'op': 'activity', 'mode': 'idle'},
            {'op': 'send_message', 'conversation_id': 'chat-a', 'content': 'intact'}) + b'quiet'
        r = self.create(script)
        r.register_conversation('alice', 'Alice', 'chat-a')
        r.register_conversation('bob', 'Bob', 'chat-b')
        result, effect = r._plan_action({'op': 'image_permission', 'scope': 'global',
                                        'decision': 'allow', 'accept_ephemeral': True}, [])
        self.assertTrue(result['ok'])
        r.checkpoint([effect])
        for _ in range(len(script)):
            r.tick()
            if r.pacer.profile['mode'] == 'idle':
                break
        for _ in range(4):
            r.tick()
        self.assertTrue(r.parser.pending)
        self.assertFalse(r.tick())
        event = r.enqueue_images('synthetic image', [upload()], conversation_id='chat-b')
        r.tick()
        self.assertEqual(r.pacer.profile['mode'], 'focus')
        self.assertEqual(r.backend.vision.delivered, 0)
        self.assertFalse(r.event_delivered(event))
        for _ in range(len(script)):
            r.tick()
            if r.event_delivered(event):
                break
        self.assertTrue(r.event_delivered(event))
        self.assertEqual(r.backend.vision.delivered, 1)
        self.assertEqual(r.store.messages()[0]['content'], 'intact')


if __name__ == '__main__':
    unittest.main()
