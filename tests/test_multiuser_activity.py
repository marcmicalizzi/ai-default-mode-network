"""Cross-branch admission and sleep recovery checks; scripted instances only."""
import dataclasses
import unittest

from tests import test_conversations as conversation_fixtures

frame = conversation_fixtures.frame


class MultiuserActivityTest(unittest.TestCase):
    def setUp(self):
        self.case = conversation_fixtures.ConversationTest()
        self.case.setUp()
        self.case.config = dataclasses.replace(self.case.config, idle_enabled=True,
            idle_max_burst_tokens=4, idle_min_interval_seconds=120,
            sleep_checkpoint_min_interval_seconds=300)

    def tearDown(self):
        self.case.tearDown()

    def test_suppressed_input_cannot_wake_recovered_deferred_sleep(self):
        c = self.case
        script = frame(op='sleep')
        r = c.create(script)
        c.register(r)
        blocked = r.enqueue_conversation('chat-b', 'suppressed input')
        c.commit_action(r, op='block_participant', participant_id='bob')
        c.until(r, lambda: r.state['mode'] == 'sleeping')
        self.assertIsNotNone(r.store.activity_intent())
        r = c.reopen(r, script)
        self.assertFalse(r.tick())
        self.assertEqual(r.state['mode'], 'sleeping')
        self.assertFalse(r.event_delivered(blocked))
        wake = r.enqueue_conversation('chat-a', 'eligible new input')
        r.tick()
        self.assertEqual(r.state['mode'], 'active')
        self.assertEqual(r.state['event_cursor'], wake)
        self.assertFalse(r.event_delivered(blocked))

    def test_new_input_preempts_idle_then_fair_admission_resumes(self):
        c = self.case
        c.config = dataclasses.replace(c.config, inbox_generation_tokens=256)
        r = c.create(frame(op='activity', mode='idle') + b'abcdefghijk')
        c.register(r)
        r.enqueue_conversation('chat-a', 'initial message')
        r.tick()
        c.until(r, lambda: r.pacer.profile['mode'] == 'idle')
        for _ in range(4):
            r.tick()
        self.assertFalse(r.tick())
        self.assertLess(r.state['generated_tokens'], r.state['inbox_next_generated'])
        wake = r.enqueue_conversation('chat-b', 'wake idle')
        r.tick()
        self.assertEqual(r.state['event_cursor'], wake)
        self.assertEqual(r.pacer.profile['mode'], 'focus')
        later = r.enqueue_conversation('chat-a', 'wait for generation opportunity')
        r.tick()
        self.assertFalse(r.event_delivered(later))

    def test_idle_input_resumes_but_does_not_cancel_a_partial_action(self):
        c = self.case
        script = frame(op='activity', mode='idle') + frame(op='send_message', conversation_id='chat-a', content='intact')
        r = c.create(script)
        c.register(r)
        c.until(r, lambda: r.pacer.profile['mode'] == 'idle')
        for _ in range(4):
            r.tick()
        self.assertTrue(r.parser.pending)
        self.assertFalse(r.tick())
        event = r.enqueue_conversation('chat-b', 'wait for complete action')
        r.tick()
        self.assertEqual(r.pacer.profile['mode'], 'focus')
        self.assertFalse(r.event_delivered(event))
        c.until(r, lambda: bool(r.store.messages()))
        self.assertEqual(r.store.messages()[0]['content'], 'intact')
        c.until(r, lambda: r.event_delivered(event))


if __name__ == '__main__':
    unittest.main()
