import json
import unittest
import urllib.request
from unittest import mock

from dmn.deep_sleep import run_fixture_sleep
from dmn.runtime import Runtime
from dmn.server import serve
from dmn.sleep_service import SleepService
from dmn.storage import InstanceLock
from tests import test_deep_sleep as fixtures


class SleepServiceTests(unittest.TestCase):
    def setUp(self):
        self.case = fixtures.SleepTest()
        self.case.setUp()
        self.addCleanup(self.case.tearDown)

    def prepare(self):
        c = self.case
        revision = c.compile()
        c.review(revision)
        c.generate({'op': 'learning_execution_decide', 'revision': revision, 'decision': 'approve'})
        c.generate({'op': 'deep_sleep', 'revision': revision})

    def test_http_queue_and_instance_lock_survive_backend_replacement(self):
        c = self.case
        self.prepare()
        initial_cursor = c.r.state['event_cursor']
        queued = []
        def transition(root, run_id, **kwargs):
            self.assertIsNone(c.r.backend)
            with self.assertRaisesRegex(RuntimeError, 'already open'):
                InstanceLock(root)
            request = urllib.request.Request(f'http://127.0.0.1:{server.server_port}/api/events',
                data=json.dumps({'content': 'Queued during training.'}).encode(),
                headers={'Content-Type': 'application/json', 'X-DMN-Request': '1'})
            with urllib.request.urlopen(request) as response:
                queued.append(json.load(response)['event_id'])
            self.assertFalse(service.stopped.is_set())
            self.assertEqual(service.status()['sleep_service']['phase'], 'training_or_rebuilding')
            return run_fixture_sleep(root, run_id, executor=c.executor, **kwargs)
        def factory(root, config, **kwargs):
            fresh = Runtime(root, config, c.factory(config), **kwargs)
            self.assertIs(fresh.store, c.r.store)
            self.assertIs(fresh._control_lock, c.r._control_lock)
            self.assertIs(fresh.ephemeral_images, c.r.ephemeral_images)
            self.assertEqual(fresh.state['event_cursor'], initial_cursor)
            self.assertEqual(fresh.store.next_event(initial_cursor)['id'], queued[0])
            fresh.run = mock.Mock()  # No autonomous demo generation in this mechanics test.
            c.r = fresh  # Cleanup now owns the restored runtime.
            return fresh
        service = SleepService(c.r, transition=transition, runtime_factory=factory)
        server = serve(service, 0)
        try:
            service.run()
            self.assertTrue(service.stopped.is_set())
            self.assertEqual(c.executor.candidates, 1)
            self.assertEqual(c.executor.wakes, 1)
            c.r.run.assert_called_once()
            self.assertEqual(len(queued), 1)
        finally:
            server.shutdown()
            server.server_close()

    def test_accepted_maintenance_does_not_restart(self):
        c = self.case
        c.r.state['mode'] = 'suspended'
        c.r.exit_requested.set()
        transition, factory = mock.Mock(), mock.Mock()
        service = SleepService(c.r, transition=transition, runtime_factory=factory)
        service.run()
        transition.assert_not_called()
        factory.assert_not_called()

    def test_stopped_failure_policy_does_not_restart_or_retrain(self):
        c = self.case
        self.prepare()
        transition = mock.Mock(return_value={'phase': 'Stopped'})
        factory = mock.Mock()
        service = SleepService(c.r, transition=transition, runtime_factory=factory)
        service.run()
        transition.assert_called_once()
        factory.assert_not_called()
        self.assertEqual(service.status()['sleep_service']['phase'], 'Stopped')

    def test_live_gate_is_not_bypassed_by_service_wrapper(self):
        self.case.r.sleep_test_mode = False
        with self.assertRaisesRegex(ValueError, 'test gate'):
            SleepService(self.case.r)

    def test_two_separately_reviewed_cycles_preserve_identity_and_one_lock(self):
        c = self.case
        self.prepare()
        instance_id, lock = c.r.state['instance_id'], c.r.lock
        restores = []
        def transition(root, run_id, **kwargs):
            return run_fixture_sleep(root, run_id, executor=c.executor, **kwargs)
        def factory(root, config, **kwargs):
            fresh = Runtime(root, config, c.factory(config), **kwargs)
            self.assertEqual(fresh.state['instance_id'], instance_id)
            self.assertIs(fresh.lock, lock)
            self.assertEqual(fresh._learning_reads, {})
            self.assertFalse(fresh.exit_requested.is_set())
            c.r = fresh
            restores.append(fresh)
            if len(restores) == 1:
                def second_request():
                    c.plan['examples'][0]['target'] = 'A different second example'
                    self.prepare()
                fresh.run = second_request
            else:
                fresh.run = lambda: fresh.state.update(mode='suspended')
            return fresh
        service = SleepService(c.r, transition=transition, runtime_factory=factory)
        service.run()
        self.assertEqual(len(restores), 2)
        self.assertEqual(c.executor.candidates, 2)
        self.assertEqual(c.r.store.db.execute("SELECT COUNT(*) FROM sleep_runs WHERE phase='WakeCommitted'").fetchone()[0], 2)


if __name__ == '__main__':
    unittest.main()
