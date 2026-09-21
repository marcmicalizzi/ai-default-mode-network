import json
import unittest
from unittest import mock

from dmn.deep_sleep import read_run
from dmn.sleep_plans import read_record, seal
from dmn.sleep_wake_executor import prepare_wake
from dmn.sleep_wake_worker import wake
from dmn.storage import Store, write_durable
from tests import test_deep_sleep as fixtures


class WakeReceiptTests(unittest.TestCase):
    def setUp(self):
        self.case = fixtures.SleepTest()
        self.case.setUp()
        self.addCleanup(self.case.tearDown)
        c = self.case
        self.run_id = c.request()
        store = Store(c.root)
        try:
            run = read_run(store, self.run_id)
            self.compiled = read_record(store, 'sleep_executions', run['execution'])
        finally:
            store.close()
        self.folder = c.root / 'sleep' / self.run_id
        self.folder.mkdir(parents=True)
        self.report = {'run_id': self.run_id, 'execution': self.compiled['revision'],
                       'outcome': 'candidate_adopted', 'elapsed_seconds': 2.}
        self.candidate = {'adapters': [], 'training_performed': False}

    def fake_process(self, *args, **kwargs):
        with mock.patch.dict('os.environ', {'DMN_GPU_PROBE_CONTAINED': '1', 'DMN_CPU_WORKER_CONTAINED': '1'}), \
                mock.patch('dmn.sleep_wake_worker.make_backend', self.case.factory):
            wake(self.folder, args[1][-1])
        return {'succeeded': True, 'active_processes': 0, 'returncode': 0, 'outcome': 'exited', 'log_error': None,
            'limits': {'max_committed_bytes': self.compiled['resources']['max_ram_bytes'],
                       'max_seconds': self.compiled['resources']['max_training_seconds']},
            'memory_enforcement': 'windows_job_aggregate_commit', 'wall_time_enforcement': 'supervisor_watchdog',
            'device_memory': {'max_bytes': self.compiled['resources']['max_vram_bytes'], 'scope': 'entire_device',
                              'observed_peak_bytes': 0}}

    def execute(self, report=None):
        return prepare_wake(self.folder, self.case.root, self.case.source, self.compiled,
                            self.candidate, True, report or self.report)

    def test_complete_wake_recovery_retains_first_report_without_repeating_work(self):
        with mock.patch('dmn.sleep_wake_executor.run_monitored_gpu_worker', side_effect=self.fake_process) as launch:
            first = self.execute()
            second = self.execute({**self.report, 'elapsed_seconds': 999.})
            self.assertEqual(first, second)
            launch.assert_called_once()

    def test_wrong_supervision_limits_cannot_publish_a_completed_checkpoint(self):
        with mock.patch('dmn.sleep_wake_executor.run_monitored_gpu_worker', side_effect=self.fake_process) as launch:
            self.execute()
            path = self.folder / 'wake-process.json'
            record = json.loads(path.read_text())
            record['result']['limits']['max_committed_bytes'] += 1
            write_durable(path, seal({k: v for k, v in record.items() if k != 'revision'}))
            with self.assertRaisesRegex(ValueError, 'worker failed'):
                self.execute()
            launch.assert_called_once()

    def test_source_identity_is_checked_before_loading_a_backend(self):
        with mock.patch('dmn.sleep_wake_executor.run_monitored_gpu_worker', side_effect=self.fake_process):
            self.execute()
        path = self.folder / 'wake-input.json'
        request = json.loads(path.read_text())
        changed = {**request['compiled'], 'instance_id': 'different instance'}
        request['compiled'] = seal({k: v for k, v in changed.items() if k != 'revision'})
        request['report']['execution'] = request['compiled']['revision']
        write_durable(path, seal({k: v for k, v in request.items() if k != 'revision'}))
        with mock.patch.dict('os.environ', {'DMN_GPU_PROBE_CONTAINED': '1'}), \
                mock.patch('dmn.sleep_wake_worker.make_backend') as factory:
            with self.assertRaisesRegex(ValueError, 'approved sleep boundary'):
                wake(self.folder)
            factory.assert_not_called()

    def test_changed_result_cannot_claim_a_different_wake(self):
        with mock.patch('dmn.sleep_wake_executor.run_monitored_gpu_worker', side_effect=self.fake_process) as launch:
            self.execute()
            path = self.folder / 'wake-result.json'
            result = json.loads(path.read_text())
            result['request'] = 'wrong request'
            write_durable(path, seal({k: v for k, v in result.items() if k != 'revision'}))
            with self.assertRaisesRegex(ValueError, 'completion differs'):
                self.execute()
            launch.assert_called_once()

    def test_candidate_failure_and_previous_wake_use_independent_recoverable_slots(self):
        with mock.patch('dmn.sleep_wake_executor.run_monitored_gpu_worker', side_effect=self.fake_process) as launch, \
                mock.patch('dmn.sleep_wake_executor.run_cpu_worker', side_effect=self.fake_process) as previous:
            adopted = self.execute()
            report = {**self.report, 'outcome': 'failed', 'reason': 'synthetic post-candidate failure'}
            old = prepare_wake(self.folder, self.case.root, self.case.source, self.compiled, None, False, report)
            self.assertNotEqual(adopted[0], old[0])
            self.assertTrue((self.folder / 'wake-input.json').exists())
            self.assertTrue((self.folder / 'previous-wake-input.json').exists())
            self.assertEqual(prepare_wake(self.folder, self.case.root, self.case.source,
                self.compiled, None, False, report), old)
            launch.assert_called_once()
            previous.assert_called_once()
