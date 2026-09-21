import tempfile
import unittest
from pathlib import Path
from unittest import mock
from dmn.worker_limits import WorkerLimits, run_monitored_gpu_worker


class MonitorTests(unittest.TestCase):
    def test_unavailable_or_overbudget_device_never_launches(self):
        limits = WorkerLimits(128 * 1024**2, 10)
        with mock.patch('dmn.gpu_monitor.DeviceMemory') as cls, mock.patch('dmn.worker_limits._run_worker') as launch:
            device = cls.return_value
            device.used.return_value = 1001
            with self.assertRaisesRegex(ValueError, 'already exceeds'):
                run_monitored_gpu_worker('python', [], cwd='.', log='not-created.log', limits=limits, max_device_bytes=1000)
            launch.assert_not_called()
            device.close.assert_called_once()
            cls.side_effect = OSError('no telemetry')
            with self.assertRaises(OSError):
                run_monitored_gpu_worker('python', [], cwd='.', log='not-created.log', limits=limits, max_device_bytes=1000)
            launch.assert_not_called()

    def test_monitor_tracks_whole_device_and_reports_a_sampled_limit(self):
        with mock.patch('dmn.gpu_monitor.DeviceMemory') as cls, mock.patch('dmn.worker_limits._run_worker') as launch:
            device = cls.return_value
            device.uuid = 'fixture'
            device.used.side_effect = [10, 900, 1001]
            def execute(*args, **kwargs):
                self.assertIsNone(kwargs['resource_guard']())
                self.assertEqual(kwargs['resource_guard'](), 'device_memory_limit')
                return {'succeeded': False, 'outcome': 'device_memory_limit'}
            launch.side_effect = execute
            result = run_monitored_gpu_worker('python', [], cwd='.', log='not-created.log',
                limits=WorkerLimits(128 * 1024**2, 10), max_device_bytes=1000)
            self.assertEqual(result['device_memory']['observed_peak_bytes'], 1001)
            self.assertEqual(result['device_memory']['scope'], 'entire_device')
            self.assertTrue(result['device_memory']['transient_overshoot_possible'])
            device.close.assert_called_once()
