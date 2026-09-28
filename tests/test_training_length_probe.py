import contextlib
import io
from pathlib import Path
import unittest
from unittest import mock

from scripts import probe_training_length as probe


class TrainingLengthProbeTests(unittest.TestCase):
    def test_length_and_allocator_bounds_are_explicit(self):
        with mock.patch.object(probe, 'inspect', return_value={'synthetic_text_only': True}) as inspect:
            value = probe.plan(Path('synthetic-source'), 512, 23552)
            inspect.assert_called_once_with(Path('synthetic-source'), sequence_tokens=512, steps=2)
        self.assertEqual(value['masked_prefix_tokens'], 128)
        self.assertEqual(value['loss'], 'production_shifted_target_only_v1')
        self.assertFalse(value['adoption_authorized'])
        for length, budget in ((128,22528),(4096,22528),(512,24577),(512,True)):
            with self.assertRaises(ValueError):
                probe.plan(Path('synthetic-source'), length, budget)

    def test_chunked_comparison_is_explicit_and_bounded(self):
        with mock.patch.object(probe, 'inspect', return_value={}):
            value = probe.plan(Path('source'), 2048, 24576, 'chunked')
            self.assertEqual(value['output_chunk_tokens'], 64)
            self.assertEqual(value['masked_prefix_tokens'], 512)
            self.assertFalse(value['compare_full_gradients'])
        for length, mode in ((1024, 'chunked'), (512, 'full')):
            with self.assertRaises(ValueError):
                probe.plan(Path('source'), length, 24576, mode, True)

    def test_default_inspection_never_dispatches_gpu_worker(self):
        args = ['probe', '--source', 'synthetic-source', '--tokens', '512']
        with mock.patch('sys.argv', args), mock.patch.object(probe, 'plan', return_value={'synthetic_only': True}), \
                mock.patch.object(probe, 'run_monitored_gpu_worker') as run, contextlib.redirect_stdout(io.StringIO()):
            probe.main()
        run.assert_not_called()

    def test_execution_requires_separate_interpreter_and_fresh_output(self):
        args = ['probe', '--source', 'synthetic-source', '--execute']
        with mock.patch('sys.argv', args), mock.patch.object(probe, 'plan', return_value={}), \
                mock.patch.object(probe, 'run_monitored_gpu_worker') as run, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                probe.main()
        run.assert_not_called()
