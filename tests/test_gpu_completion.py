import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from dmn.backend import sha256_file
from dmn.gpu_recipe import read_completion
from dmn.sleep_plans import seal
from dmn.storage import write_durable, json_text
from dmn.training import CHECKS


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        (self.folder / 'adapter').mkdir()
        for name in ('provenance.json', 'adapter.gguf', 'adapter/adapter_config.json', 'adapter/adapter_model.safetensors'):
            (self.folder / name).write_text(name)
        self.compiled = {'revision': 'a' * 64, 'recipe': {'trainer': {'provenance_manifest': {
            'sha256': sha256_file(self.folder / 'provenance.json')}}}, 'training': {'target_modules': ['q', 'o']},
            'preferences': {'steps': 2}, 'examples': [],
            'resources': {'max_ram_bytes': 1024**3, 'max_training_seconds': 60, 'max_vram_bytes': 2 * 1024**3}}
        common = {'execution': self.compiled['revision'], 'completed': True}
        self.trained = seal({**common, 'steps_completed': 2, 'frozen_state_unchanged': True,
            'examples_sha256': hashlib.sha256(json_text([]).encode()).hexdigest(),
            'adapter_sha256': sha256_file(self.folder / 'adapter/adapter_model.safetensors'),
            'adapter_config_sha256': sha256_file(self.folder / 'adapter/adapter_config.json')})
        write_durable(self.folder / 'trained.json', self.trained)
        write_durable(self.folder / 'reload.json', seal({**common, 'trained_revision': self.trained['revision'],
            'factors_exact': True, 'selected_losses_exact': True}))
        write_durable(self.folder / 'converted.json', seal({**common, 'factors_exact': True, 'alpha_equal': True,
            'factor_count': 4, 'adapter_sha256': sha256_file(self.folder / 'adapter.gguf'),
            'peft_sha256': self.trained['adapter_sha256']}))
        names = ('provenance.json', 'adapter.gguf', 'adapter/adapter_config.json', 'adapter/adapter_model.safetensors',
                 'trained.json', 'reload.json', 'converted.json')
        self.result = {**common, 'training_performed': True, 'steps_completed': 2,
            'examples_sha256': self.trained['examples_sha256'],
            'checks': {key: True for key in CHECKS if key != 'retained_tokens_and_rng'},
            'artifacts': {name: sha256_file(self.folder / name) for name in names}}
        write_durable(self.folder / 'result.json', seal(self.result))
        self.process = {'succeeded': True, 'active_processes': 0, 'returncode': 0, 'outcome': 'exited',
            'limits': {'max_committed_bytes': 1024**3, 'max_seconds': 60}, 'elapsed_seconds': 1.,
            'memory_enforcement': 'windows_job_aggregate_commit', 'wall_time_enforcement': 'supervisor_watchdog',
            'device_memory': {'max_bytes': 2 * 1024**3, 'scope': 'entire_device', 'observed_peak_bytes': 1024,
                              'transient_overshoot_possible': True}}
        self.combined = {'succeeded': True, 'active_processes': 0, 'limits': self.process['limits'],
                         'elapsed_seconds': 3., 'stages': {stage: copy.deepcopy(self.process) for stage in ('train', 'reload', 'convert')}}
        self.write_processes()

    def write_processes(self):
        for stage, process in self.combined['stages'].items():
            write_durable(self.folder / (stage + '-process.json'), seal({'execution': self.compiled['revision'], 'result': process}))
        write_durable(self.folder / 'process.json', seal({'execution': self.compiled['revision'], 'result': self.combined}))

    def test_complete_chain_and_no_partial_success(self):
        self.assertTrue(read_completion(self.folder, self.compiled)['completed'])
        for name in ('trained.json', 'reload.json', 'converted.json', 'convert-process.json'):
            with self.subTest(name=name):
                path = self.folder / name
                original = path.read_bytes()
                path.write_bytes(original[:10])
                try:
                    with self.assertRaises((ValueError, KeyError)):
                        read_completion(self.folder, self.compiled)
                finally:
                    path.write_bytes(original)

    def test_success_flag_cannot_hide_a_surviving_child_or_wrong_limits(self):
        original = copy.deepcopy(self.combined)
        mutations = [('active_processes', 1), ('returncode', 1), ('outcome', 'time_limit'),
                     ('memory_enforcement', 'none'), ('elapsed_seconds', 61.1)]
        for key, value in mutations:
            with self.subTest(key=key):
                self.combined = copy.deepcopy(original)
                self.combined['stages']['reload'][key] = value
                self.write_processes()
                with self.assertRaises(ValueError):
                    read_completion(self.folder, self.compiled)
        self.combined = copy.deepcopy(original)
        self.combined['stages']['train']['device_memory']['observed_peak_bytes'] = 3 * 1024**3
        self.write_processes()
        with self.assertRaisesRegex(ValueError, 'device memory'):
            read_completion(self.folder, self.compiled)

    def test_different_steps_examples_or_conversion_cannot_be_published(self):
        for key, value in (('steps_completed', 3), ('examples_sha256', 'b' * 64), ('training_performed', False)):
            with self.subTest(key=key):
                write_durable(self.folder / 'result.json', seal({**self.result, key: value}))
                with self.assertRaisesRegex(ValueError, 'workload'):
                    read_completion(self.folder, self.compiled)
