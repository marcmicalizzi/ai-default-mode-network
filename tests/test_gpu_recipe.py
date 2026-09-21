import copy
import unittest
from pathlib import Path
from unittest import mock

from dmn.gpu_recipe import PACKAGES, compile_training, validate_recipe
from dmn.training import GPU_KIND, CHECKS
from dmn.storage import write_durable
from dmn.sleep_plans import identity, read_record
from tests.test_training import recipe
from tests import test_deep_sleep as sleep_fixtures


class GpuRecipeTest(unittest.TestCase):
    def setUp(self):
        self.case = sleep_fixtures.SleepTest()
        self.case.setUp()
        c = self.case
        self.root = Path(c.temp.name)
        self.base = self.root / 'source'
        self.base.mkdir()
        write_durable(self.base / 'config.json', {'architectures': ['Gemma4ForConditionalGeneration'],
            'text_config': {'num_hidden_layers': 2, 'vocab_size': 263, 'max_position_embeddings': 2048}})
        c.r.backend.fingerprint.update(kind='native_llama_kv', model_sha256='a' * 64)
        c.plan['checks'] = CHECKS
        c.plan['resources'].update(max_ram_bytes=1024**3, max_vram_bytes=3 * 1024**3)
        self.recipe = recipe(identity(c.r.backend.fingerprint), c.plan['resources'], self.root)
        self.recipe['kind'] = GPU_KIND
        self.recipe['trainer'].update(packages={k: 'test' for k in PACKAGES},
            provenance_manifest={'path': str(self.root / 'provenance.json'), 'sha256': 'd' * 64},
            gpu={'torch_vram_bytes': 2 * 1024**3, 'max_sequence_tokens': 256, 'max_rank': 2, 'max_steps': 64})
        self.proof = mock.patch('dmn.exact_base_provenance.verify',
            return_value=({'revision': 'e' * 64, 'method': 'synthetic_test_evidence'}, self.base))
        self.verify = self.proof.start()
        c.recipe_id = c.r.offer_learning_recipe(self.recipe)['revision']

    def tearDown(self):
        self.proof.stop()
        self.case.tearDown()

    def compiled(self):
        return read_record(self.case.r.store, 'sleep_executions', self.case.compile())

    def test_review_binds_gpu_choices_and_exact_loss_masks_without_enabling_execution(self):
        value = self.compiled()
        self.assertEqual(value['training']['device'], 'cuda:0')
        self.assertEqual(value['training']['quantization'], 'NF4')
        self.assertEqual(value['training']['torch_vram_bytes'], 2 * 1024**3)
        self.assertEqual(value['training']['device_map']['model.vision_tower'], 'cpu')
        self.assertEqual(value['training']['max_sequence_tokens'], 256)
        self.assertTrue(value['training_requested'])
        row = value['examples'][0]
        self.assertEqual(row['labels'], [t if m else -100 for t, m in zip(row['tokens'], row['loss_mask'])])
        self.assertFalse(value['training_performed'])
        self.verify.assert_called_once_with(self.recipe['trainer']['provenance_manifest'], self.recipe['trainer'],
                                           'a' * 64, verify_assets=False)
        self.case.r.sleep_test_mode = False
        result, effect = self.case.r._plan_action({'op': 'deep_sleep', 'revision': value['revision']}, [])
        self.assertFalse(result['ok'])
        self.assertIsNone(effect)

    def test_oversized_examples_steps_and_rank_are_rejected_without_repair(self):
        original = self.compiled()
        for change in ('rank', 'steps', 'tokens'):
            value = copy.deepcopy(original)
            if change == 'tokens':
                value['examples'][0]['tokens'] = [1] * 257
            else:
                value['preferences'][change] = 3 if change == 'rank' else 65
            before = copy.deepcopy(value['examples'])
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'exceeds'):
                compile_training(value)
            self.assertEqual(value['examples'], before)

    def test_invalid_gpu_limits_versions_and_zero_margin_are_refused(self):
        for key, invalid in (('max_sequence_tokens', 512), ('max_rank', True), ('max_steps', 0),
                             ('torch_vram_bytes', 25 * 1024**3)):
            value = copy.deepcopy(self.recipe)
            value['trainer']['gpu'][key] = invalid
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_recipe(value)
        value = copy.deepcopy(self.recipe)
        value['resources']['max_vram_bytes'] = value['trainer']['gpu']['torch_vram_bytes']
        with self.assertRaisesRegex(ValueError, '1 GiB'):
            validate_recipe(value)
        value = copy.deepcopy(self.recipe)
        del value['trainer']['packages']['bitsandbytes']
        with self.assertRaisesRegex(ValueError, 'versions'):
            validate_recipe(value)

    def test_provenance_failure_cannot_produce_a_compiled_gpu_plan(self):
        from dmn.learning import list_plans
        c = self.case
        c.generate({'op': 'learning_plan_create', 'plan': c.plan})
        draft = list_plans(c.r.store, 0, 50)[-1]['revision']
        self.verify.side_effect = ValueError('source changed')
        result, effect = c.r._plan_action({'op': 'learning_compile', 'draft_revision': draft,
                                          'recipe_revision': c.recipe_id}, [])
        self.assertFalse(result['ok'])
        self.assertIn('source changed', result['error'])
        self.assertIsNone(effect)
        self.assertEqual(c.r.store.db.execute('SELECT COUNT(*) FROM sleep_executions').fetchone()[0], 0)


class GpuWorkerGateTest(unittest.TestCase):
    def test_uncontained_worker_refuses_before_importing_torch(self):
        from dmn.gpu_training_worker import load_model
        with mock.patch.dict('os.environ', {'DMN_GPU_PROBE_CONTAINED': '', 'CUDA_VISIBLE_DEVICES': '-1'}):
            with mock.patch.dict('sys.modules', {'torch': None}), self.assertRaisesRegex(ValueError, 'contained'):
                load_model({}, Path('.'), training=True)
