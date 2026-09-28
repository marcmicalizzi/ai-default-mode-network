import copy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from dmn.gpu_recipe import PACKAGES, compile_training, validate_recipe
from dmn.training import GPU_KIND, GPU_KIND_V2, CHECKS
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

    def test_v2_compiles_full_1536_example_with_explicit_chunked_loss(self):
        from dmn.sleep_plans import implementation_identity, execution_status
        self.recipe['kind'] = GPU_KIND_V2
        self.recipe['trainer']['gpu']['max_sequence_tokens'] = 1536
        c = self.case
        c.recipe_id = c.r.offer_learning_recipe(self.recipe)['revision']
        c.plan['examples'][0].update(input='Prefix: ', target='x' * 1528)
        value = self.compiled()
        self.assertEqual(len(value['examples'][0]['tokens']), 1536)
        self.assertEqual(value['examples'][0]['labels'][:8], [-100] * 8)
        self.assertEqual(value['training']['loss_computation'],
                         {'implementation': 'checkpointed_vocabulary_chunks_v1', 'output_chunk_tokens': 64})
        self.assertIn('chunked_loss.py', implementation_identity())
        self.assertNotEqual(execution_status(c.r.store, value['revision']), 'approved')
        oversized = copy.deepcopy(value)
        oversized['examples'][0]['tokens'].append(1)
        before = copy.deepcopy(oversized['examples'])
        with self.assertRaisesRegex(ValueError, 'no truncation or splitting'):
            compile_training(oversized)
        self.assertEqual(oversized['examples'], before)
        self.recipe['trainer']['gpu']['max_sequence_tokens'] = 1537
        with self.assertRaisesRegex(ValueError, 'envelope'):
            validate_recipe(self.recipe)

    def test_v2_lower_host_cap_is_enforced_and_versions_cannot_substitute(self):
        from dmn.config import Config
        from dmn.sleep_host import guard
        from dmn.gpu_training_worker import recipe_loss, loss_for
        from dmn.chunked_loss import loss_for as chunked
        value = self.compiled()
        self.assertIs(recipe_loss(value), loss_for)
        value['recipe']['kind'] = GPU_KIND_V2
        self.assertIs(recipe_loss(value), chunked)
        config = Config(backend='llama', n_ctx=4096, experimental_compact_swa=True,
                        pack_checkpoints=True, swa_full=False, flash_attn=True,
                        type_k='q8_0', type_v='q8_0')
        with mock.patch('dmn.sleep_host.os', SimpleNamespace(name='nt')), self.assertRaisesRegex(ValueError, 'recipe version'):
            guard(config, value, self.recipe)
        value['recipe']['trainer']['gpu']['max_sequence_tokens'] = 1024
        value['examples'][0]['tokens'] = [1] * 1025
        with self.assertRaisesRegex(ValueError, 'offered NF4 length'):
            compile_training(value)

    def test_worker_rejects_mismatched_reviewed_loss_before_loading(self):
        from dmn.gpu_training_worker import request
        from dmn.sleep_plans import seal
        value = self.compiled()
        value['training']['loss_computation'] = {'implementation': 'wrong', 'output_chunk_tokens': 64}
        value = seal({k: v for k, v in value.items() if k != 'revision'})
        write_durable(self.root / 'input.json', value)
        with self.assertRaisesRegex(ValueError, 'compiled loss differs'):
            request(self.root)


class GpuWorkerGateTest(unittest.TestCase):
    def test_uncontained_worker_refuses_before_importing_torch(self):
        from dmn.gpu_training_worker import load_model
        with mock.patch.dict('os.environ', {'DMN_GPU_PROBE_CONTAINED': '', 'CUDA_VISIBLE_DEVICES': '-1'}):
            with mock.patch.dict('sys.modules', {'torch': None}), self.assertRaisesRegex(ValueError, 'contained'):
                load_model({}, Path('.'), training=True)
