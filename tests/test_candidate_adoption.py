import copy
import dataclasses
import unittest
from types import SimpleNamespace
from unittest import mock

from dmn.candidate_adoption import prepare, recover
from dmn.deep_sleep import read_run
from dmn.runtime import Runtime
from dmn.sleep_host import guard
from dmn.sleep_plans import read_record, seal, execution_status
from dmn.storage import json_text
from tests import test_gpu_recipe


class CandidateAdoptionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_gpu_recipe.GpuRecipeTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        c = self.fixture.case
        c.plan['preferences']['adoption'] = 'review_first'
        self.old = self.fixture.compiled()
        self.candidate = {'adapters': [], 'training_performed': True, 'receipt': 'a' * 64}
        self.run_id = 'b' * 32
        run = {'schema': 1, 'id': self.run_id, 'execution': self.old['revision'],
               'report': {'outcome': 'review_candidate_under_original_weights'}, 'candidate': self.candidate}
        with c.r.store.transaction() as db:
            db.execute('INSERT INTO sleep_runs VALUES(?,?,?)', (self.run_id, 'WakeCommitted', json_text(run)))
            db.execute("UPDATE sleep_executions SET status='completed' WHERE revision=?", (self.old['revision'],))
        self.recovered = mock.patch('dmn.training_executor.TrainingExecutor.recover', return_value=self.candidate)
        self.recovered.start()
        self.addCleanup(self.recovered.stop)

    def test_adoption_requires_new_complete_review_and_preserves_current_context(self):
        c = self.fixture.case
        before = c.r.backend.tokens.copy()
        c.generate({'op': 'learning_candidate_prepare', 'run_id': self.run_id})
        revision = c.r.store.db.execute('SELECT revision FROM sleep_executions ORDER BY rowid DESC LIMIT 1').fetchone()[0]
        plan = read_record(c.r.store, 'sleep_executions', revision)
        self.assertNotEqual(revision, self.old['revision'])
        self.assertFalse(plan['training_requested'])
        self.assertEqual(plan['training']['additional_steps'], 0)
        self.assertEqual(plan['candidate_reuse']['run_id'], self.run_id)
        self.assertEqual(c.r.backend.tokens[:len(before)], before)
        c.generate({'op': 'learning_execution_decide', 'revision': revision, 'decision': 'approve'})
        self.assertEqual(execution_status(c.r.store, revision), 'awaiting_review')
        c.review(revision)
        c.generate({'op': 'learning_execution_decide', 'revision': revision, 'decision': 'approve'})
        self.assertEqual(execution_status(c.r.store, revision), 'approved')
        recovered = recover(c.root, c.r.store, plan)
        self.assertFalse(recovered['training_performed'])
        self.assertEqual(recovered['receipt'], self.candidate['receipt'])

    def test_withdrawn_draft_stale_parent_or_changed_receipt_cannot_adopt(self):
        c = self.fixture.case
        plan = prepare(c.r, self.run_id)
        changed = copy.deepcopy(plan)
        changed['candidate_reuse']['receipt'] = 'c' * 64
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            recover(c.root, c.r.store, changed)
        c.r.backend.fingerprint['model_sha256'] = 'd' * 64
        with self.assertRaisesRegex(ValueError, 'unchanged parent'):
            prepare(c.r, self.run_id)
        c.r.backend.fingerprint['model_sha256'] = 'a' * 64
        c.generate({'op': 'learning_plan_withdraw', 'revision': self.old['draft_revision']})
        with self.assertRaisesRegex(ValueError, 'withdrawn'):
            recover(c.root, c.r.store, plan)

    def test_host_caps_and_pinned_trainer_remain_independent_of_model_approval(self):
        from dmn.config import Config
        config = Config(model_path='unused', n_ctx=60000, type_k='q8_0', type_v='q8_0',
                        experimental_compact_swa=True, pack_checkpoints=True, swa_full=False, flash_attn=True)
        offer = self.fixture.recipe
        with mock.patch('dmn.sleep_host.os', SimpleNamespace(name='nt')):
            guard(config, self.old, offer)
            changed = copy.deepcopy(self.old)
            changed['resources']['max_ram_bytes'] += 1
            with self.assertRaisesRegex(ValueError, 'resource offer'):
                guard(config, changed, offer)
            changed = copy.deepcopy(self.old)
            changed['recipe']['trainer']['python_sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'trainer differs'):
                guard(config, changed, offer)
            with self.assertRaisesRegex(ValueError, 'context envelope'):
                guard(dataclasses.replace(config, n_ctx=65536), self.old, offer)

    def test_continuation_offer_keeps_trainer_and_follows_adoption_only_lineage(self):
        from dmn.adapters import AdapterSpec
        from dmn.sleep_host import offer_for_runtime
        from dmn.sleep_plans import identity
        c = self.fixture.case
        spec = AdapterSpec('synthetic.gguf', 'd' * 64, 'a' * 64, .1)
        c.r.config = dataclasses.replace(c.r.config, backend='llama', model_path='unused', lora_adapters=(spec,))
        c.r.backend.fingerprint['lora_adapters'] = [spec.identity()]
        parent = identity(c.r.backend.fingerprint)
        trained = read_run(c.r.store, self.run_id)
        trained['candidate']['adapters'] = [dataclasses.asdict(spec)]
        reused = seal({**{k:v for k,v in self.old.items() if k != 'revision'},
                       'candidate_reuse':{'run_id':self.run_id}})
        adopted = {'schema':1, 'id':'f' * 32, 'execution':reused['revision'],
                   'report':{'outcome':'candidate_adopted', 'new_weights':parent}}
        with c.r.store.transaction() as db:
            db.execute('UPDATE sleep_runs SET payload=? WHERE id=?', (json_text(trained), self.run_id))
            db.execute('INSERT INTO sleep_executions VALUES(?,?,?,?)',
                       (reused['revision'], json_text(reused), 'completed', 0))
            db.execute('INSERT INTO sleep_runs VALUES(?,?,?)',
                       (adopted['id'], 'WakeCommitted', json_text(adopted)))
        work = c.root / 'sleep' / self.run_id / 'worker'
        (work / 'adapter').mkdir(parents=True)
        (work / 'adapter/adapter_config.json').write_text('{}')
        (work / 'adapter/adapter_model.safetensors').write_bytes(b'synthetic factors')
        original = copy.deepcopy(self.fixture.recipe)
        with mock.patch('dmn.gpu_recipe.read_completion') as completion:
            offered = offer_for_runtime(c.r, self.fixture.recipe)
            again = offer_for_runtime(c.r, self.fixture.recipe)
        self.assertEqual(completion.call_args.args[0], work)
        self.assertEqual(offered, again)
        self.assertEqual(offered['parent'], parent)
        self.assertEqual(offered['trainer']['python'], original['trainer']['python'])
        self.assertEqual(self.fixture.recipe, original)
        self.assertEqual(offered['trainer']['parent_adapter_manifest']['path'],
                         str(work / 'continuation-manifest.json'))


if __name__ == '__main__':
    unittest.main()
