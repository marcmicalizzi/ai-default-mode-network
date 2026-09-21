"""Disposable real NF4 service: review-first, adoption-only, then continuation.

All decisions are injected mechanics fixtures, never consent attributed to a
model. Its direct command accepts only generated models below 4 MiB; the separate
31B wrapper explicitly pins the full model/projector and contains the whole trial.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dmn.backend import make_backend, sha256_file
from dmn.config import Config
from dmn.deep_sleep import read_run, run_fixture_sleep
from dmn.gpu_recipe import PACKAGES
from dmn.learning import HELP, list_plans
from dmn.runtime import Runtime
from dmn.sleep_plans import identity, read_record
from dmn.sleep_service import SleepService
from dmn.storage import json_text, write_durable
from dmn.training import GPU_KIND, CHECKS


def validate(output, proof_path, training_python, *, full_model=None, projector=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    proof = json.loads(proof_path.read_text())
    full = full_model is not None
    model = full_model if full else proof_path.parent / 'base.gguf'
    if full:
        from dmn.exact_base_provenance import BASE_SHA
        if sha256_file(model) != BASE_SHA or os.environ.get('DMN_GPU_PROBE_CONTAINED') != '1':
            raise ValueError('full-model service validation requires the pinned base and outer containment')
        if projector is None or sha256_file(projector) != '21487ff26d08f7ddd1d654d3bbfc1ae1020aab3119f5bf654742ce4697732e4e':
            raise ValueError('full-model service validation requires the matching projector')
    elif proof.get('request', {}).get('kind') != 'tiny_cpu_base_provenance_v1' or model.stat().st_size > 4 * 1024**2:
        raise ValueError('only a generated tiny disposable inference model is allowed')
    output.mkdir(parents=True, exist_ok=False)
    config = Config(model_path=str(model), vision_projector_path=str(projector) if full else '',
        n_ctx=60000, n_gpu_layers=-1, n_threads=8 if full else 2, offload_kqv=True,
        prompt_format='plain', type_k='q8_0', type_v='q8_0', flash_attn=True, swa_full=False,
        experimental_compact_swa=True, pack_checkpoints=True, clock_interval_seconds=0,
        preparation_tokens=1, checkpoint_tokens=100000, checkpoint_policy='effects')
    resources = {'max_ram_bytes': 4 * 1024**3, 'max_training_seconds': 240,
                 'max_vram_bytes': 12 * 1024**3, 'max_disk_bytes': 2 * 1024**3}
    if full:
        resources = {'max_ram_bytes':32 * 1024**3, 'max_training_seconds':3600,
                     'max_vram_bytes':31 * 1024**3, 'max_disk_bytes':16 * 1024**3}
        from dmn.training import CONVERTER_REVISION
        conversion = {key: proof[key] for key in ('base_manifest', 'converter_manifest')}
        conversion.update(converter_revision=CONVERTER_REVISION, inference_name='gemma-4-31B-it-uncensored-heretic')
    else:
        conversion = proof['request']['conversion']
    trainer = {**conversion, 'learning_rate': .0001, 'seed': 17,
        'python': str(training_python), 'python_sha256': sha256_file(training_python),
        'packages': json.loads(subprocess.check_output([str(training_python), '-c',
            'import json,importlib.metadata as m; print(json.dumps({n:m.version(n) for n in ' + repr(PACKAGES) + '}))'], text=True)),
        'provenance_manifest': proof['provenance_manifest'] if full else {'path': str(proof_path), 'sha256': sha256_file(proof_path)},
        'gpu': {'torch_vram_bytes': (22 if full else 2) * 1024**3, 'max_sequence_tokens': 256, 'max_rank': 2, 'max_steps': 64}}
    backend = make_backend(config)
    recipe = {'schema': 1, 'kind': GPU_KIND, 'parent': identity(backend.fingerprint),
              'resources': resources, 'checks': CHECKS, 'trainer': trainer}
    root = output / 'instance'
    with mock.patch('dmn.runtime.PROTOCOL', 'Disposable supervised service test. Injected actions are not model consent.'):
        runtime = Runtime(root, config, backend, sleep_offer=recipe)
    if full:
        unit = runtime.backend.tokenize('Synthetic validation context. Red green blue. ')
        runtime._eval((unit * (55000 // len(unit) + 1))[:55000 - len(runtime.backend.tokens)])
    observations, source_tokens, queued = [], [], []

    def action(**value):
        raw = ('\n<dmn_action>' + json_text(value) + '</dmn_action>\n').encode()
        # Scripted actions are not native choices. Use a known ordinary token
        # so a sampled EOG cannot silently put the fixture to sleep midway.
        token = runtime.backend.tokenize(' synthetic')[0]
        if runtime.backend.is_eog(token):
            raise ValueError('fixture action token must not be EOG')
        planned = runtime._plan_action
        def checked(action, staged):
            result, effect = planned(action, staged)
            if not result['ok']:
                raise ValueError('scripted action refused: ' + json_text(result))
            return result, effect
        with mock.patch.object(runtime.backend, 'piece', return_value=raw), mock.patch.object(runtime.backend, 'sample', return_value=token), mock.patch.object(runtime, '_plan_action', side_effect=checked):
            runtime._generate_one()

    def review_and_sleep(revision):
        # The generated fixture has an almost byte-level vocabulary. Reserve
        # room for its complete paginated review so retirement does not correctly
        # invalidate the middle of this scripted reading sequence. Actual model
        # readers must restart their review after retirement, as the contract says.
        runtime._ensure_space(16000 if full else 40000)
        length = len(json_text(read_record(runtime.store, 'sleep_executions', revision)))
        while runtime._learning_reads.get(revision, 0) < length:
            before = runtime._learning_reads.get(revision, 0)
            action(op='learning_execution_read', revision=revision, offset=before, limit=2000)
            if runtime._learning_reads.get(revision, 0) <= before:
                raise ValueError('review made no progress')
        action(op='learning_execution_decide', revision=revision, decision='approve')
        action(op='deep_sleep', revision=revision)
        if runtime.state['mode'] != 'deep_sleep':
            raise ValueError('reviewed plan did not reach its sleep boundary')
        source_tokens.append(runtime.backend.tokens.copy())

    def compile_new(adoption, target):
        plan = copy.deepcopy(HELP['create']['plan'])
        plan.update(sources=[], examples=[{'input':'Disposable synthetic test record:\n', 'target':target, 'sources':[], 'purpose':'new'}],
                    checks=CHECKS, resources=resources)
        plan['preferences'].update(steps=2, scale=.1, adoption=adoption)
        action(op='learning_plan_create', plan=plan)
        draft = list_plans(runtime.store, 0, 50)[-1]['revision']
        offered = runtime.store.db.execute('SELECT revision FROM sleep_recipes ORDER BY rowid DESC LIMIT 1').fetchone()[0]
        action(op='learning_compile', draft_revision=draft, recipe_revision=offered)
        revision = runtime.store.db.execute('SELECT revision FROM sleep_executions ORDER BY rowid DESC LIMIT 1').fetchone()[0]
        review_and_sleep(revision)

    def transition(root, run_id, **kwargs):
        if not queued:
            queued.append(runtime.enqueue('Synthetic input queued while inference is unloaded.'))
        result = run_fixture_sleep(root, run_id, **kwargs)
        if result['phase'] != 'WakeCommitted':
            raise ValueError('service transition stopped: ' + json_text(result.get('report')))
        path = root / 'checkpoints' / result['wake_checkpoint']
        engine = json.loads((path / 'engine.json').read_text())
        if engine['tokens'] != source_tokens[-1]:
            raise ValueError('transition did not preserve its current retained context')
        observations.append({'run_id':run_id, 'retained_tokens':len(engine['tokens']),
            'outcome':result['report']['outcome'], 'training_performed':result['report']['training_performed']})
        print(json.dumps(observations[-1]), flush=True)
        return result

    def factory(root, config, **kwargs):
        nonlocal runtime
        fresh = Runtime(root, config, **kwargs)
        runtime = fresh
        if (fresh.state['last_restore']['prompt_tokens_reevaluated'] != 0 or
                fresh.state['last_restore']['restored_tokens'] != len(source_tokens[-1])):
            raise ValueError('service native restore replayed or changed tokens')
        def next_stage():
            if full:
                fresh.state['mode'] = 'suspended'
            elif len(observations) == 1:
                if fresh.config.lora_adapters:
                    raise ValueError('review-first silently adopted a candidate')
                action(op='learning_candidate_prepare', run_id=observations[0]['run_id'])
                revision = fresh.store.db.execute('SELECT revision FROM sleep_executions ORDER BY rowid DESC LIMIT 1').fetchone()[0]
                if not read_record(fresh.store, 'sleep_executions', revision).get('candidate_reuse'):
                    raise ValueError('adoption-only plan was not prepared')
                review_and_sleep(revision)
            elif len(observations) == 2:
                if len(fresh.config.lora_adapters) != 1:
                    raise ValueError('adoption-only did not install one adapter')
                compile_new('automatic_if_checks_pass', 'A second selected synthetic target')
            else:
                fresh.state['mode'] = 'suspended'
        fresh.run = next_stage
        return fresh

    service = SleepService(runtime, transition=transition, runtime_factory=factory)
    try:
        service.offer_recipe()
        runtime.tick()
        action(op='send_message', content='Synthetic publication before learning.')
        compile_new('automatic_if_checks_pass' if full else 'review_first', 'A first selected synthetic target')
        service.run()
        if [o['training_performed'] for o in observations] != ([True] if full else [True, False, True]):
            raise ValueError('service repeated training or skipped a requested cycle')
        if len(runtime.store.messages()) != 1 or runtime.store.next_event(runtime.state['event_cursor'])['id'] != queued[0]:
            raise ValueError('historical publication or pending input changed')
        # Adoption-only may retain a worker-free run directory. Its original
        # completed factors remain the lineage for the separately reviewed third cycle.
        if not full:
            third = read_run(runtime.store, observations[2]['run_id'])
            compiled = read_record(runtime.store, 'sleep_executions', third['execution'])
            if not compiled['lineage']:
                raise ValueError('third cycle did not preserve the deployed PEFT parent')
        write_durable(output / 'result.json', {'completed':True, 'synthetic_only':True,
            'injected_choices_are_not_model_consent':True, 'continuous_service':True,
            'queued_input_preserved':True, 'published_messages_unchanged':True,
            'strict_native_restore_without_replay':True, 'full_model':full, 'projector_loaded':full,
            'cycles':observations})
    finally:
        service.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'proof', 'training-python'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    validate(args.output.resolve(), args.proof.resolve(), args.training_python.resolve())
