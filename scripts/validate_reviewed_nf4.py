"""Disposable reviewed NF4 sleep/reload/wake test with injected mechanics choices."""
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
from dmn.backend import sha256_file
from dmn.config import Config
from dmn.deep_sleep import run_fixture_sleep
from dmn.gpu_recipe import PACKAGES
from dmn.learning import HELP, list_plans
from dmn.runtime import Runtime
from dmn.sleep_plans import identity, read_record
from dmn.storage import Store, json_text, write_durable
from dmn.training import GPU_KIND, CHECKS


def validate(output, proof_path, training_python):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Tiny native inference stays on CPU.
    proof = json.loads(proof_path.read_text())
    if proof.get('request', {}).get('kind') != 'tiny_cpu_base_provenance_v1':
        raise ValueError('requires the generated tiny reproduction fixture')
    model = proof_path.parent / 'base.gguf'
    if model.stat().st_size > 4 * 1024**2:
        raise ValueError('only a tiny disposable inference model is allowed')
    output.mkdir(parents=True, exist_ok=False)
    config = Config(model_path=str(model), n_ctx=32768, n_gpu_layers=0, n_threads=1, offload_kqv=False,
        prompt_format='plain', type_k='q8_0', type_v='q8_0', flash_attn=True, swa_full=False, experimental_compact_swa=True,
        pack_checkpoints=True, clock_interval_seconds=0, preparation_tokens=1,
        checkpoint_tokens=100000, checkpoint_policy='effects')
    root = output / 'instance'
    with mock.patch('dmn.runtime.PROTOCOL', 'Disposable NF4 integration. Injected actions test mechanics, not consent.'):
        runtime = Runtime(root, config, sleep_test_mode=True)
    try:
        def action(**value):
            raw = ('\n<dmn_action>' + json_text(value) + '</dmn_action>\n').encode()
            with mock.patch.object(runtime.backend, 'piece', return_value=raw):
                runtime._generate_one()

        plan = copy.deepcopy(HELP['create']['plan'])
        plan.update(sources=[], examples=[{'input': 'Prefix: ', 'target': 'Chosen target', 'sources': [], 'purpose': 'new'}],
                    checks=CHECKS, resources={'max_ram_bytes': 4 * 1024**3, 'max_training_seconds': 240,
                        'max_vram_bytes': 12 * 1024**3, 'max_disk_bytes': 2 * 1024**3})
        plan['preferences'].update(steps=2, scale=.1, adoption='automatic_if_checks_pass')
        trainer = {**proof['request']['conversion'], 'learning_rate': .0001, 'seed': 17,
            'python': str(training_python), 'python_sha256': sha256_file(training_python),
            'packages': json.loads(subprocess.check_output([str(training_python), '-c',
                'import json,importlib.metadata as m; print(json.dumps({n:m.version(n) for n in ' + repr(PACKAGES) + '}))'], text=True)),
            'provenance_manifest': {'path': str(proof_path), 'sha256': sha256_file(proof_path)},
            'gpu': {'torch_vram_bytes': 2 * 1024**3, 'max_sequence_tokens': 256, 'max_rank': 2, 'max_steps': 64}}
        recipe = {'schema': 1, 'kind': GPU_KIND, 'parent': identity(runtime.backend.fingerprint),
                  'resources': plan['resources'], 'checks': CHECKS, 'trainer': trainer}
        offered = runtime.offer_learning_recipe(recipe)['revision']
        runtime.tick()
        action(op='send_message', content='Fixture publication before learning.')
        action(op='learning_plan_create', plan=plan)
        draft = list_plans(runtime.store, 0, 50)[-1]['revision']
        action(op='learning_compile', draft_revision=draft, recipe_revision=offered)
        row = runtime.store.db.execute('SELECT revision FROM sleep_executions ORDER BY rowid DESC LIMIT 1').fetchone()
        if row is None:
            raise ValueError('fixture compilation failed')
        revision = row[0]
        compiled = read_record(runtime.store, 'sleep_executions', revision)
        length = len(json_text(compiled))
        while runtime._learning_reads.get(revision, 0) < length:
            before = runtime._learning_reads.get(revision, 0)
            action(op='learning_execution_read', revision=revision, offset=before, limit=2000)
            if runtime._learning_reads.get(revision, 0) <= before:
                raise ValueError('compiled-plan fixture review made no progress')
        action(op='learning_execution_decide', revision=revision, decision='approve')
        action(op='deep_sleep', revision=revision)
        if runtime.state['mode'] != 'deep_sleep':
            raise ValueError('fixture did not reach the explicit sleep boundary')
        run_id = runtime.state['sleep_run_id']
        tokens = runtime.backend.tokens.copy()
        event_cursor = runtime.state['event_cursor']
    finally:
        runtime.close()
    store = Store(root)
    try:
        queued = store.enqueue('user_message', {'content': 'Unselected queued fixture input.'})
    finally:
        store.close()
    result = run_fixture_sleep(root, run_id, _contained_wake=True)
    write_durable(output / 'transition.json', result)
    if result['phase'] != 'WakeCommitted' or result.get('report', {}).get('outcome') != 'candidate_adopted':
        raise ValueError('NF4 fixture transition did not adopt: ' + json_text(result.get('report', {})))
    store = Store(root)
    try:
        saved = store.latest()
    finally:
        store.close()
    manifest = json.loads((saved / 'manifest.json').read_text())
    wake_engine = json.loads((saved / 'engine.json').read_text())
    wake_state = json.loads((saved / 'runtime.json').read_text())
    if wake_engine['tokens'] != tokens or wake_state['event_cursor'] != event_cursor:
        raise ValueError('wake checkpoint changed the retained tokens or pending-input cursor')
    # The normal resume/wake notices can themselves trigger retirement at a
    # nearly full context. Verify the actual transition before adding notices.
    write_durable(output / 'wake-evidence.json', {'retained_tokens_exact': True,
        'retained_tokens': len(tokens), 'queued_input_cursor_unchanged': True})
    runtime = Runtime(root, Config(**manifest['fingerprint']['config']), sleep_test_mode=True)
    try:
        if runtime.state['last_restore']['prompt_tokens_reevaluated'] != 0:
            raise ValueError('strict native wake restore replayed tokens')
        retirements = runtime.state['context_retirements'] - wake_state['context_retirements']
        if (runtime.state['last_restore']['restored_tokens'] != len(tokens) or
                runtime.state['event_cursor'] != event_cursor or retirements not in (0, 1) or
                (not retirements and runtime.backend.tokens[:len(tokens)] != tokens)):
            raise ValueError('restored tokens, retirement accounting or queued input cursor changed')
        if len(runtime.store.messages()) != 1 or runtime.store.next_event(event_cursor)['id'] != queued:
            raise ValueError('publication or pending input changed')
        write_durable(output / 'result.json', {'completed': True, 'synthetic_only': True,
            'injected_choices_are_not_model_consent': True, 'retained_tokens': len(tokens),
            'strict_restore_zero_replay': True, 'published_messages_unchanged': True, 'queued_input_preserved': True,
            'context_retirements_during_resume': retirements,
            'training': result['report']['candidate']['training']})
    finally:
        runtime.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'proof', 'training-python'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    validate(args.output.resolve(), args.proof.resolve(), args.training_python.resolve())
