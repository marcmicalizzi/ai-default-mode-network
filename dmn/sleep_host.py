"""Host resource offer for the explicitly enabled, supervised NF4 service."""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path

from .training import GPU_KIND


def read_offer(path):
    if path is None:
        return None
    path = Path(path)
    if path.stat().st_size > 1024**2:
        raise ValueError('sleep offer exceeds the configuration limit')
    value = json.loads(path.read_text(encoding='utf-8'))
    from .gpu_recipe import validate_recipe
    validate_recipe(value)
    if os.name != 'nt':
        raise ValueError('supervised NF4 sleep currently requires Windows job containment')
    return value


def guard(config, compiled, offer):
    from .gpu_recipe import SCOPE, validate_recipe
    if offer is None:
        raise ValueError('no executable trainer is enabled; supply an explicit supervised sleep offer')
    validate_recipe(offer)
    if os.name != 'nt' or config.backend != 'llama':
        raise ValueError('supervised NF4 sleep requires the Windows native backend')
    if (config.n_ctx > 60160 or not config.experimental_compact_swa or not config.pack_checkpoints or
            config.type_k != 'q8_0' or config.type_v != 'q8_0'):
        raise ValueError('live sleep requires the validated compact packed Q8 context envelope (at most 60160 tokens)')
    if compiled['execution_scope'] != SCOPE or compiled['recipe']['kind'] != GPU_KIND:
        raise ValueError('live service only executes reviewed NF4 recipes')
    recipe = compiled['recipe']
    if any(compiled['resources'][k] > v for k, v in offer['resources'].items()):
        raise ValueError('approved plan exceeds the current host resource offer')
    # Continuation changes only the exact parent adapter manifest. It cannot
    # exchange interpreters, source weights, converters or the GPU workload cap.
    strip = lambda trainer: {k: v for k, v in trainer.items() if k != 'parent_adapter_manifest'}
    if strip(recipe['trainer']) != strip(offer['trainer']):
        raise ValueError('approved trainer differs from the enabled host offer')
    if recipe['parent'].get('model_sha256') != offer['parent'].get('model_sha256'):
        raise ValueError('sleep offer belongs to a different base model')


def offer_for_runtime(runtime, offer):
    """Offer continuation of an adopted managed adapter, without approving it."""
    from .sleep_plans import identity, read_record
    parent = identity(runtime.backend.fingerprint)
    value = copy.deepcopy(offer)
    if parent == value['parent']:
        return value
    from .deep_sleep import read_run
    with runtime.store.mutex:
        completed = runtime.store.db.execute("SELECT id FROM sleep_runs WHERE phase='WakeCommitted' ORDER BY rowid DESC").fetchall()
    run = next((matching for row in completed if (matching := read_run(runtime.store, row[0]))['report'].get('outcome') ==
                'candidate_adopted' and matching['report'].get('new_weights') == parent), None)
    if run is None:
        raise ValueError('no completed managed adapter lineage for this host offer')
    previous = read_record(runtime.store, 'sleep_executions', run['execution'])
    original = previous.get('candidate_reuse', {}).get('run_id', run['id'])
    trained = read_run(runtime.store, original)
    compiled = read_record(runtime.store, 'sleep_executions', trained['execution'])
    from .gpu_recipe import read_completion
    work = runtime.root / 'sleep' / original / 'worker'
    read_completion(work, compiled)
    from .adapters import AdapterSpec
    if [a.identity() for a in runtime.config.lora_adapters] != [
            AdapterSpec(**a).identity() for a in trained['candidate']['adapters']]:
        raise ValueError('current adapter differs from the completed training lineage')
    from .training import tree_manifest
    from .storage import write_durable
    from .backend import sha256_file
    manifest = work / 'continuation-manifest.json'
    payload = tree_manifest(work / 'adapter')
    if manifest.exists():
        if json.loads(manifest.read_text()) != payload:
            raise ValueError('continuation adapter manifest changed')
    else:
        write_durable(manifest, payload)
    value['parent'] = parent
    value['trainer']['parent_adapter_manifest'] = {'path': str(manifest), 'sha256': sha256_file(manifest)}
    return value
