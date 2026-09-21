"""Contained native wake preparation, without sampling or publishing a checkpoint."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys

from .backend import make_backend, sha256_file
from .config import Config
from .deep_sleep import FixtureExecutor, _checkpoint
from .diskspace import check_space
from .ending import Lifecycle, _owned
from .sleep_plans import seal, implementation_identity, identity
from .storage import write_durable


def wake(folder, slot='candidate'):
    if slot not in {'candidate', 'previous'}:
        raise ValueError('invalid native wake slot')
    prefix = '' if slot == 'candidate' else 'previous-'
    contained = 'DMN_GPU_PROBE_CONTAINED' if slot == 'candidate' else 'DMN_CPU_WORKER_CONTAINED'
    if os.environ.get(contained) != '1':
        raise ValueError('native wake requires the contained worker launcher')
    request = json.loads((folder / (prefix + 'wake-input.json')).read_text())
    if seal({k: v for k, v in request.items() if k != 'revision'}) != request:
        raise ValueError('native wake request changed')
    if request['adopt'] != (slot == 'candidate'):
        raise ValueError('native wake request and output slot differ')
    compiled = request['compiled']
    if (seal({k: v for k, v in compiled.items() if k != 'revision'}) != compiled or
            compiled['implementation'] != implementation_identity()):
        raise ValueError('native wake implementation differs from the reviewed plan')
    root = Path(request['root']).resolve()
    if folder != root / 'sleep' / request['report']['run_id']:
        raise ValueError('native wake folder differs from its owned transition')
    Lifecycle(root).require_open()
    source, manifest, state = _checkpoint(root, request['source'])
    if (identity(manifest['fingerprint']) != compiled['parent'] or
            state.get('instance_id') != compiled['instance_id'] or state.get('mode') != 'deep_sleep' or
            state.get('sleep_run_id') != request['report']['run_id'] or state.get('hold') or
            request['report']['execution'] != compiled['revision']):
        raise ValueError('native wake source differs from the approved sleep boundary')
    config = Config(**manifest['fingerprint']['config'])
    if sha256_file(Path(sys.executable)) != request['python_sha256']:
        raise ValueError('native wake interpreter changed')
    for spec in (request.get('candidate') or {}).get('adapters', []):
        path = Path(spec['path'])
        _owned(path, root / 'adapters')
        if sha256_file(path) != spec['sha256']:
            raise ValueError('managed wake adapter changed')

    # Two deterministic output slots cover candidate and previous-weight wake.
    # The existing source is not charged as a new allocation. These checks guard
    # trusted native writes; they are not a filesystem sandbox for arbitrary code.
    import uuid
    slots = [root / 'checkpoints' / uuid.uuid5(uuid.UUID(request['report']['run_id']), mode).hex
             for mode in ('adopt', 'previous')]
    def guard(path, amount, purpose):
        used = sum(p.stat().st_size for location in [folder, *slots] if location.exists()
                   for p in location.rglob('*') if p.is_file())
        if used + amount > compiled['resources']['max_disk_bytes']:
            raise ValueError('native wake exceeds the reviewed disk allowance')
        check_space(path, amount, config.checkpoint_reserve_bytes, purpose)
    def factory(config):
        backend = make_backend(config)
        backend.storage_guard = guard
        backend._state_work_dir = folder
        original_save = backend.save
        def save(path):
            guard(path, backend.checkpoint_size_bytes(), 'native wake checkpoint')
            return original_save(path)
        backend.save = save
        return backend
    size = sum((source / name).stat().st_size for name in manifest['files'])
    guard(root / 'checkpoints', size * 2, 'native wake workspace')
    executor = FixtureExecutor(factory, fixture_only=False)
    name, report = executor.wake(root, source, manifest, state, request['candidate'], request['adopt'], request['report'])
    Lifecycle(root).require_open()
    write_durable(folder / (prefix + 'wake-result.json'), seal({'execution': compiled['revision'], 'request': request['revision'],
        'checkpoint': name, 'report': report, 'completed': True, 'sampling_performed': False}))


if __name__ == '__main__':
    folder = Path(sys.argv[1]).resolve()
    slot = sys.argv[2] if len(sys.argv) > 2 else 'candidate'
    try:
        wake(folder, slot)
    except Exception as exc:
        write_durable(folder / (('previous-' if slot == 'previous' else '') + 'wake-failure.json'),
                      {'error_type': type(exc).__name__, 'reason': str(exc)[:2000]})
        raise
