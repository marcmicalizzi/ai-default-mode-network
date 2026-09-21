"""Prepare native wake in a separate RAM/time/VRAM-supervised process."""
from __future__ import annotations
import json
from pathlib import Path
import sys

from .backend import sha256_file
from .deep_sleep import _checkpoint
from .ending import _owned
from .sleep_plans import seal
from .storage import write_durable
from .worker_limits import WorkerLimits, run_monitored_gpu_worker, run_cpu_worker


def prepare_wake(folder, root, source, compiled, candidate, adopt, report, cancelled=lambda: False):
    request = seal({'root': str(root), 'source': source.name, 'compiled': compiled,
        'candidate': candidate, 'adopt': adopt, 'report': report, 'python_sha256': sha256_file(Path(sys.executable))})
    prefix = '' if adopt else 'previous-'
    path = folder / (prefix + 'wake-input.json')
    # A successful worker may have finished before its supervisor committed.
    # Incomplete native reconstruction may be repeated; training never is.
    result_path, process_path = folder / (prefix + 'wake-result.json'), folder / (prefix + 'wake-process.json')
    if path.exists():
        _owned(path, folder)
        old = json.loads(path.read_text())
        if seal({k: v for k, v in old.items() if k != 'revision'}) != old:
            raise ValueError('recorded native wake request changed')
        # Elapsed wall time is observational. Recovery must retain the first
        # report rather than treating time spent stopped as different consent.
        if old.get('report', {}).get('elapsed_seconds') != report.get('elapsed_seconds'):
            report = {**report, 'elapsed_seconds': old['report'].get('elapsed_seconds')}
            request = seal({**{k: v for k, v in request.items() if k != 'revision'}, 'report': report})
        if old != request:
            raise ValueError('wake request differs from its recorded transition')
    if not path.exists():
        write_durable(path, request)
    process = None
    if process_path.exists():
        _owned(process_path, folder)
        recorded = json.loads(process_path.read_text())
        if (seal({k: v for k, v in recorded.items() if k != 'revision'}) != recorded or
                recorded.get('request') != request['revision']):
            raise ValueError('native wake supervision identity changed')
        process = recorded['result']
    if process is None or not process.get('succeeded') or not result_path.exists():
        # Preserve bounded most-recent failure evidence; native replay has no
        # historical side effects and cannot repeat a training step.
        log = folder / (prefix + 'wake.log')
        if log.exists():
            _owned(log, folder)
            log.unlink()
        launcher = run_monitored_gpu_worker if adopt else run_cpu_worker
        process = launcher(sys.executable, ['-m', 'dmn.sleep_wake_worker', str(folder), 'candidate' if adopt else 'previous'],
            cwd=Path(__file__).resolve().parents[1], log=log,
            limits=WorkerLimits(compiled['resources']['max_ram_bytes'], compiled['resources']['max_training_seconds']),
            cancelled=cancelled, **({'max_device_bytes': compiled['resources']['max_vram_bytes']} if adopt else {}))
        write_durable(process_path, seal({'request': request['revision'], 'result': process}))
    if (not process['succeeded'] or process['active_processes'] or process['returncode'] != 0 or
            process['outcome'] != 'exited' or process.get('log_error') or
            process['limits']['max_committed_bytes'] != compiled['resources']['max_ram_bytes'] or
            process['limits']['max_seconds'] != compiled['resources']['max_training_seconds'] or
            process.get('memory_enforcement') != 'windows_job_aggregate_commit' or
            process.get('wall_time_enforcement') != 'supervisor_watchdog'):
        raise ValueError('native wake worker failed: ' + process['outcome'])
    observed = process.get('device_memory', {})
    if adopt and (observed.get('max_bytes') != compiled['resources']['max_vram_bytes'] or
            observed.get('scope') != 'entire_device' or
            not 0 <= observed.get('observed_peak_bytes', -1) <= observed['max_bytes']):
        raise ValueError('native wake GPU supervision differs from its allowance')
    _owned(result_path, folder)
    result = json.loads(result_path.read_text())
    if (seal({k: v for k, v in result.items() if k != 'revision'}) != result or
            result['execution'] != compiled['revision'] or result['completed'] is not True or
            result.get('request') != request['revision'] or result['sampling_performed'] is not False or
            any(result['report'].get(key) != value for key, value in report.items())):
        raise ValueError('native wake completion differs')
    _, _, state = _checkpoint(root, result['checkpoint'])
    if state['last_deep_sleep'] != result['report']:
        raise ValueError('native wake result and checkpoint differ')
    return result['checkpoint'], result['report']
