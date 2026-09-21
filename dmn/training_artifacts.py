"""Bound trusted adapter serialization before opening output files.

This is an application allocation check, not an OS filesystem quota. No full
model save is permitted. Large tensor serialization happens inside the worker's
aggregate RAM limit; only the checked adapter bytes reach disk.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

MAX_ADAPTER_BYTES = 128 * 1024**2
MAX_CONFIG_BYTES = 128 * 1024


def copy_bounded(source, target, maximum):
    """Refuse growth while copying, before writing beyond the declared bound."""
    if Path(source).stat().st_size > maximum:
        raise ValueError('artifact exceeds its copy allowance')
    total = 0
    with Path(source).open('rb') as src, Path(target).open('xb') as dst:
        while block := src.read(min(65536, maximum - total + 1)):
            if total + len(block) > maximum:
                raise ValueError('artifact grew beyond its copy allowance')
            dst.write(block)
            total += len(block)
        dst.flush()
        os.fsync(dst.fileno())


def write_bytes(path, data, maximum):
    if len(data) > maximum:
        raise ValueError('adapter artifact exceeds its bounded output allowance')
    with Path(path).open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def save_adapter(model, folder, expected_names):
    from peft import get_peft_model_state_dict
    from safetensors.torch import save
    factors = get_peft_model_state_dict(model)
    if factors.keys() != expected_names:
        raise ValueError('serialization contains unexpected adapter tensors')
    size = sum(t.numel() * t.element_size() for t in factors.values())
    if size + 1024**2 > MAX_ADAPTER_BYTES:
        raise ValueError('adapter factors exceed the serialization allowance')
    payload = save({n: t.detach().contiguous().cpu() for n, t in factors.items()}, metadata={'format': 'pt'})
    config = model.peft_config['default'].to_dict()
    config['inference_mode'] = True
    encoded = json.dumps(config, indent=2, sort_keys=True,
                         default=lambda value: sorted(value) if isinstance(value, set) else value).encode()
    if len(payload) > MAX_ADAPTER_BYTES or len(encoded) > MAX_CONFIG_BYTES:
        raise ValueError('serialized adapter exceeds the output allowance')
    folder.mkdir(exist_ok=False)
    write_bytes(folder / 'adapter_model.safetensors', payload, MAX_ADAPTER_BYTES)
    write_bytes(folder / 'adapter_config.json', encoded, MAX_CONFIG_BYTES)
