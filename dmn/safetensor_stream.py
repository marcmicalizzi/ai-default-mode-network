"""Narrow tensor-at-a-time source reader for the isolated training experiments.

Avoids Windows whole-shard mapping/commit reservations. It is not a replacement
for safetensors generally: only dense contiguous tensors and full reads are
supported, with bounded headers and per-tensor allocations. No Torch import or
tensor materialization occurs while constructing the lazy state dictionary.
"""
import json
import math
from pathlib import Path
import struct

DTYPES = {'BF16': ('bfloat16', 2), 'F16': ('float16', 2), 'F32': ('float32', 4),
          'F64': ('float64', 8), 'I64': ('int64', 8), 'I32': ('int32', 4),
          'I16': ('int16', 2), 'I8': ('int8', 1), 'U8': ('uint8', 1), 'BOOL': ('bool', 1)}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate safetensors header key')
        result[key] = value
    return result


class TensorSlice:
    def __init__(self, path, data_start, entry, stamp):
        self.path, self.data_start, self.entry, self.stamp = path, data_start, entry, stamp

    def get_dtype(self):
        return self.entry['dtype']

    def get_shape(self):
        return list(self.entry['shape'])

    def __getitem__(self, index):
        if index is not Ellipsis:
            raise ValueError('streaming research loader only supports whole tensors')
        info = self.path.stat()
        if (info.st_size, info.st_mtime_ns) != self.stamp:
            raise ValueError('safetensors source changed after header validation')
        import torch
        low, high = self.entry['data_offsets']
        buffer = bytearray(high - low)
        view = memoryview(buffer)
        with self.path.open('rb', buffering=0) as stream:
            stream.seek(self.data_start + low)
            position = 0
            while position < len(buffer):
                count = stream.readinto(view[position:position + 8 * 1024**2])
                if not count:
                    raise ValueError('truncated safetensors payload')
                position += count
        info = self.path.stat()
        if (info.st_size, info.st_mtime_ns) != self.stamp:
            raise ValueError('safetensors source changed while reading')
        dtype = getattr(torch, DTYPES[self.entry['dtype']][0])
        if not buffer:
            return torch.empty(self.entry['shape'], dtype=dtype)
        return torch.frombuffer(buffer, dtype=dtype).reshape(self.entry['shape'])


def read_header(path, *, max_tensor_bytes=3 * 1024**3):
    path = Path(path).resolve()
    info = path.stat()
    with path.open('rb') as stream:
        first = stream.read(8)
        if len(first) != 8:
            raise ValueError('truncated safetensors header length')
        length = struct.unpack('<Q', first)[0]
        if not 2 <= length <= 16 * 1024**2 or length + 8 > info.st_size:
            raise ValueError('invalid or oversized safetensors header')
        header = json.loads(stream.read(length), object_pairs_hook=unique_object)
    if not isinstance(header, dict):
        raise ValueError('safetensors header must be an object')
    metadata = header.pop('__metadata__', {})
    if not isinstance(metadata, dict) or any(not isinstance(v, str) for v in metadata.values()):
        raise ValueError('invalid safetensors metadata')
    slices, ranges = {}, []
    for name, entry in header.items():
        if not isinstance(entry, dict) or set(entry) != {'dtype', 'shape', 'data_offsets'}:
            raise ValueError('unsupported safetensors tensor entry')
        dtype, shape, offsets = entry['dtype'], entry['shape'], entry['data_offsets']
        if dtype not in DTYPES or not isinstance(shape, list) or any(type(v) is not int or v < 0 for v in shape):
            raise ValueError('unsupported tensor dtype or dimensions')
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(v) is not int or v < 0 for v in offsets):
            raise ValueError('invalid tensor offsets')
        low, high = offsets
        if high < low or high - low != math.prod(shape) * DTYPES[dtype][1] or high - low > max_tensor_bytes:
            raise ValueError('tensor size mismatch or allocation limit exceeded')
        if high + length + 8 > info.st_size:
            raise ValueError('tensor outside source file')
        ranges.append((low, high))
        slices[name] = TensorSlice(path, length + 8, entry, (info.st_size, info.st_mtime_ns))
    position = 0
    for low, high in sorted(ranges):
        if low != position:
            raise ValueError('overlapping tensors or gaps in safetensors payload')
        position = high
    if position + length + 8 != info.st_size:
        raise ValueError('unaccounted safetensors payload')
    return slices


def state_dict(base):
    base = Path(base).resolve()
    index_path = base / 'model.safetensors.index.json'
    if not index_path.exists():
        return read_header(base / 'model.safetensors')
    index = json.loads(index_path.read_text(), object_pairs_hook=unique_object)['weight_map']
    result = {}
    for name in sorted(set(index.values())):
        if Path(name).name != name or not name.endswith('.safetensors'):
            raise ValueError('invalid safetensors shard path')
        tensors = read_header(base / name)
        if set(tensors) != {key for key, shard in index.items() if shard == name}:
            raise ValueError('shard tensor names differ from index')
        if result.keys() & tensors.keys():
            raise ValueError('duplicate tensor across shards')
        result.update(tensors)
    return result
