"""Restricted codec for pinned llama.cpp Gemma4 session-v9 ISWA state.

Used only by explicit offline conversion and disposable research. Only
single-sequence, non-transposed F16/Q8 KV is accepted. Callers must establish
the native revision and model independently;
the session header alone does not identify a compatible build or model.
Source: ggml-org/llama.cpp at 4df29be4f4c3673f428170fda944a5b19f743bb8.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import struct

from .compact_cache import validate_retirements


@dataclass(frozen=True)
class Tensor:
    kind: int
    row_bytes: int
    offset: int


@dataclass(frozen=True)
class Cache:
    start: int
    end: int
    positions: tuple[int, ...]
    layers: int
    tensors: tuple[Tensor, ...]  # all K, then all V


@dataclass(frozen=True)
class State:
    path: Path
    size: int
    tokens: tuple[int, ...]
    global_cache: Cache
    local_cache: Cache


class Reader:
    def __init__(self, stream):
        self.stream = stream
        self.size = stream.seek(0, 2)
        stream.seek(0)

    def read(self, n):
        if n < 0 or n > self.size - self.stream.tell():
            raise ValueError("truncated native state")
        data = self.stream.read(n)
        if len(data) != n:
            raise ValueError("native state changed while reading")
        return data

    def unpack(self, fmt):
        return struct.unpack("<" + fmt, self.read(struct.calcsize("<" + fmt)))

    def skip(self, n):
        if n < 0 or n > self.size - self.stream.tell():
            raise ValueError("truncated native tensor")
        self.stream.seek(n, 1)


def _cache(reader, token_count):
    start = reader.stream.tell()
    streams, count = reader.unpack("II")
    if streams != 1 or not 1 <= count <= token_count:
        raise ValueError("require one nonempty sequence with bounded cell count")
    positions = []
    for _ in range(count):
        pos, n_seq, seq = reader.unpack("iIi")
        if n_seq != 1 or seq != 0 or not 0 <= pos < token_count:
            raise ValueError("unsupported native cell metadata")
        positions.append(pos)
    if len(set(positions)) != count:
        raise ValueError("duplicate native cell positions")
    transposed, layers = reader.unpack("II")
    if transposed or not 1 <= layers <= 512:
        raise ValueError("require non-transposed KV and a bounded layer count")
    tensors = []
    for _ in range(2 * layers):
        kind, row_bytes = reader.unpack("iQ")
        if kind not in (1, 8) or not 1 <= row_bytes <= 1024 * 1024:
            raise ValueError("unsupported KV type or row size")
        if row_bytes % (2 if kind == 1 else 34):
            raise ValueError("invalid native KV block size")
        tensors.append(Tensor(kind, row_bytes, reader.stream.tell()))
        reader.skip(count * row_bytes)
    return Cache(start, reader.stream.tell(), tuple(positions), layers, tuple(tensors))


def inspect_state(path):
    path = Path(path)
    with path.open("rb") as stream:
        reader = Reader(stream)
        magic, version, count = reader.unpack("III")
        if magic != 0x6767736E or version != 9 or not 1 <= count <= 16 * 1024 * 1024:
            raise ValueError("require a bounded full-context session-v9 file")
        tokens = reader.unpack("i" * count)
        if min(tokens) < 0:
            raise ValueError("invalid token IDs")
        size, = reader.unpack("I")
        if size != 6 or reader.read(size) != b"gemma4":
            raise ValueError("only Gemma4 ISWA state is supported by this restricted codec")
        global_cache = _cache(reader, count)
        local_cache = _cache(reader, count)
        if reader.stream.tell() != reader.size:
            raise ValueError("unexpected trailing native state")
        if set(global_cache.positions) != set(range(count)):
            raise ValueError("global cache does not contain every retained position")
        return State(path, reader.size, tokens, global_cache, local_cache)


def _copy(source, target, offset, size):
    source.seek(offset)
    while size:
        data = source.read(min(size, 8 * 1024 * 1024))
        if not data:
            raise ValueError("source changed during native state copy")
        target.write(data)
        size -= len(data)


def compact_state(source, destination, window):
    """Copy global KV verbatim and retain every unmasked local row, without eval.

    STANDARD SWA masks p when max_position - p >= window. Preserve the window
    ending at the last evaluated token, including its boundary row even though
    the next token will no longer attend to that row.
    """
    state = inspect_state(source)
    if type(window) is not int or not 1 <= window <= 16 * 1024 * 1024:
        raise ValueError("invalid SWA window")
    cutoff = max(0, len(state.tokens) - window)
    local = state.local_cache
    indices = [i for i, pos in enumerate(local.positions) if pos >= cutoff]
    if {local.positions[i] for i in indices} != set(range(cutoff, len(state.tokens))):
        raise ValueError("local cache is missing part of the required recent window")
    # Refuse overwrite, including the source itself. Stream tensors to bound RAM.
    with state.path.open("rb") as src, Path(destination).open("xb") as dst:
        _copy(src, dst, 0, local.start)
        dst.write(struct.pack("<II", 1, len(indices)))
        for i in indices:
            dst.write(struct.pack("<iIi", local.positions[i], 1, 0))
        dst.write(struct.pack("<II", 0, local.layers))
        for tensor in local.tensors:
            dst.write(struct.pack("<iQ", tensor.kind, tensor.row_bytes))
            # Coalesce physical runs; avoids thousands of seeks for large files.
            runs = []
            for i in indices:
                if runs and runs[-1][0] + runs[-1][1] == i:
                    runs[-1][1] += 1
                else:
                    runs.append([i, 1])
            for first, count in runs:
                _copy(src, dst, tensor.offset + first * tensor.row_bytes, count * tensor.row_bytes)
    converted = inspect_state(destination)
    if converted.tokens != state.tokens:
        raise AssertionError("native token IDs changed during conversion")
    return {"source_bytes": state.size, "converted_bytes": converted.size,
            "global_cells_retained": len(state.global_cache.positions),
            "local_cells_retained": len(indices), "masked_local_cells_removed": len(local.positions) - len(indices),
            "window": window, "prompt_tokens_reevaluated": 0}


def row_hashes(state, cache):
    """Content-free per-position evidence, without keeping tensor data in RAM."""
    hashes = {}
    with state.path.open("rb") as stream:
        for index, tensor in enumerate(cache.tensors):
            stream.seek(tensor.offset)
            for pos in cache.positions:
                data = stream.read(tensor.row_bytes)
                if len(data) != tensor.row_bytes:
                    raise ValueError("truncated native row")
                hashes[index, pos] = hashlib.sha256(data).hexdigest()
    return hashes


def verify_compaction(source, destination, window):
    """Stream byte comparisons with bounded RAM; do not hash/store private rows."""
    before, after = inspect_state(source), inspect_state(destination)
    cutoff = max(0, len(before.tokens) - window)
    indices = [i for i, p in enumerate(before.local_cache.positions) if p >= cutoff]
    if (before.tokens != after.tokens or before.local_cache.start != after.local_cache.start or
            after.local_cache.positions != tuple(before.local_cache.positions[i] for i in indices) or
            set(after.local_cache.positions) != set(range(cutoff, len(before.tokens)))):
        raise ValueError("converted positions or tokens differ from the required state")
    with before.path.open("rb") as left, after.path.open("rb") as right:
        def compare(a, b, count):
            left.seek(a)
            right.seek(b)
            while count:
                n = min(count, 8 * 1024 * 1024)
                first, second = left.read(n), right.read(n)
                if len(first) != n or len(second) != n or first != second:
                    raise ValueError("conversion changed retained native KV bytes")
                count -= n
        compare(0, 0, before.local_cache.start)
        a, b = before.local_cache, after.local_cache
        if a.layers != b.layers:
            raise ValueError("converted local layer count differs")
        for original, copied in zip(a.tensors, b.tensors):
            if (original.kind, original.row_bytes) != (copied.kind, copied.row_bytes):
                raise ValueError("converted local tensor geometry differs")
            for target_row, source_row in enumerate(indices):
                compare(original.offset + source_row * original.row_bytes,
                        copied.offset + target_row * copied.row_bytes, original.row_bytes)
    return {"global_section_byte_equal": True, "retained_local_kv_rows_byte_equal": True,
            "tokens_equal": True}
