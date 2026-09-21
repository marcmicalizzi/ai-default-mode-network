"""Read-only tensor audit of the pinned 31B source and published GGUF.

Requantizes source rows in bounded RAM with the pinned native quantizer. This
research evidence is NOT a reusable production provenance receipt: it does not
reproduce the full GGUF file, metadata, tokenizer or publisher's command line.
"""
import argparse
import ctypes as C
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dmn.backend import sha256_file
from dmn.storage import write_durable
from dmn.worker_limits import WorkerLimits, run_cpu_worker
from scripts.probe_qlora_31b import inspect
from scripts.safetensor_stream import state_dict
from scripts.validate_31b_native import BASE_SHA


def audit_tokenizer(fields, source):
    """Mirror the pinned Gemma4 vocabulary conversion without loading weights."""
    import gguf
    import numpy as np
    vocab = gguf.LlamaHfVocab(source / 'model')
    tokens, scores, types = [], [], []
    visible = {'<|channel>', '<channel|>', '<|tool_call>', '<tool_call|>',
               '<|tool_response>', '<tool_response|>', '<|"|>'}
    for token, score, kind in vocab.all_tokens():
        text = token.decode()
        tokens.append(text)
        scores.append(score)
        types.append(gguf.TokenType.USER_DEFINED if text in visible else kind)
    prefix = 'tokenizer.ggml.'
    def field(name):
        return fields[prefix + name]
    if tokens != field('tokens'):
        raise ValueError('published token vocabulary differs from source')
    np.testing.assert_array_equal(np.asarray(scores, dtype=np.float32), np.asarray(field('scores'), dtype=np.float32))
    np.testing.assert_array_equal(types, field('token_type'))
    if field('model') != 'gemma4' or field('add_space_prefix') is not False or field('add_bos_token') is not True:
        raise ValueError('published tokenizer flags differ from pinned converter')
    special = gguf.SpecialVocab(source / 'model', load_merges=True)
    for kind, token in special.special_token_ids.items():
        key = getattr(gguf.Keys.Tokenizer, kind.upper() + '_ID')
        if fields[key] != token:
            raise ValueError('published special token differs: ' + kind)
    for kind, value in special.add_special_token.items():
        if kind != 'bos' and field('add_' + kind + '_token') != value:
            raise ValueError('published special-token policy differs: ' + kind)
    if special.merges and special.merges != field('merges'):
        raise ValueError('published tokenizer merges differ')
    if not isinstance(special.chat_template, str) or not isinstance(fields['tokenizer.chat_template'], str):
        raise ValueError('multiple chat templates require a separate comparison')
    template_equal = fields['tokenizer.chat_template'] == special.chat_template
    return {'vocabulary_entries': len(tokens), 'tokens_scores_types_equal': True,
            'special_tokens_equal': True, 'chat_template_equal': template_equal,
            'source_chat_template_sha256': hashlib.sha256(special.chat_template.encode()).hexdigest(),
            'gguf_chat_template_sha256': hashlib.sha256(fields['tokenizer.chat_template'].encode()).hexdigest(),
            'full_runtime_tokenization_equivalence_proved': False}


def read_rows(tensor, start, count):
    import numpy as np
    shape = tensor.get_shape()
    cols = shape[-1] if len(shape) > 1 else math.prod(shape)
    if tensor.get_dtype() != 'BF16' or not 0 <= start < start + count <= (shape[0] if len(shape) > 1 else 1):
        raise ValueError('unexpected source dtype or row range')
    info = tensor.path.stat()
    if (info.st_size, info.st_mtime_ns) != tensor.stamp:
        raise ValueError('source changed after header validation')
    with tensor.path.open('rb') as stream:
        stream.seek(tensor.data_start + tensor.entry['data_offsets'][0] + start * cols * 2)
        raw = stream.read(count * cols * 2)
    if len(raw) != count * cols * 2:
        raise ValueError('truncated source tensor')
    return (np.frombuffer(raw, dtype='<u2').astype('<u4') << 16).view('<f4')


def worker(folder):
    import numpy as np
    request = json.loads((folder / 'input.json').read_text())
    source, base = Path(request['source']), Path(request['gguf'])
    plan = inspect(source)
    write_durable(folder / 'progress.json', {'phase': 'verifying_assets'})
    record = json.loads((source / 'source.json').read_text())
    for name, entry in record['files'].items():
        if (not request.get('tokenizer_only') and name.endswith('.safetensors')
                and sha256_file(source / 'model' / name) != entry['lfs_sha256']):
            raise ValueError('source weight hash mismatch')
    if sha256_file(base) != BASE_SHA:
        raise ValueError('published GGUF identity mismatch')
    sys.path.insert(0, request['gguf_python'])
    import gguf
    reader = gguf.GGUFReader(base)
    if request.get('tokenizer_only'):
        # Release GGUFReader's numerous tiny field views before loading the HF
        # tokenizer, avoiding an unnecessary overlap of metadata allocations.
        fields = {name: value.contents() for name, value in reader.fields.items() if name.startswith('tokenizer.')}
        del reader
        import gc
        gc.collect()
        evidence = audit_tokenizer(fields, source)
        write_durable(folder / 'result.json', {'completed': True, **evidence, 'gguf_sha256': BASE_SHA,
            'source_files': {p.name: sha256_file(p) for p in (source / 'model').iterdir()
                             if p.is_file() and (p.name.startswith('tokenizer') or p.suffix == '.jinja')},
            'source_weights_hashed': False, 'production_provenance_gate_passed': False})
        return
    targets = {t.name: t for t in reader.tensors}
    sources = {}
    mapping = gguf.get_tensor_name_map(gguf.MODEL_ARCH.GEMMA4, 60)
    for name, tensor in state_dict(source / 'model').items():
        if not name.startswith('model.language_model.'):
            continue
        key = name.replace('model.language_model.', 'model.')
        if key.endswith(('layer_scalar', 'per_dim_scale')):
            key += '.weight'
        mapped = mapping.get_name(key, try_suffixes=('.weight',))
        if mapped is None or mapped in sources:
            raise ValueError('unmapped or duplicate text tensor: ' + name)
        sources[mapped] = tensor
    if set(targets) != set(sources) | {'rope_freqs.weight'}:
        raise ValueError('GGUF/source text tensor sets differ: ' + str(set(targets) ^ (set(sources) | {'rope_freqs.weight'})))
    dll_path = Path(request['ggml_base'])
    dll = C.CDLL(str(dll_path))
    quantize = dll.ggml_quantize_chunk
    # ABI from pinned ggml/include/ggml.h; no CUDA context is constructed.
    quantize.restype = C.c_size_t
    quantize.argtypes = [C.c_int, C.c_void_p, C.c_void_p, C.c_int64, C.c_int64, C.c_int64, C.c_void_p]
    counts, failures, rows_checked, bytes_checked = {}, [], 0, 0
    started = time.monotonic()
    for index, (name, target) in enumerate(targets.items()):
        if name == 'rope_freqs.weight':
            expected = np.array([1.] * 64 + [1e30] * 192, dtype='<f4')
            np.testing.assert_array_equal(target.data, expected)
            continue
        tensor = sources[name]
        if list(target.shape) != list(reversed(tensor.get_shape())):
            raise ValueError('tensor shape differs: ' + name)
        kind = target.tensor_type
        if kind not in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.Q4_K, gguf.GGMLQuantizationType.Q6_K):
            raise ValueError('unexpected tensor quantization: ' + str(kind))
        counts[kind.name] = counts.get(kind.name, 0) + 1
        shape = tensor.get_shape()
        nrows, cols = (shape[0], shape[1]) if len(shape) == 2 else (1, math.prod(shape))
        block, width = gguf.GGML_QUANT_SIZES[kind]
        row_bytes = cols // block * width
        actual = target.data.reshape(nrows, row_bytes if kind != gguf.GGMLQuantizationType.F32 else cols)
        chunk_rows = max(1, 1024**2 // cols)
        ranges = ([(row, min(chunk_rows, nrows - row)) for row in range(0, nrows, chunk_rows)]
                  if request['full'] else [(row, 1) for row in sorted({0, nrows // 2, nrows - 1})])
        def compare_rows(item):
            row, count = item
            values = read_rows(tensor, row, count)
            if kind == gguf.GGMLQuantizationType.F32:
                expected = values.view(np.uint8)
            else:
                expected = np.empty(row_bytes * count, dtype=np.uint8)
                written = quantize(int(kind), values.ctypes.data, expected.ctypes.data, 0, count, cols, None)
                if written != expected.nbytes:
                    raise ValueError('native quantizer returned wrong byte count')
            equal = np.array_equal(actual[row:row + count].reshape(-1).view(np.uint8), expected)
            return row, count, expected.nbytes, equal
        # Native quantization releases the GIL. Each task owns its source and
        # output buffers; joining before the next tensor keeps closure state fixed.
        with ThreadPoolExecutor(max_workers=request.get('threads', 1)) as pool:
            for row, count, checked, equal in pool.map(compare_rows, ranges):
                rows_checked += count
                bytes_checked += checked
                if not equal:
                    failures.append({'tensor': name, 'first_differing_chunk_row': row, 'rows_in_chunk': count})
                    break
        write_durable(folder / 'progress.json', {'phase': 'comparing', 'tensors_completed': index + 1,
            'tensors_total': len(targets), 'mismatch_tensors': len(failures), 'rows_checked': rows_checked,
            'seconds': time.monotonic() - started})
    if inspect(source) != plan:
        raise ValueError('source metadata changed')
    write_durable(folder / 'result.json', {'completed': True, 'all_compared_bytes_equal': not failures,
        'full_tensor_payload_comparison': request['full'], 'tensor_types': counts, 'rows_checked': rows_checked,
        'bytes_compared': bytes_checked, 'mismatches': failures, 'seconds': time.monotonic() - started,
        'source_revision': plan['revision'], 'gguf_sha256': BASE_SHA, 'quantizer_sha256': sha256_file(dll_path),
        'full_gguf_reproduced': False, 'tokenizer_verified': False, 'production_provenance_gate_passed': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'gguf', 'gguf-python', 'ggml-base', 'python', 'output', 'worker'):
        parser.add_argument('--' + name, type=Path)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument('--full', action='store_true', help='compare every tensor row, not three sampled rows per matrix')
    scope.add_argument('--tokenizer-only', action='store_true', help='compare tokenizer assets instead of weight payloads')
    parser.add_argument('--threads', type=int, choices=(1, 2, 4), default=1)
    parser.add_argument('--max-seconds', type=int, default=1800)
    args = parser.parse_args()
    if args.worker:
        try:
            return worker(args.worker)
        except Exception as exc:
            write_durable(args.worker / 'failure.json', {'type': type(exc).__name__, 'reason': str(exc)[:4000]})
            raise
    if not all((args.source, args.gguf, args.gguf_python, args.ggml_base, args.python, args.output)):
        parser.error('all asset paths and a fresh --output are required')
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    write_durable(folder / 'input.json', {**{name: str(getattr(args, name).resolve())
        for name in ('source', 'gguf', 'gguf_python', 'ggml_base')}, 'full': args.full,
        'tokenizer_only': args.tokenizer_only, 'threads': args.threads})
    process = run_cpu_worker(args.python, [__file__, '--worker', str(folder)], cwd=ROOT,
        log=folder / 'worker.log', limits=WorkerLimits(2048 * 1024**2, args.max_seconds))
    write_durable(folder / 'process.json', process)
    print(json.dumps(process, indent=2))
    raise SystemExit(0 if process['succeeded'] else 1)


if __name__ == '__main__':
    main()
