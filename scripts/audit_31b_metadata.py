"""Compare source-derived inference metadata without materializing model weights."""
import argparse
import json
from pathlib import Path
import sys
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dmn.backend import sha256_file
from dmn.storage import write_durable
from dmn.worker_limits import WorkerLimits, run_cpu_worker


def compare(base, source, converter):
    import numpy as np
    sys.path.insert(0, str(converter / 'gguf-py'))
    import gguf
    reader = gguf.GGUFReader(base)
    actual = {name: field.contents() for name, field in reader.fields.items()
              if name.startswith('gemma4.') or name == 'general.architecture'}
    # The reader owns many metadata views. Release it before importing Torch.
    del reader
    import gc
    gc.collect()
    sys.path.insert(0, str(converter))
    from conversion.gemma import Gemma4Model
    # Only configuration-to-metadata conversion is requested. Prohibit the
    # converter's default whole-shard index/mapping and all tensor materialization.
    with mock.patch.object(Gemma4Model, 'index_tensors', return_value={}):
        model = Gemma4Model(source, gguf.LlamaFileType.ALL_F32, source / 'unused.gguf')
    model.set_gguf_parameters()
    expected = {}
    for name, field in model.gguf_writer.kv_data[0].items():
        if name.startswith('gemma4.') or name == 'general.architecture':
            value = field.value
            if field.type == gguf.GGUFValueType.FLOAT32:
                value = float(np.float32(value))
            expected[name] = value
    mismatches = []
    for name in sorted(expected.keys() | actual.keys()):
        if name not in expected or name not in actual or expected[name] != actual[name]:
            mismatches.append({'key': name, 'expected': expected.get(name), 'actual': actual.get(name)})
    return {'all_inference_metadata_equal': not mismatches,
            'keys_compared': len(expected.keys() | actual.keys()), 'mismatches': mismatches,
            'inference_metadata': actual, 'scope': 'architecture and gemma4.* keys; no template replacement',
            'model_weights_loaded': False}


def worker(folder):
    request = json.loads((folder / 'input.json').read_text())
    result = compare(Path(request['gguf']), Path(request['source']), Path(request['converter']))
    write_durable(folder / 'result.json', {**result, 'completed': True,
        'source_config_sha256': sha256_file(Path(request['source']) / 'config.json'),
        'gguf_sha256': sha256_file(Path(request['gguf']))})
    if not result['all_inference_metadata_equal']:
        raise ValueError('source-derived inference metadata differs; inspect the private experiment report')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'gguf', 'converter', 'python', 'output', 'worker'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker.resolve())
    if not all((args.source, args.gguf, args.converter, args.python, args.output)):
        parser.error('source, gguf, converter, python and a fresh output are required')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_durable(output / 'input.json', {key: str(getattr(args, key).resolve())
                                         for key in ('source', 'gguf', 'converter')})
    process = run_cpu_worker(args.python, [__file__, '--worker', str(output)], cwd=ROOT,
        log=output / 'worker.log', limits=WorkerLimits(2 * 1024**3, 600))
    write_durable(output / 'process.json', process)
    print(json.dumps(process, indent=2))
    raise SystemExit(0 if process['succeeded'] else 1)


if __name__ == '__main__':
    main()
