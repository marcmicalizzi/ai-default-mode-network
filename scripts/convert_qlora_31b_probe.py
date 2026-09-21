"""CPU-only factor conversion check for a completed synthetic 31B GPU probe.

No native model, instance or inference GGUF is opened. This is not provenance
verification for an existing inference base or authorization to adopt an adapter.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dmn.backend import sha256_file
from dmn.storage import write_durable
from dmn.training import verify_tree
from dmn.worker_limits import WorkerLimits, run_cpu_worker
from scripts.probe_qlora_31b import inspect


def validate(probe):
    result = json.loads((probe / 'result.json').read_text())
    if result.get('completed') is not True or result.get('steps_completed') != 1:
        raise ValueError('requires the completed one-step 31B experiment')
    plan = inspect(Path(result['plan']['source']))
    if plan != result['plan'] or not plan['synthetic_text_only']:
        raise ValueError('source or declared experiment changed')
    adapter = probe / 'adapter/adapter_model.safetensors'
    if adapter.stat().st_size > 32 * 1024**2 or sha256_file(adapter) != result['adapter_sha256']:
        raise ValueError('adapter differs from completed experiment or exceeds size bound')
    config = json.loads((probe / 'adapter/adapter_config.json').read_text())
    if (config.get('r') != 2 or config.get('lora_alpha') != 4 or config.get('bias') != 'none'
            or config.get('use_dora') or config.get('use_rslora') or config.get('modules_to_save')
            or set(config['target_modules']) != set(plan['profile']['target_modules'])):
        raise ValueError('unexpected adapter geometry or targets')
    return plan


def worker(folder):
    import numpy as np
    from safetensors.numpy import load_file
    request = json.loads((folder / 'input.json').read_text())
    probe = Path(request['probe'])
    plan = validate(probe)
    converter = verify_tree(request['converter'])
    subprocess.run([sys.executable, str(converter / 'convert_lora_to_gguf.py'), '--outtype', 'f32',
        '--base', str(Path(plan['source']) / 'model'), '--outfile', str(folder / 'adapter.gguf'),
        str(probe / 'adapter')], check=True, stdin=subprocess.DEVNULL)
    sys.path.insert(0, str(converter / 'gguf-py'))
    import gguf
    actual = {t.name: t.data for t in gguf.GGUFReader(folder / 'adapter.gguf').tensors}
    factors = load_file(probe / 'adapter/adapter_model.safetensors')
    expected = {}
    for layer in range(plan['profile']['num_hidden_layers']):
        for hf, cpp in (('q_proj', 'attn_q'), ('o_proj', 'attn_output')):
            for side in ('A', 'B'):
                expected[f'blk.{layer}.{cpp}.weight.lora_{side.lower()}'] = factors[
                    f"base_model.model.{plan['profile']['text_prefix']}.layers.{layer}.self_attn.{hf}.lora_{side}.weight"]
    if set(actual) != set(expected) or len(factors) != len(expected):
        raise ValueError('GGUF factor set differs from expected text factors')
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])
    if validate(probe) != plan:
        raise ValueError('experiment changed during conversion')
    verify_tree(request['converter'])
    write_durable(folder / 'result.json', {'completed': True, 'synthetic_text_only': True,
        'factor_count': len(actual), 'all_factors_bit_equal': True,
        'adapter_sha256': sha256_file(folder / 'adapter.gguf'),
        'native_evaluation_performed': False, 'inference_gguf_provenance_verified': False,
        'adoption_authorized': False})


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('probe', 'converter-manifest', 'training-python', 'output', 'worker'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker.resolve())
    if not all((args.probe, args.converter_manifest, args.training_python, args.output)):
        parser.error('--probe, --converter-manifest, --training-python and --output are required')
    validate(args.probe.resolve())
    reference = {'path': str(args.converter_manifest.resolve()), 'sha256': sha256_file(args.converter_manifest)}
    verify_tree(reference)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_durable(output / 'input.json', {'probe': str(args.probe.resolve()), 'converter': reference})
    process = run_cpu_worker(args.training_python, [__file__, '--worker', str(output)], cwd=ROOT,
        log=output / 'worker.log', limits=WorkerLimits(1536 * 1024**2, 180))
    write_durable(output / 'process.json', process)
    print(json.dumps(process, indent=2))
    raise SystemExit(0 if process['succeeded'] else 1)


if __name__ == '__main__':
    main()
