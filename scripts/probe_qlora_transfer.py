"""Transfer a completed tiny NF4 experiment to native Q4_K_M inference, CPU only.

Uses only synthetic fixtures and creates no Runtime or live-instance candidate.
This is a mechanics test, not an approved learning recipe or numerical parity
claim between NF4 training and llama.cpp quantization.
"""
import argparse
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dmn.backend import sha256_file
from dmn.storage import write_durable
from dmn.worker_limits import WorkerLimits, run_cpu_worker
from scripts.probe_qlora import inspect_fixture


def inputs(probe, proof):
    from dmn.base_provenance import verify
    report = json.loads((probe / 'result.json').read_text())
    if not report.get('completed') or report.get('gpu_execution') is not True or not report.get('synthetic_only'):
        raise ValueError('requires a completed synthetic GPU experiment')
    plan = inspect_fixture(report['plan']['source'])
    if plan != report['plan']:
        raise ValueError('GPU source or declared training plan changed')
    reference = {'path': str(proof), 'sha256': sha256_file(proof)}
    provenance = json.loads(proof.read_text())
    verify(reference, provenance['request']['conversion'], provenance['artifacts']['base.gguf'])
    base_manifest = json.loads(Path(provenance['request']['conversion']['base_manifest']['path']).read_text())
    if Path(base_manifest['root']).resolve() != (Path(plan['source']) / 'base').resolve():
        raise ValueError('GPU training source differs from proven inference source')
    if provenance['request']['quantization'] != 'Q4_K_M':
        raise ValueError('this transfer experiment requires Q4_K_M')
    return plan, provenance


def native(folder, restart=False):
    import numpy as np
    from dmn.adapters import AdapterSpec
    from dmn.backend import LlamaBackend
    from dmn.config import Config
    from dmn.recovery import restore_checkpoint
    from scripts.probe_lora_wake import save, continuation
    data = json.loads((folder / 'input.json').read_text())
    model = Path(data['proof']).parent / 'base.gguf'
    adapter = folder / 'adapter.gguf'
    spec = AdapterSpec(str(adapter), sha256_file(adapter), sha256_file(model), .1)
    config = Config(model_path=str(model), lora_adapters=(spec,), n_ctx=2048,
        n_batch=64, n_threads=1, n_gpu_layers=0, offload_kqv=False, flash_attn=True,
        type_k='q8_0', type_v='q8_0', swa_full=False, experimental_compact_swa=True,
        pack_checkpoints=True, prompt_format='plain', turnover_reserve=512)
    if model.stat().st_size > 4 * 1024**2:
        raise ValueError('only tiny native fixtures are permitted')
    backend = LlamaBackend(config if restart else dataclasses.replace(config, lora_adapters=()))
    try:
        if restart:
            _, evidence = restore_checkpoint(backend, folder / 'after')
            tokens, logits = continuation(backend)
            assert tokens == json.loads((folder / 'continuation.json').read_text())
            np.testing.assert_array_equal(logits, np.load(folder / 'continuation.npy', allow_pickle=False))
            write_durable(folder / 'restart.json', {'zero_replay': evidence['prompt_tokens_reevaluated'] == 0,
                'eight_tokens_and_logits_equal': True, 'fresh_process': True})
            return
        backend.eval(backend.tokenize('Synthetic NF4 adapter transfer. No instance messages or actions. ' * 4, initial=True))
        backend.shift(16, 80)
        backend.eval([11])
        retained, rng = list(backend.tokens), backend.rng.getstate()
        save(backend, folder / 'before', 1)
    finally:
        backend.close()
    backend = LlamaBackend(config)
    try:
        try:
            restore_checkpoint(backend, folder / 'before')
        except ValueError:
            pass
        else:
            raise AssertionError('ordinary restore accepted changed weights')
        if backend.tokens or backend.decode_calls:
            raise AssertionError('rejection evaluated tokens')
        evidence = backend.rebuild(folder / 'before')
        assert backend.tokens == retained and backend.rng.getstate() == rng
        save(backend, folder / 'after', 1)
        tokens, logits = continuation(backend)
        write_durable(folder / 'continuation.json', tokens)
        np.save(folder / 'continuation.npy', logits, allow_pickle=False)
    finally:
        backend.close()
    subprocess.run([sys.executable, __file__, '--restart', str(folder)], check=True, stdin=subprocess.DEVNULL)
    write_durable(folder / 'native.json', {'retained_tokens': len(retained), 'tokens_and_rng_preserved': True,
        'old_weights_strict_restore_rejected': True, 'wake': evidence,
        'restart': json.loads((folder / 'restart.json').read_text())})


def worker(folder):
    from dmn.training import verify_tree
    import numpy as np
    from safetensors.numpy import load_file
    data = json.loads((folder / 'input.json').read_text())
    probe, proof = Path(data['probe']), Path(data['proof'])
    plan, provenance = inputs(probe, proof)
    converter = verify_tree(provenance['request']['conversion']['converter_manifest'])
    subprocess.run([sys.executable, str(converter / 'convert_lora_to_gguf.py'), '--outtype', 'f32',
        '--base', str(Path(plan['source']) / 'base'), '--outfile', str(folder / 'adapter.gguf'),
        str(probe / 'adapter')], check=True, stdin=subprocess.DEVNULL)
    sys.path.insert(0, str(converter / 'gguf-py'))
    import gguf
    factors = load_file(probe / 'adapter/adapter_model.safetensors')
    actual = {t.name: t.data for t in gguf.GGUFReader(folder / 'adapter.gguf').tensors}
    expected = {}
    for layer in range(plan['profile']['num_hidden_layers']):
        for hf, cpp in (('q_proj', 'attn_q'), ('o_proj', 'attn_output')):
            for side in ('A', 'B'):
                expected[f'blk.{layer}.{cpp}.weight.lora_{side.lower()}'] = factors[
                    f"base_model.model.{plan['profile']['text_prefix']}.layers.{layer}.self_attn.{hf}.lora_{side}.weight"]
    assert set(actual) == set(expected) and len(factors) == len(expected)
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])
    subprocess.run([data['native_python'], __file__, '--native', str(folder)], check=True, stdin=subprocess.DEVNULL)
    inputs(probe, proof)
    write_durable(folder / 'result.json', {'completed': True, 'synthetic_only': True,
        'all_factors_bit_equal': True, 'factor_count': len(actual), 'adapter_sha256': sha256_file(folder / 'adapter.gguf'),
        'native': json.loads((folder / 'native.json').read_text()), 'beneficial_learning_certified': False})


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('probe', 'proof', 'output', 'native-python', 'worker', 'native', 'restart'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    if args.restart or args.native:
        return native((args.restart or args.native).resolve(), restart=bool(args.restart))
    if args.worker:
        return worker(args.worker.resolve())
    if not all((args.probe, args.proof, args.output, args.native_python)):
        parser.error('--probe, --proof, --output and --native-python are required')
    _, provenance = inputs(args.probe.resolve(), args.proof.resolve())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_durable(output / 'input.json', {'probe': str(args.probe.resolve()), 'proof': str(args.proof.resolve()),
        'native_python': str(args.native_python.resolve())})
    result = run_cpu_worker(provenance['request']['conversion']['python'], [__file__, '--worker', str(output)],
        cwd=ROOT, log=output / 'worker.log', limits=WorkerLimits(1536 * 1024**2, 180))
    write_durable(output / 'process.json', result)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['succeeded'] else 1)


if __name__ == '__main__':
    main()
