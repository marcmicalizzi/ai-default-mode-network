"""Disposable 31B GPU wake/restart and vision checks; never opens an instance.

Runs separate contained processes and fixed synthetic inputs. The published base
hash is an identity check, not proof of its relationship to training safetensors.
"""
import argparse
import base64
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dmn.adapters import AdapterSpec
from dmn.backend import LlamaBackend, sha256_file, tuples
from dmn.config import Config
from dmn.recovery import restore_checkpoint
from dmn.storage import write_durable
from dmn.worker_limits import WorkerLimits, run_gpu_research_worker
from scripts.probe_lora_wake import save, continuation

BASE_SHA = '7c65a35e7c4e53cba6c5e02cc9eeb850eb4251f4d9ad120c2caa6de23c5a6395'
PROJECTOR_SHA = '21487ff26d08f7ddd1d654d3bbfc1ae1020aab3119f5bf654742ce4697732e4e'
PHASES = ('base', 'wake', 'restart', 'vision', 'vision-restart')


def worker(root, phase):
    if os.environ.get('DMN_GPU_PROBE_CONTAINED') != '1' or os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('requires the explicitly contained GPU launcher')
    import numpy as np
    request = json.loads((root / 'input.json').read_text())
    config = Config.read(root / 'config.local.json')
    if phase == 'base':
        config = replace(config, lora_adapters=())
    if phase.startswith('vision'):
        # Borrow only Pillow from a separate existing environment, without
        # changing the running inference environment's installed packages.
        sys.path.append(request['pillow_site'])
        config = replace(config, vision_projector_path=request['projector'], lora_adapters=())
        if sha256_file(Path(request['projector'])) != PROJECTOR_SHA:
            raise ValueError('projector identity differs from the reviewed fixture')
    write_durable(root / 'progress.json', {'phase': phase, 'stage': 'loading'})
    backend = LlamaBackend(config)
    try:
        if backend.fingerprint['model_sha256'] != BASE_SHA:
            raise ValueError('unexpected base model identity')
        report = {'phase': phase, 'completed': False}
        write_durable(root / 'progress.json', {'phase': phase, 'stage': 'loaded'})
        if phase == 'base':
            unit = backend.tokenize('Synthetic cache mechanics. Red green blue. ')
            tokens = backend.tokenize('Disposable validation context. ', initial=True)
            tokens += (unit * (3072 // len(unit) + 1))[:3072 - len(tokens)]
            backend.eval(tokens)
            backend.shift(32, 1024)
            backend.eval(backend.tokenize(' Context retirement completed. '))
            save(backend, root / 'before', 1)
            report.update(retained_tokens=len(backend.tokens), retirement_window=backend.retirement_window)
        elif phase == 'wake':
            rejections = {}
            for policy in ('strict', 'rebuild'):
                try:
                    restore_checkpoint(backend, root / 'before', policy)
                except ValueError as exc:
                    if ('environment differs' if policy == 'strict' else 'adapter identity') not in str(exc):
                        raise
                    rejections[policy] = str(exc)
                else:
                    raise AssertionError('ordinary restore accepted different weights')
            assert backend.decode_calls == 0 and not backend.tokens
            started = time.monotonic()
            evidence = backend.rebuild(root / 'before')  # explicit experimental weight transition
            engine = json.loads((root / 'before/engine.json').read_text())
            assert backend.tokens == engine['tokens'] and backend.rng.getstate() == tuples(engine['rng'])
            save(backend, root / 'after', 1)
            report.update(rebuild=evidence, rebuild_and_checkpoint_seconds=time.monotonic() - started,
                          ordinary_restore_rejections=rejections)
            chosen, logits = continuation(backend, 16)
            np.save(root / 'continuation.npy', logits, allow_pickle=False)
            write_durable(root / 'continuation.json', chosen)
        elif phase == 'restart':
            _, evidence = restore_checkpoint(backend, root / 'after')
            assert evidence['decode_calls_during_load'] == 0 and evidence['prompt_tokens_reevaluated'] == 0
            chosen, logits = continuation(backend, 16)
            assert chosen == json.loads((root / 'continuation.json').read_text())
            expected = np.load(root / 'continuation.npy', allow_pickle=False)
            np.testing.assert_array_equal(logits, expected)
            report.update(restore=evidence, continuation_tokens_equal=True, continuation_logits_bit_equal=True)
        elif phase == 'vision':
            from PIL import Image, ImageDraw
            from dmn.attachments import decode_uploads
            picture = Image.new('RGB', (128, 128), 'red')
            ImageDraw.Draw(picture).rectangle((64, 0, 127, 127), fill='blue')
            stream = io.BytesIO()
            picture.save(stream, format='PNG')
            picture.close()
            images = decode_uploads([{'media_type': 'image/png', 'data_base64': base64.b64encode(stream.getvalue()).decode()}])
            backend.eval(backend.tokenize('Synthetic image test. ', initial=True))
            keep = len(backend.tokens)
            started = time.monotonic()
            with backend.vision.prepare(images) as prepared:
                positions = prepared.positions
                visual_count = prepared.slots.count(-1)
                prepared.evaluate()
            image_seconds = time.monotonic() - started
            unit = backend.tokenize(' Synthetic text after the image. ')
            suffix = (unit * (backend.retirement_window // len(unit) + 2))[:backend.retirement_window + 16]
            backend.eval(suffix)
            assert -1 in backend.tokens
            save(backend, root / 'visual', 0)
            chosen, logits = continuation(backend, 8)
            np.save(root / 'visual-continuation.npy', logits, allow_pickle=False)
            write_durable(root / 'visual-continuation.json', chosen)
            report.update(image_input_positions=positions, visual_positions=visual_count,
                          image_decode_seconds=image_seconds, keep_prefix=keep,
                          vision_fingerprint=backend.vision.fingerprint)
        elif phase == 'vision-restart':
            _, evidence = restore_checkpoint(backend, root / 'visual')
            assert evidence['decode_calls_during_load'] == 0 and evidence['prompt_tokens_reevaluated'] == 0
            before = backend.decode_calls
            try:
                backend.rebuild(root / 'visual')
            except ValueError as exc:
                if 'visual positions' not in str(exc):
                    raise
            else:
                raise AssertionError('text rebuild accepted retained image positions')
            assert backend.decode_calls == before
            chosen, logits = continuation(backend, 8)
            assert chosen == json.loads((root / 'visual-continuation.json').read_text())
            np.testing.assert_array_equal(logits, np.load(root / 'visual-continuation.npy', allow_pickle=False))
            keep = json.loads((root / 'vision-result.json').read_text())['keep_prefix']
            last = max(i for i, token in enumerate(backend.tokens) if token == -1)
            backend.shift(keep, last - keep + 1)
            backend.eval(backend.tokenize(' Image retired. '))
            assert -1 not in backend.tokens
            save(backend, root / 'visual-retired', 1)
            rebuilt = backend.rebuild(root / 'visual-retired')
            assert -1 not in backend.tokens
            engine = json.loads((root / 'visual/engine.json').read_text())
            assert not {'images', 'pixels', 'embeddings'} & engine.keys()
            report.update(restore=evidence, continuation_tokens_equal=True, continuation_logits_bit_equal=True,
                          retained_image_rebuild_rejected=True, retired_image_rebuild=rebuilt, raw_pixel_archive=False)
        report.update(completed=True, fingerprint=backend.fingerprint, runtime_constructed=False,
                      adoption_authorized=False, inference_gguf_provenance_verified=False)
        write_durable(root / (phase + '-result.json'), report)
    finally:
        backend.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'adapter', 'projector', 'native-python', 'pillow-site', 'output', 'worker'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--phase', choices=PHASES)
    args = parser.parse_args()
    if args.worker:
        try:
            return worker(args.worker, args.phase)
        except Exception as exc:
            write_durable(args.worker / (args.phase + '-failure.json'), {'type': type(exc).__name__, 'reason': str(exc)[:4000]})
            raise
    if not all((args.model, args.adapter, args.projector, args.native_python, args.pillow_site, args.output)):
        parser.error('all asset paths and a fresh output directory are required')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    config = Config(model_path=str(args.model.resolve()),
        lora_adapters=(AdapterSpec(str(args.adapter.resolve()), sha256_file(args.adapter), BASE_SHA, .1),),
        n_ctx=4096, n_batch=256, n_threads=4, n_gpu_layers=99, flash_attn=True,
        type_k='q8_0', type_v='q8_0', swa_full=False, experimental_compact_swa=True,
        pack_checkpoints=True, prompt_format='plain')
    write_durable(root / 'config.local.json', config.to_dict())
    write_durable(root / 'input.json', {'projector': str(args.projector.resolve()), 'pillow_site': str(args.pillow_site.resolve())})
    for phase in PHASES:
        process = run_gpu_research_worker(args.native_python, [__file__, '--worker', str(root), '--phase', phase],
            cwd=ROOT, log=root / (phase + '.log'), limits=WorkerLimits(32768 * 1024**2, 1200), allow_gpu=True)
        write_durable(root / (phase + '-process.json'), process)
        print(json.dumps({'phase': phase, **process}), flush=True)
        if not process['succeeded']:
            raise SystemExit(1)
    write_durable(root / 'result.json', {'completed': True, 'phases': list(PHASES),
        'synthetic_only': True, 'runtime_constructed': False, 'adoption_authorized': False})


if __name__ == '__main__':
    main()
