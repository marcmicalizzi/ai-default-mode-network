"""Disposable 31B length probes using production NF4 loading and target-only loss.

No instance is opened, no recipe limit is raised, and no adapter is adopted.
Default execution only prints the synthetic research plan; --execute uses GPU 0.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dmn.backend import sha256_file
from dmn.storage import json_text, write_durable
from dmn.worker_limits import WorkerLimits, run_monitored_gpu_worker
from scripts.probe_qlora_31b import inspect


def plan(source, length, budget, loss_mode='full', compare_full=False):
    if length not in (256, 512, 1024, 1536, 2048) or type(budget) is not int or not 16384 <= budget <= 24576:
        raise ValueError('research supports 256/512/1024/1536/2048 tokens and 16..24 GiB Torch ceilings')
    if loss_mode not in ('full', 'chunked') or type(compare_full) is not bool:
        raise ValueError('unknown research loss or comparison')
    if compare_full and (loss_mode != 'chunked' or length > 512):
        raise ValueError('full-gradient comparison is bounded to chunked runs up to 512 tokens')
    return {'probe': inspect(source, sequence_tokens=length, steps=2), 'torch_vram_mib': budget,
            'loss': 'production_shifted_target_only_v1' if loss_mode == 'full' else 'research_chunked_target_only_v1',
            'loss_mode': loss_mode, 'output_chunk_tokens': 64 if loss_mode == 'chunked' else None,
            'compare_full_gradients': compare_full, 'masked_prefix_tokens': length // 4,
            'deployment_scale': .1, 'synthetic_only': True, 'adoption_authorized': False,
            'implementation': {str(p.relative_to(ROOT)): sha256_file(p) for p in (
                Path(__file__), ROOT/'dmn/gpu_training_worker.py', ROOT/'dmn/qlora_prepare.py',
                ROOT/'dmn/safetensor_stream.py', ROOT/'dmn/training_artifacts.py',
                ROOT/'scripts/probe_qlora_31b.py', ROOT/'scripts/chunked_training_loss.py')}}


def worker(folder):
    request = json.loads((folder/'input.json').read_text())
    research = request['plan']
    probe = research['probe']
    if research != plan(Path(probe['source']), probe['training']['sequence_tokens'], research['torch_vram_mib'],
                        research['loss_mode'], research['compare_full_gradients']):
        raise ValueError('research plan or implementation changed')
    if os.environ.get('DMN_GPU_PROBE_CONTAINED') != '1' or os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('explicit contained GPU worker required')
    started = time.monotonic()
    def phase(name, **values):
        value = {'phase': name, 'worker_pid': os.getpid(), 'seconds': time.monotonic()-started, **values}
        write_durable(folder/'progress.json', value)
        print(json.dumps(value), flush=True)
    source = Path(probe['source'])
    phase('verifying_source')
    for name, item in json.loads((source/'source.json').read_text())['files'].items():
        if name.endswith('.safetensors') and sha256_file(source/'model'/name) != item['lfs_sha256']:
            raise ValueError('pinned source hash mismatch')
    from dmn.gpu_training_worker import load_model, loss_for as full_loss, frozen_hashes, set_scale, check_placement
    from scripts.chunked_training_loss import loss_for as chunked_loss, compare_gradients
    loss_for = full_loss if research['loss_mode'] == 'full' else chunked_loss

    def evaluate(model, rows):
        model.eval()
        with torch.no_grad():
            values = [float(loss_for(model, row)) for row in rows]
        if not all(math.isfinite(v) for v in values):
            raise ValueError('nonfinite selected-example loss')
        return values
    from dmn.training_models import factor_names
    from dmn.training_artifacts import save_adapter
    compiled_shape = {'recipe': {'trainer': {'seed': 814}}, 'training': {
        'torch_vram_bytes': research['torch_vram_mib'] * 1024**2,
        'model_profile': probe['profile'], 'device_map': probe['device_map']}}
    phase('loading_nf4')
    model, staged = load_model(compiled_shape, source/'model', training=request['phase']=='train')
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
    from transformers import PreTrainedTokenizerFast
    from safetensors.torch import load_file
    tokenizer = PreTrainedTokenizerFast.from_pretrained(source/'model', local_files_only=True)
    reference = tokenizer.encode('Synthetic training mechanics: red, green, blue. This is a disposable test.', add_special_tokens=True)
    length = probe['training']['sequence_tokens']
    ids = [reference[0]] + (reference[1:] * (length // (len(reference)-1)+1))[:length-1]
    row = {'tokens': ids, 'labels': [-100]*research['masked_prefix_tokens'] + ids[research['masked_prefix_tokens']:]}
    example_hash = hashlib.sha256(json_text(row).encode()).hexdigest()
    if request['phase'] == 'reload':
        original_folder = Path(request['probe'])
        original = json.loads((original_folder/'result.json').read_text())
        if (original['research'] != research or original['example_sha256'] != example_hash or
                original['adapter_sha256'] != sha256_file(original_folder/'adapter/adapter_model.safetensors') or
                original['adapter_config_sha256'] != sha256_file(original_folder/'adapter/adapter_config.json')):
            raise ValueError('trained experiment or adapter changed')
        phase('loading_adapter')
        model = PeftModel.from_pretrained(model, original_folder/'adapter', local_files_only=True, is_trainable=False)
        check_placement(model)
        stored = load_file(original_folder/'adapter/adapter_model.safetensors')
        loaded = get_peft_model_state_dict(model)
        if stored.keys() != loaded.keys() or any(not torch.equal(stored[n], t.detach().cpu()) for n,t in loaded.items()):
            raise ValueError('reload changed adapter factors')
        phase('evaluating_reload')
        set_scale(model, 1.)
        training_loss = evaluate(model, [row])[0]
        set_scale(model, research['deployment_scale'])
        deployment_loss = evaluate(model, [row])[0]
        if training_loss != original['loss_after_training_scale'] or deployment_loss != original['loss_after_deployment_scale']:
            raise ValueError('fresh-process reload changed selected-example loss')
        write_durable(folder/'result.json', {'completed': True, 'fresh_process': True, 'factors_exact': True,
            'selected_losses_exact': True, 'training_loss': training_loss, 'deployment_loss': deployment_loss,
            'peak_torch_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_torch_reserved_bytes': torch.cuda.max_memory_reserved(), 'adoption_authorized': False})
        return
    model = get_peft_model(model, LoraConfig(task_type='CAUSAL_LM', r=2, lora_alpha=4,
        target_modules=probe['profile']['target_modules'], lora_dropout=0., bias='none', init_lora_weights=True))
    check_placement(model)
    params = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if {n for n,_ in params} != factor_names(probe['profile']):
        raise ValueError('unexpected trainable tensor set')
    initial = {n:p.detach().cpu().clone() for n,p in params}
    phase('hashing_frozen_base')
    frozen = frozen_hashes(model)
    phase('evaluating_baseline')
    before = evaluate(model, [row])[0]
    optimizer = torch.optim.AdamW([p for _,p in params], lr=.0001, betas=(.9,.999), eps=1e-8, weight_decay=0.)
    model.train()
    updates = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        phase('forward', step=step+1)
        loss = loss_for(model, row)
        if not torch.isfinite(loss):
            raise ValueError('nonfinite training loss')
        phase('backward', step=step+1)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([p for _,p in params], 1., error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        updates.append({'step': step+1, 'loss': float(loss.detach()), 'gradient_norm': float(norm)})
        del loss, norm
        phase('updated', step=step+1, peak_torch_allocated_bytes=torch.cuda.max_memory_allocated())
    changed = sum(not torch.equal(initial[n], p.detach().cpu()) for n,p in params)
    if not changed or any(not torch.isfinite(p).all() for _,p in params):
        raise ValueError('empty or nonfinite adapter update')
    del optimizer, initial
    for _,p in params:
        p.grad = None
    peak_before_comparison = torch.cuda.max_memory_allocated()
    comparison = None
    if research['compare_full_gradients']:
        phase('comparing_full_gradients')
        comparison = compare_gradients(model, row, full_loss)
    phase('verifying_frozen_base')
    if frozen_hashes(model) != frozen:
        raise ValueError('frozen base changed')
    phase('evaluating_adapter')
    after = evaluate(model, [row])[0]
    set_scale(model, research['deployment_scale'])
    deployed = evaluate(model, [row])[0]
    set_scale(model, 1.)
    save_adapter(model, folder/'adapter', factor_names(probe['profile'], saved=True))
    write_durable(folder/'result.json', {'completed': True, 'plan': probe, 'research': research,
        'steps_completed': 2, 'sequence_tokens': length, 'example_sha256': example_hash,
        'loss_before': before, 'loss_after_training_scale': after, 'loss_after_deployment_scale': deployed,
        'updates': updates, 'changed_factors': changed, 'frozen_state_unchanged': True,
        'cpu_staged_f32_casts': staged, 'peak_torch_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_allocated_before_reference_comparison_bytes': peak_before_comparison,
        'full_gradient_comparison': comparison,
        'peak_torch_reserved_bytes': torch.cuda.max_memory_reserved(), 'seconds': time.monotonic()-started,
        'packages': {n: importlib.metadata.version(n) for n in ('torch','transformers','peft','bitsandbytes','accelerate')},
        'adapter_sha256': sha256_file(folder/'adapter/adapter_model.safetensors'),
        'adapter_config_sha256': sha256_file(folder/'adapter/adapter_config.json'), 'adoption_authorized': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('train','reload'), default='train')
    for name in ('source','output','training-python','probe','worker'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--tokens', type=int, choices=(256,512,1024,1536,2048), default=256)
    parser.add_argument('--loss', choices=('full','chunked'), default='full')
    parser.add_argument('--compare-full-gradients', action='store_true')
    parser.add_argument('--torch-vram-mib', type=int, default=22528)
    parser.add_argument('--max-seconds', type=int, default=3600)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.worker:
        try:
            worker(args.worker)
        except Exception as exc:
            write_durable(args.worker/'failure.json', {'type': type(exc).__name__, 'reason': str(exc)[:6000]})
            raise
        return
    if args.phase == 'reload':
        if not args.probe:
            parser.error('reload requires --probe')
        result = json.loads((args.probe/'result.json').read_text())
        if not result.get('completed'):
            parser.error('reload requires completed training')
        research = result['research']
    else:
        if not args.source:
            parser.error('training requires --source')
        research = plan(args.source, args.tokens, args.torch_vram_mib, args.loss, args.compare_full_gradients)
    request = {'plan': research, 'phase': args.phase, 'probe': str(args.probe.resolve()) if args.probe else None}
    if not args.execute:
        print(json.dumps(request, indent=2))
        return
    if not args.output or not args.training_python:
        parser.error('execution requires fresh --output and --training-python')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    write_durable(args.output/'input.json', request)
    process = run_monitored_gpu_worker(args.training_python, [__file__, '--worker', str(args.output)],
        cwd=ROOT, log=args.output/'worker.log', limits=WorkerLimits(32*1024**3, args.max_seconds),
        max_device_bytes=31*1024**3)
    write_durable(args.output/'process.json', process)
    print(json.dumps(process, indent=2))
    raise SystemExit(0 if process['succeeded'] else 1)


if __name__ == '__main__':
    main()
