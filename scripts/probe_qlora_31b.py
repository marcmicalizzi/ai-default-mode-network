"""Explicit bounded NF4 feasibility test on a pinned 31B source, never an instance.

Default inspection performs no CUDA work. Output is a disposable synthetic-text
adapter, not a candidate for Syllas. No claim of equivalence to its published
inference GGUF or approval to adopt any trained artifact is made here.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dmn.backend import sha256_file
from dmn.storage import write_durable
from dmn.training_models import model_profile, factor_names
from dmn.worker_limits import WorkerLimits, run_gpu_research_worker
from scripts.qlora_prepare import ALLOCATOR, configure_allocator

REPO = 'llmfan46/gemma-4-31B-it-uncensored-heretic'
REVISION = 'd5bfc0d99e308beb9805440806161ad0233df357'


def inspect(source, *, sequence_tokens=None, steps=1):
    if sequence_tokens is not None and (type(sequence_tokens) is not int or sequence_tokens not in (128, 256, 512, 1024)):
        raise ValueError('workload length must be 128, 256, 512 or 1024 synthetic tokens')
    if type(steps) is not int or not 1 <= steps <= 4:
        raise ValueError('research workload permits one to four steps')
    source = source.resolve()
    record = json.loads((source / 'source.json').read_text())
    if record.get('repo') != REPO or record.get('revision') != REVISION:
        raise ValueError('this experiment requires the explicitly pinned source')
    if record['bytes'] > 64 * 1024**3 or sum(v['size'] for v in record['files'].values()) != record['bytes']:
        raise ValueError('source exceeds the fixed research envelope')
    for name, entry in record['files'].items():
        if Path(name).name != name or (source / 'model' / name).stat().st_size != entry['size']:
            raise ValueError('missing source asset or size mismatch')
    base = source / 'model'
    profile = model_profile(base, wrapped=True)
    config = json.loads((base / 'config.json').read_text())
    if profile['num_hidden_layers'] != 60 or config['text_config']['hidden_size'] != 5376 or profile['vocab_size'] != 262144:
        raise ValueError('unexpected 31B model geometry')
    plan = {'repo': REPO, 'revision': REVISION, 'source': str(source), 'profile': profile,
        'source_metadata_sha256': sha256_file(source / 'source.json'),
        'config_sha256': sha256_file(base / 'config.json'), 'synthetic_text_only': True,
        'gpu_execution': False, 'training': {'steps': 1, 'rank': 2, 'alpha': 4, 'sequence_limit': 32,
        'quantization': 'NF4', 'nested_quantization': True, 'compute_dtype': 'bfloat16',
        'learning_rate': .0001, 'optimizer': 'AdamW', 'gradient_checkpointing': 'non-reentrant'},
        'device_map': {'': 0, 'model.vision_tower': 'cpu', 'model.embed_vision': 'cpu'},
        'source_loader': 'tensor_stream_v1',
        'torch_allocator_configuration': ALLOCATOR,
        'preparation': 'static_placement; CPU-staged large F32 casts; standard PEFT frozen-base preparation',
        'total_process_vram_quota_enforced': False, 'inference_gguf_provenance_verified': False}
    plan['training']['steps'] = steps
    if sequence_tokens is not None:
        plan['training'].update(sequence_tokens=sequence_tokens, sequence_limit=sequence_tokens)
    return plan


def execute(folder):
    request = json.loads((folder / 'input.json').read_text())
    workload = request['plan']['training']
    plan = inspect(Path(request['plan']['source']), sequence_tokens=workload.get('sequence_tokens'), steps=workload['steps'])
    if plan != request['plan'] or os.environ.get('DMN_GPU_PROBE_CONTAINED') != '1' or os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('requires unchanged plan and explicit contained GPU launcher')
    source = Path(plan['source'])
    record = json.loads((source / 'source.json').read_text())
    write_durable(folder / 'progress.json', {'phase': 'verifying_source'})
    for name, entry in record['files'].items():
        if name.endswith('.safetensors'):
            print('Verifying source weight hash: ' + name, flush=True)
            if sha256_file(source / 'model' / name) != entry['lfs_sha256']:
                raise ValueError('source weight hash mismatch')
    configure_allocator()
    import torch
    import bitsandbytes as bnb
    import psutil
    from transformers import BitsAndBytesConfig, Gemma4Config, Gemma4ForConditionalGeneration, PreTrainedTokenizerFast
    from scripts.safetensor_stream import state_dict
    from peft import LoraConfig, get_peft_model
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(814)
    budget = request['torch_vram_mib'] * 1024**2
    free, total = torch.cuda.mem_get_info(0)
    if not 16 * 1024**3 <= budget <= 24 * 1024**3 or free < budget + 512 * 1024**2:
        raise ValueError('insufficient free VRAM for the requested research allocator budget')
    torch.cuda.set_per_process_memory_fraction(budget / total, 0)
    torch.cuda.reset_peak_memory_stats(0)
    stages = []
    started = time.monotonic()

    def stage(name):
        item = {'name': name, 'seconds': time.monotonic() - started,
            'torch_allocated_bytes': torch.cuda.memory_allocated(0), 'torch_reserved_bytes': torch.cuda.memory_reserved(0),
            'peak_torch_allocated_bytes': torch.cuda.max_memory_allocated(0),
            'inactive_split_bytes': torch.cuda.memory_stats(0).get('inactive_split_bytes.all.current', 0),
            'rss_bytes': psutil.Process().memory_info().rss}
        stages.append(item)
        write_durable(folder / 'progress.json', {'stages': stages})
        print(json.dumps(item), flush=True)

    stage('loading')
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_enable_fp32_cpu_offload=True,
        llm_int8_skip_modules=['lm_head', 'model.vision_tower', 'model.embed_vision'])
    model = Gemma4ForConditionalGeneration.from_pretrained(None,
        config=Gemma4Config.from_pretrained(source / 'model', local_files_only=True), state_dict=state_dict(source / 'model'),
        dtype=torch.bfloat16, quantization_config=quantization, device_map=plan['device_map'], attn_implementation='eager')
    stage('loaded_nf4')
    quantized = [n for n, m in model.named_modules() if isinstance(m, bnb.nn.Linear4bit)]
    if not quantized or any(not n.startswith('model.language_model.layers.') for n in quantized):
        raise ValueError('unexpected quantized module set')
    from scripts.qlora_prepare import prepare
    model, staged_casts = prepare(model)
    model = get_peft_model(model, LoraConfig(task_type='CAUSAL_LM', r=2, lora_alpha=4,
        target_modules=plan['profile']['target_modules'], lora_dropout=0., bias='none'))
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if {n for n, p in trainable} != factor_names(plan['profile']):
        raise ValueError('unexpected trainable tensor set')
    if any(p.device.type != 'cpu' for n, p in model.named_parameters() if '.vision_tower.' in n or '.embed_vision.' in n):
        raise ValueError('frozen vision components were not kept on CPU')
    stage('prepared_adapter')

    def frozen_hashes():
        result = {}
        for name, tensor in model.state_dict().items():
            if 'lora_' in name:
                continue
            digest = hashlib.sha256()
            for chunk in tensor.detach().reshape(-1).split(1024**2):
                digest.update(chunk.contiguous().cpu().view(torch.uint8).numpy().tobytes())
            result[name] = digest.hexdigest()
        return result

    before_frozen = frozen_hashes()
    initial = {n: p.detach().cpu().clone() for n, p in trainable}
    tokenizer = PreTrainedTokenizerFast.from_pretrained(source / 'model', local_files_only=True)
    ids = tokenizer.encode('Synthetic training mechanics: red, green, blue. This is a disposable test.', add_special_tokens=True)
    reference_ids = list(ids)
    if workload.get('sequence_tokens'):
        size = workload['sequence_tokens']
        ids = [ids[0]] + (ids[1:] * (size // (len(ids) - 1) + 1))[:size - 1]
    if not 2 <= len(ids) <= workload['sequence_limit']:
        raise ValueError('synthetic sequence exceeds the reviewed experiment length')
    tokens = torch.tensor([ids], device='cuda:0')
    optimizer = torch.optim.AdamW([p for n, p in trainable], lr=.0001, weight_decay=0.)
    model.train()
    updates = []
    for step in range(workload['steps']):
        optimizer.zero_grad(set_to_none=True)
        stage(f'forward_{step + 1}')
        loss = model(tokens, labels=tokens, use_cache=False).loss
        if not torch.isfinite(loss):
            raise ValueError('nonfinite forward loss')
        stage(f'backward_{step + 1}')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([p for n, p in trainable], 1., error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        updates.append({'step': step + 1, 'loss': float(loss.detach()), 'gradient_norm': float(norm)})
        del loss, norm
        stage(f'updated_{step + 1}')
    changed = [n for n, p in trainable if not torch.equal(initial[n], p.detach().cpu())]
    if not changed or any(not torch.isfinite(p).all() for n, p in trainable):
        raise ValueError('adapter update is empty or nonfinite')
    del optimizer, initial
    for _, parameter in trainable:
        parameter.grad = None
    model.eval()
    reference_tokens = torch.tensor([reference_ids], device='cuda:0')
    with torch.no_grad():
        reference_logits = model(reference_tokens, use_cache=False).logits[:, -1].float().cpu().numpy()
    import numpy as np
    np.save(folder / 'reload-logits.npy', reference_logits, allow_pickle=False)
    if frozen_hashes() != before_frozen:
        raise ValueError('frozen base or vision weights changed')
    model.save_pretrained(folder / 'adapter', safe_serialization=True, save_embedding_layers=False)
    stage('saved')
    if inspect(source, sequence_tokens=workload.get('sequence_tokens'), steps=workload['steps']) != plan:
        raise ValueError('source metadata changed during the experiment')
    write_durable(folder / 'result.json', {'completed': True, 'plan': plan, 'gpu_execution': True,
        'device': torch.cuda.get_device_name(0), 'packages': {n: importlib.metadata.version(n)
        for n in ('torch', 'transformers', 'peft', 'bitsandbytes', 'accelerate')}, 'stages': stages,
        'sequence_tokens': len(ids), 'steps_completed': len(updates), 'loss_before_update': updates[0]['loss'],
        'gradient_norm_before_clipping': updates[0]['gradient_norm'], 'updates': updates,
        'reload_reference': {'token_ids': reference_ids, 'logits_sha256': sha256_file(folder / 'reload-logits.npy')},
        'trainable_parameters': sum(p.numel() for n, p in trainable),
        'changed_factors': len(changed), 'frozen_state_unchanged': True, 'quantized_modules': len(quantized),
        'cpu_staged_f32_casts': staged_casts,
        'peak_torch_allocated_bytes': torch.cuda.max_memory_allocated(0), 'peak_torch_reserved_bytes': torch.cuda.max_memory_reserved(0),
        'total_process_vram_quota_enforced': False, 'beneficial_learning_certified': False,
        'adapter_sha256': sha256_file(folder / 'adapter/adapter_model.safetensors'),
        'inference_gguf_provenance_verified': False, 'adoption_authorized': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--training-python', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--torch-vram-mib', type=int, default=22528)
    parser.add_argument('--max-ram-mib', type=int, default=32768)
    parser.add_argument('--max-seconds', type=int, default=1200)
    parser.add_argument('--sequence-tokens', type=int, choices=(128, 256, 512, 1024))
    parser.add_argument('--steps', type=int, choices=(1, 2, 3, 4), default=1)
    args = parser.parse_args()
    if args.worker:
        try:
            return execute(args.worker)
        except Exception as exc:
            write_durable(args.worker / 'failure.json', {'type': type(exc).__name__, 'reason': str(exc)[:4000]})
            raise
    if not args.source:
        parser.error('--source is required')
    plan = inspect(args.source, sequence_tokens=args.sequence_tokens, steps=args.steps)
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    if not args.output or not args.training_python:
        parser.error('execution requires fresh --output and isolated --training-python')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_durable(output / 'input.json', {'plan': plan, 'torch_vram_mib': args.torch_vram_mib})
    result = run_gpu_research_worker(args.training_python, [__file__, '--worker', str(output)], cwd=ROOT,
        log=output / 'worker.log', limits=WorkerLimits(args.max_ram_mib * 1024**2, args.max_seconds), allow_gpu=True)
    write_durable(output / 'process.json', result)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['succeeded'] else 1)


if __name__ == '__main__':
    main()
