"""Fresh-process NF4/PEFT reload check for the disposable synthetic 31B probe."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dmn.backend import sha256_file
from dmn.storage import write_durable
from dmn.worker_limits import WorkerLimits, run_gpu_research_worker
from scripts.convert_qlora_31b_probe import validate
from scripts.qlora_prepare import configure_allocator, prepare


def worker(folder):
    request = json.loads((folder / 'input.json').read_text())
    if os.environ.get('DMN_GPU_PROBE_CONTAINED') != '1' or os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('requires the explicitly contained GPU launcher')
    probe = Path(request['probe'])
    plan = validate(probe)
    result = json.loads((probe / 'result.json').read_text())
    reference = result['reload_reference']
    if sha256_file(probe / 'reload-logits.npy') != reference['logits_sha256']:
        raise ValueError('reference logits changed')
    source = Path(plan['source'])
    write_durable(folder / 'progress.json', {'phase': 'verifying_source'})
    record = json.loads((source / 'source.json').read_text())
    for name, entry in record['files'].items():
        if name.endswith('.safetensors') and sha256_file(source / 'model' / name) != entry['lfs_sha256']:
            raise ValueError('source weight hash mismatch')
    configure_allocator()
    import numpy as np
    import torch
    from transformers import BitsAndBytesConfig, Gemma4Config, Gemma4ForConditionalGeneration
    from peft import PeftModel
    from scripts.safetensor_stream import state_dict
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(814)
    budget = 22528 * 1024**2
    free, total = torch.cuda.mem_get_info(0)
    if free < budget + 512 * 1024**2:
        raise ValueError('insufficient free VRAM for the research allocator budget')
    torch.cuda.set_per_process_memory_fraction(budget / total, 0)
    write_durable(folder / 'progress.json', {'phase': 'loading_nf4'})
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_enable_fp32_cpu_offload=True,
        llm_int8_skip_modules=['lm_head', 'model.vision_tower', 'model.embed_vision'])
    model = Gemma4ForConditionalGeneration.from_pretrained(None,
        config=Gemma4Config.from_pretrained(source / 'model', local_files_only=True),
        state_dict=state_dict(source / 'model'), dtype=torch.bfloat16, quantization_config=quantization,
        device_map=plan['device_map'], attn_implementation='eager')
    model, staged = prepare(model, gradient_checkpointing=False)
    write_durable(folder / 'progress.json', {'phase': 'loading_adapter'})
    model = PeftModel.from_pretrained(model, probe / 'adapter', local_files_only=True, is_trainable=False).eval()
    if (any(p.device.type == 'meta' for p in model.parameters())
            or any(p.device.type != 'cpu' for n, p in model.named_parameters() if '.vision_tower.' in n or '.embed_vision.' in n)):
        raise ValueError('adapter reload changed the explicit static placement')
    write_durable(folder / 'progress.json', {'phase': 'evaluating_reloaded_adapter'})
    with torch.no_grad():
        logits = model(torch.tensor([reference['token_ids']], device='cuda:0'), use_cache=False).logits[:, -1].float().cpu().numpy()
    expected = np.load(probe / 'reload-logits.npy', allow_pickle=False)
    np.testing.assert_allclose(logits, expected, rtol=0, atol=1e-5)
    if validate(probe) != plan:
        raise ValueError('source/adapter metadata changed during reload')
    write_durable(folder / 'result.json', {'completed': True, 'fresh_process': True,
        'synthetic_text_only': True, 'logits_bit_equal': bool(np.array_equal(logits, expected)),
        'maximum_absolute_logit_error': float(np.abs(logits - expected).max()),
        'reference_token_count': len(reference['token_ids']), 'cpu_staged_f32_casts': staged,
        'adapter_sha256': result['adapter_sha256'], 'reference_logits_sha256': reference['logits_sha256'],
        'peak_torch_allocated_bytes': torch.cuda.max_memory_allocated(0),
        'peak_torch_reserved_bytes': torch.cuda.max_memory_reserved(0),
        'total_process_vram_quota_enforced': False, 'adoption_authorized': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('probe', 'training-python', 'output', 'worker'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    if args.worker:
        try:
            return worker(args.worker.resolve())
        except Exception as exc:
            write_durable(args.worker / 'failure.json', {'type': type(exc).__name__, 'reason': str(exc)[:4000]})
            raise
    if not all((args.probe, args.training_python, args.output)):
        parser.error('--probe, --training-python and a fresh --output are required')
    validate(args.probe.resolve())
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    write_durable(folder / 'input.json', {'probe': str(args.probe.resolve())})
    result = run_gpu_research_worker(args.training_python, [__file__, '--worker', str(folder)], cwd=ROOT,
        log=folder / 'worker.log', limits=WorkerLimits(32768 * 1024**2, 1200), allow_gpu=True)
    write_durable(folder / 'process.json', result)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['succeeded'] else 1)


if __name__ == '__main__':
    main()
