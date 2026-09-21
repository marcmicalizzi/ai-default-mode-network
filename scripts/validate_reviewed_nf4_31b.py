"""Run the reviewed NF4 worker chain on explicit synthetic 31B examples only.

This does not open an instance, supply model consent, or install an adapter for
Syllas. Approval mechanics and native wake are separate integration checks.
"""
import argparse
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dmn.backend import sha256_file
from dmn.sleep_plans import seal, implementation_identity
from dmn.storage import write_durable
from dmn.worker_limits import WorkerLimits, run_cpu_worker


def prepare(folder):
    import importlib.metadata as metadata
    from transformers import PreTrainedTokenizerFast
    from dmn.gpu_recipe import PACKAGES, compile_training
    from dmn.training import GPU_KIND, CHECKS
    from dmn.exact_base_provenance import BASE_SHA
    from dmn.learning import HELP
    request = json.loads((folder / 'request.json').read_text())
    evidence = json.loads(Path(request['proof']).read_text())
    base = Path(json.loads(Path(evidence['base_manifest']['path']).read_text())['root'])
    tokenizer = PreTrainedTokenizerFast.from_pretrained(base, local_files_only=True)
    prefix = 'Disposable synthetic test record:\n'
    target = 'An external observation can be checked before it is incorporated into learning. '
    while len(tokenizer.encode(prefix + target + target, add_special_tokens=False)) <= 256:
        target += target
    tokens = tokenizer.encode(prefix + target, add_special_tokens=False)
    boundary = tokenizer.encode(prefix, add_special_tokens=False)
    if tokens[:len(boundary)] != boundary:
        raise ValueError('synthetic example crosses the loss boundary')
    examples = [{'input': prefix, 'target': target, 'sources': [], 'purpose': 'new', 'tokens': tokens,
        'loss_mask': [0] * len(boundary) + [1] * (len(tokens) - len(boundary)),
        'labels': [-100] * len(boundary) + tokens[len(boundary):]}]
    resources = {'max_training_seconds': 3600, 'max_ram_bytes': 32 * 1024**3,
                 'max_vram_bytes': 31 * 1024**3, 'max_disk_bytes': 1024**3}
    trainer = {key: evidence[key] for key in ('base_manifest', 'converter_manifest', 'provenance_manifest')}
    template = json.loads(Path(request['template']).read_text())['recipe']['trainer']
    trainer.update({key: value for key, value in template.items() if key not in trainer})
    trainer.update(python=sys.executable, python_sha256=sha256_file(Path(sys.executable)),
        packages={key: metadata.version(key) for key in PACKAGES}, seed=17, learning_rate=.0001,
        inference_name='gemma-4-31B-it-uncensored-heretic',
        gpu={'torch_vram_bytes': 22 * 1024**3, 'max_sequence_tokens': 256, 'max_rank': 2, 'max_steps': 64})
    parent = {'kind': 'native_llama_kv', 'model_sha256': BASE_SHA, 'lora_adapters': []}
    recipe = seal({'schema': 1, 'kind': GPU_KIND, 'parent': parent, 'resources': resources,
                   'checks': CHECKS, 'trainer': trainer})
    preferences = dict(HELP['create']['plan']['preferences'])
    preferences.update(rank=2, alpha=4, steps=2, scale=.1, adoption='automatic_if_checks_pass')
    compiled = {'schema': 1, 'instance_id': 'synthetic-' + uuid.uuid4().hex, 'draft_revision': 'synthetic',
        'recipe': recipe, 'parent': parent, 'examples': examples, 'preferences': preferences,
        'resources': resources, 'checks': CHECKS, 'implementation': implementation_identity(),
        'training_performed': False, 'synthetic_only': True, 'model_approval_supplied': False}
    compile_training(compiled)
    write_durable(folder / 'compiled.json', seal(compiled))


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--prepare':
        prepare(Path(sys.argv[2]).resolve())
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        for name in ('output', 'proof', 'template', 'training-python'):
            parser.add_argument('--' + name, type=Path, required=True)
        args = parser.parse_args()
        folder = args.output.resolve()
        folder.mkdir(parents=True, exist_ok=False)
        write_durable(folder / 'request.json', {'proof': str(args.proof.resolve()), 'template': str(args.template.resolve())})
        process = run_cpu_worker(args.training_python, [str(Path(__file__).resolve()), '--prepare', str(folder)],
            cwd=ROOT, log=folder / 'prepare.log', limits=WorkerLimits(4 * 1024**3, 240))
        write_durable(folder / 'prepare-process.json', process)
        if not process['succeeded']:
            raise SystemExit('synthetic plan preparation failed')
        from dmn.gpu_training_executor import GpuTrainingExecutor
        compiled = json.loads((folder / 'compiled.json').read_text())
        candidate = GpuTrainingExecutor(folder).candidate(folder, compiled)
        write_durable(folder / 'result.json', {'completed': True, 'synthetic_only': True,
            'model_approval_supplied': False, 'candidate_installed_for_instance': False, 'candidate': candidate})
        print(json.dumps({'completed': True, 'steps': candidate['training']['steps_completed'],
                          'trainable_parameters': candidate['training']['trainable_parameters']}))
