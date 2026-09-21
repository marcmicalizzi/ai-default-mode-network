"""Bind complete public-source audits into a local reviewed-training provenance record."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dmn.backend import sha256_file
from dmn.storage import write_durable
from dmn.worker_limits import WorkerLimits, run_cpu_worker


def worker(folder):
    from dmn.exact_base_provenance import prepare
    from dmn.training import tree_manifest
    request = json.loads((folder / 'input.json').read_text())
    trainer = {}
    for key, path, python_only in (('base_manifest', request['source'], False),
                                   ('converter_manifest', request['converter'], True)):
        target = folder / (key + '.json')
        write_durable(target, tree_manifest(Path(path), python_only=python_only,
                                            skip_hf_download_cache=key == 'base_manifest'))
        trainer[key] = {'path': str(target), 'sha256': sha256_file(target)}
    reference = prepare(folder / 'provenance.json', trainer,
        tensor_audit=request['tensor_audit'], tokenizer_audit=request['tokenizer_audit'],
        metadata_audit=request['metadata_audit'])
    write_durable(folder / 'result.json', {'completed': True, **trainer, 'provenance_manifest': reference})


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        worker(Path(sys.argv[2]).resolve())
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        for name in ('output', 'source', 'converter', 'tensor-audit', 'tokenizer-audit', 'metadata-audit'):
            parser.add_argument('--' + name, type=Path, required=True)
        args = parser.parse_args()
        folder = args.output.resolve()
        folder.mkdir(parents=True, exist_ok=False)
        write_durable(folder / 'input.json', {k: str(v.resolve()) for k, v in vars(args).items() if k != 'output'})
        result = run_cpu_worker(sys.executable, [str(Path(__file__).resolve()), '--worker', str(folder)],
            cwd=ROOT, log=folder / 'worker.log', limits=WorkerLimits(2 * 1024**3, 1200))
        write_durable(folder / 'process.json', result)
        print(json.dumps(result, indent=2))
        if not result['succeeded']:
            raise SystemExit(1)
