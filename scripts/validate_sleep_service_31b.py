"""Explicit, contained full-size synthetic service handoff; never opens Syllas."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dmn.storage import write_durable
from dmn.worker_limits import WorkerLimits, run_monitored_gpu_worker


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'proof', 'training-python', 'model', 'projector'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.worker:
        from scripts.validate_sleep_service import validate
        validate(args.output.resolve()/'trial', args.proof.resolve(), args.training_python.resolve(),
                 full_model=args.model.resolve(), projector=args.projector.resolve())
    else:
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
        command = [str(Path(__file__).resolve()), '--worker']
        for name in ('output', 'proof', 'training_python', 'model', 'projector'):
            command += ['--'+name.replace('_','-'), str(getattr(args,name).resolve())]
        process = run_monitored_gpu_worker(Path(sys.executable), command, cwd=ROOT, log=output/'service.log',
            limits=WorkerLimits(48 * 1024**3, 7200), max_device_bytes=31 * 1024**3)
        write_durable(output/'process.json', process)
        success = process['succeeded'] and (output/'trial/result.json').exists()
        print(json.dumps({'completed':success, 'outcome':process['outcome'], 'synthetic_only':True}))
        raise SystemExit(0 if success else 1)
