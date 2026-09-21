"""Download the pinned 31B research source; weights require an explicit flag.

Uses a workspace-local cache and anonymous access. No model code is executed.
This supplies a disposable feasibility probe, not inference-base provenance.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dmn.storage import write_durable
from scripts.probe_qlora_31b import REPO, REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--download-weights', action='store_true')
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ['HF_HOME'] = str(root / 'cache')
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ.pop('HF_HUB_OFFLINE', None)
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi(token=False).model_info(REPO, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise ValueError('source revision mismatch')
    files = {f.rfilename: {'size': f.size, 'lfs_sha256': f.lfs.sha256 if f.lfs else None}
             for f in info.siblings if f.rfilename.endswith(('.json', '.safetensors', '.jinja'))
             or f.rfilename == 'README.md'}
    if any(Path(name).name != name or type(entry['size']) is not int or entry['size'] < 0
           for name, entry in files.items()):
        raise ValueError('unexpected source paths or metadata')
    total = sum(f['size'] for f in files.values())
    if total > 64 * 1024**3:
        raise ValueError('source exceeds fixed research download envelope')
    patterns = list(files) if args.download_weights else [n for n in files if not n.endswith('.safetensors')]
    requested = sum(files[n]['size'] for n in patterns)
    if shutil.disk_usage(root).free < requested * 2 + 10 * 1024**3:
        raise ValueError('insufficient free space for download and reserve')
    record = {'repo': REPO, 'revision': REVISION, 'bytes': total, 'files': files}
    manifest = root / 'source.json'
    if manifest.exists() and json.loads(manifest.read_text()) != record:
        raise ValueError('existing source metadata differs; use a fresh output directory')
    write_durable(manifest, record)
    snapshot_download(REPO, revision=REVISION, allow_patterns=patterns, local_dir=root / 'model',
                      cache_dir=root / 'cache', max_workers=2, token=False)
    print(json.dumps({'source': str(root / 'model'), 'revision': REVISION,
                      'requested_bytes': requested, 'weights_requested': args.download_weights}, indent=2))


if __name__ == '__main__':
    main()
