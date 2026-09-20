"""Validate a server-rendered bundle with native vocabulary only, no inference."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dmn.backend import LlamaBackend, sha256_file
from dmn.config import Config
from dmn.initial_context import validate_bundle
from dmn.storage import write_durable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    import llama_cpp
    import llama_cpp.llama_cpp as api
    config = Config.read(args.bundle / "config.json")
    api.llama_backend_init()
    params = api.llama_model_default_params()
    params.vocab_only, params.n_gpu_layers = True, 0
    model = api.llama_model_load_from_file(os.fsencode(config.model_path), params)
    if not model:
        raise RuntimeError("native vocabulary failed to load")
    try:
        # Use the production tokenization method, without allocating a context
        # or pretending this read-only vocabulary handle is an inference engine.
        backend = LlamaBackend.__new__(LlamaBackend)
        backend.api, backend.vocab, backend.config = api, api.llama_model_get_vocab(model), config
        backend.template = api.llama_model_chat_template(model, None).decode()
        backend.fingerprint = {"model_sha256": sha256_file(Path(config.model_path))}
        manifest, _, tokens = validate_bundle(args.bundle, backend)
        report = {"verified": True, "tokens": len(tokens), "source_build": manifest["source_build"],
            "binding_version": llama_cpp.__version__, "tensor_data_loaded": False, "prompt_tokens_evaluated": 0,
            "source_prefix_tokens": manifest["keep_prefix_tokens"], "source_first_tokens": tokens[:8]}
        write_durable(args.report, report)
        print(json.dumps(report, indent=2))
    finally:
        api.llama_model_free(model)


if __name__ == "__main__":
    main()
