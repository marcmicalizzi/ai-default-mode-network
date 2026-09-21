"""CPU-only native quantization child; never creates an inference context."""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import sys

from .backend import sha256_file
from .storage import write_durable


def native_identity():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import llama_cpp
    from llama_cpp import llama_cpp as api
    from .native_logging import configure_native_logging
    configure_native_logging(api)
    library = Path(api._lib._name).resolve().parent
    return {"python": str(Path(sys.executable).resolve()), "python_sha256": sha256_file(Path(sys.executable)),
            "binding_version": llama_cpp.__version__, "binding_source": str(Path(api.__file__).resolve()),
            "binding_source_sha256": sha256_file(Path(api.__file__)),
            "binaries": {str(p.resolve()): sha256_file(p) for p in sorted(library.rglob("*"))
                         if p.is_file() and p.suffix in {".dll", ".so", ".dylib"}}}


def quantize(folder):
    from llama_cpp import llama_cpp as api
    request = json.loads((folder / "input.json").read_text())["request"]
    identity = native_identity()
    if identity != request["quantizer"]:
        raise ValueError("native quantizer differs from the bound environment")
    source, output = folder / "base-f32.gguf", folder / "base.gguf"
    if source.stat().st_size > 4 * 1024**2 or output.exists():
        raise ValueError("quantization accepts only a fresh tiny artifact")
    types = {"Q8_0": api.LLAMA_FTYPE_MOSTLY_Q8_0, "Q4_0": api.LLAMA_FTYPE_MOSTLY_Q4_0,
             "Q4_K_M": api.LLAMA_FTYPE_MOSTLY_Q4_K_M}
    params = api.llama_model_quantize_default_params()
    params.nthread, params.ftype = 1, types[request["quantization"]]
    params.allow_requantize = False
    params.pure = request["quantization"] != "Q4_K_M"
    actual = {}
    for name, _ in params._fields_:
        value = getattr(params, name)
        if type(value) in (int, bool):
            actual[name] = value
        elif not value:
            actual[name] = None
        else:
            raise ValueError("nonempty quantizer pointer defaults are not supported")
    source_hash = sha256_file(source)
    if api.llama_model_quantize(os.fsencode(source), os.fsencode(output), ctypes.byref(params)):
        raise ValueError("native CPU quantization failed")
    if sha256_file(source) != source_hash or native_identity() != identity:
        raise ValueError("source or quantizer changed during conversion")
    write_durable(folder / "native.json", {"identity": identity, "parameters": actual,
        "source_sha256": source_hash, "output_sha256": sha256_file(output)})


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if sys.argv[1:] == ["identity"]:
        print(json.dumps(native_identity()))
    else:
        quantize(Path(sys.argv[1]).resolve())
