"""Read GGUF metadata only; never map tensors or allocate a model/KV cache."""
import argparse
import hashlib
import json
import struct
from pathlib import Path


def inspect(path, include_template=False):
    file_size = path.stat().st_size
    scalar_types = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?",
                    10: "Q", 11: "q", 12: "d"}
    with path.open("rb") as file:
        def unpack(fmt):
            return struct.unpack("<" + fmt, file.read(struct.calcsize("<" + fmt)))[0]

        def string(keep=True):
            size = unpack("Q")
            if size > file_size - file.tell():
                raise ValueError("invalid GGUF string length")
            if keep:
                return file.read(size).decode("utf-8")
            file.seek(size, 1)

        def value(kind, keep=True):
            if kind in scalar_types:
                return unpack(scalar_types[kind])
            if kind == 8:
                return string(keep)
            if kind == 9:
                element, count = unpack("I"), unpack("Q")
                if element in scalar_types:
                    if keep and count <= 256:
                        return [unpack(scalar_types[element]) for _ in range(count)]
                    file.seek(count * struct.calcsize("<" + scalar_types[element]), 1)
                else:
                    for _ in range(count):
                        value(element, False)
                return {"array_type": element, "length": count}
            raise ValueError(f"unsupported GGUF type {kind}")

        if file.read(4) != b"GGUF":
            raise ValueError("not a little-endian GGUF file")
        version = unpack("I")
        if version not in {2, 3}:
            raise ValueError(f"unsupported GGUF version {version}")
        tensors, entries = unpack("Q"), unpack("Q")
        metadata = {}
        for _ in range(entries):
            key = string()
            metadata[key] = value(unpack("I"))
        template = metadata.get("tokenizer.chat_template")
        if isinstance(template, str) and not include_template:
            metadata["tokenizer.chat_template"] = {"characters": len(template),
                    "sha256": hashlib.sha256(template.encode()).hexdigest()}
        return {"path": str(path.resolve()), "bytes": file_size, "gguf_version": version,
                "tensor_count": tensors, "metadata": metadata, "tensor_data_loaded": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = inspect(args.model)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
