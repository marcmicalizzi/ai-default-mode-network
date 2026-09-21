# 31B NF4 feasibility experiment

This is a separate, explicitly launched resource experiment. It never opens an
instance or reads its conversations, thoughts, memories or checkpoints. It uses
one short synthetic sentence and writes a disposable adapter. It does not enable
the production deep-sleep trainer or authorize adoption of its output.

The source is `llmfan46/gemma-4-31B-it-uncensored-heretic`, pinned to revision
`d5bfc0d99e308beb9805440806161ad0233df357`. The selected files total
62,578,693,527 bytes. Their download size is not the GPU training footprint:
decoder linear weights are loaded into nested NF4, while non-quantized tensors,
adapter factors, gradients, activations and temporary buffers need additional
space. The frozen vision tower and projection are explicitly placed on CPU.

## Reproduce

Use the isolated environment from the [tiny GPU experiment](qlora-gpu-probe.md).
The source preparation command downloads pinned metadata/tokenizer files by
default. Downloading both large weight shards requires `--download-weights`:

```powershell
.venv-train-qlora-probe\Scripts\python.exe scripts/prepare_qlora_source.py --output data/qlora-31b-source --download-weights
.venv\Scripts\python.exe scripts/probe_qlora_31b.py --source data/qlora-31b-source
```

The second command only inspects local metadata and prints the intended plan.
It does not import Torch or initialize CUDA. After arranging GPU maintenance,
explicit execution uses a new output directory:

```powershell
.venv\Scripts\python.exe scripts/probe_qlora_31b.py --execute --source data/qlora-31b-source --output data/qlora-31b-run --training-python .venv-train-qlora-probe/Scripts/python.exe --max-ram-mib 32768 --max-seconds 1200 --torch-vram-mib 22528
```

The experiment hashes both source shards against the pinned LFS digests before
loading them. It uses rank two with alpha four on the exact text q/o projection
paths, non-reentrant gradient checkpointing, BF16 NF4 computation and one AdamW
step at a learning rate of 0.0001. Input is limited to 32 tokens. Frozen base,
vision and quantization state are hashed before and after the update.

## Measured result

**The one-step experiment passed on Windows/RTX 5090 on 2026-09-21.**

| Measurement | Result |
| --- | --- |
| Synthetic sequence | 16 tokens, one optimizer step |
| Trainable parameters | 3,584,000; rank two, text q/o projections |
| Quantized modules | 410 decoder linears |
| Adapter tensors | 240 finite factors, 120 changed on the first step |
| Frozen state | Base, vision and quantization state unchanged |
| Prepared Torch allocations | 20,929,156,608 bytes, about 19.49 GiB |
| Peak Torch allocated | 21,270,654,464 bytes, about 19.81 GiB |
| Peak Torch reserved | 23,592,960,000 bytes, about 21.97 GiB |
| Windows job peak commit counter | 33,100,963,840 bytes, about 30.83 GiB |
| Forward through optimizer update | About 2.63 seconds |
| Entire worker, including integrity checks/loading | About 408.9 seconds |
| Cleanup | Zero remaining job processes |

Most elapsed time went to source hashing, quantization/loading and frozen-state
verification. This result does not measure long-example or repeated-step memory
requirements. The 32 GiB job limit had little remaining margin during this run.

Windows Job Objects enforce aggregate committed RAM and process-tree cleanup;
the supervisor enforces elapsed time. The 22 GiB setting limits PyTorch's
allocator, **not all process/driver VRAM**. No whole-process GPU quota or disk
quota is claimed. Allocation failure is reported without raising limits or
automatically trying again. Linux containment remains separate work.

`input.json`, `progress.json`, `process.json` and, on success, `result.json` record
the plan, stage measurements and outcome. Failures preserve diagnostics. A
Windows job peak counter can include refused allocations and is not a reliable
measure of successfully resident host memory; process RSS and Torch allocation
counters are reported separately.

## Windows loading and preparation

The first real-source attempt failed before GPU loading with Windows error 1455
while opening the 49.9 GB shard. The research-only streaming loader avoids that
whole-shard operation: it validates dense tensor headers and offsets, then reads
one complete tensor at a time, bounded to 3 GiB. It has no Torch dependency for
inspection, rejects malformed or changed assets, and is not a general replacement
for safetensors. Tiny-model values and training/reload losses were checked against
the standard reader.

Subsequent attempts reached NF4 loading at about 16.73 GiB of Torch allocations,
then failed on a 5.25 GiB embedding conversion. Preparation now stages large
half-precision tensors in RAM, converts them there and transfers the F32 result.
It preserves the Parameter object and tied references. Removing inference hooks
can restore an offloaded component's original GPU placement, so the helper also
reapplies CPU placement for the unused vision components and checks it.

The worker fixes its native PyTorch allocator to `max_split_size_mb:128` to avoid
splitting large temporary weight buffers into retained fragments. This setting
does not increase its memory ceiling. See the
[PyTorch allocator documentation](https://docs.pytorch.org/docs/2.14/notes/cuda.html#optimizing-memory-usage-with-pytorch-alloc-conf).
The modified preparation passes the tiny four-step, frozen-state and PEFT-reload
checks with exactly the same selected losses as the original experiment.

## Adapter conversion

A separate CPU worker converted the completed 31B adapter and checked all 240
GGUF factor tensors against PEFT bit-for-bit. It passed in about 13.3 seconds
under a 1,536 MiB job limit, with a peak commit counter of 359,280,640 bytes.
No 31B native model was loaded for this check.

Reuse the pinned converter tree manifest from the
[base-provenance setup](base-provenance.md), then use a new output directory:

```powershell
.venv\Scripts\python.exe scripts/convert_qlora_31b_probe.py --probe data/qlora-31b-run --converter-manifest data/provenance-inputs/converter.json --training-python .venv-train-probe/Scripts/python.exe --output data/qlora-31b-conversion
```

The tool checks the completed source plan, adapter digest, rank, targets and
converter tree before conversion. Its report explicitly distinguishes factor
conversion from native evaluation or inference-base provenance.

## Scope of a successful result

The synthetic gradient step establishes a narrow resource/mechanics result.
It does not establish useful learning, safe repeated updates, long-example
capacity, full 31B adapter reload/llama.cpp behavior, or equivalence between this
source revision and an existing published GGUF. The
[remaining production gates](reviewed-training.md#remaining-production-gates)
still apply, including instance review, base provenance, failure choices and
service ownership during sleep and wake.
