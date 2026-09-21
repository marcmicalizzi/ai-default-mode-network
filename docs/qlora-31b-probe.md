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
step at a learning rate of 0.0001 by default. The default input remains limited
to 32 tokens. Explicit `--sequence-tokens 128|256|512|1024` and `--steps 1|2|3|4`
select longer, repeated synthetic workloads. They do not raise resource limits.
Frozen base, vision and quantization state are hashed before and after updates.

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
verification. The 32 GiB job limit had little remaining margin during this run.

### Longer workload boundary

The follow-up on the same day passed **256 synthetic tokens and two optimizer
steps**, with all 240 factors changed and frozen state unchanged. Peak Torch
allocation was 22,456,119,808 bytes (20.91 GiB); peak reserved was 23,592,960,000
bytes (21.97 GiB). The updates took about 3.9 seconds; the complete worker took
440.3 seconds, including loading and integrity checks. Repeated synthetic text
is a mechanics workload, not evidence of useful learning.

At **512 tokens**, the first backward pass requested another 512 MiB and hit the
22 GiB allocator ceiling. It exited with no remaining job processes and no
completed adapter. Limits were not increased; 1,024 tokens was consequently not
attempted. This is a boundary for this implementation and allocator budget,
not a claim that the GPU cannot support a more memory-efficient training path.

```powershell
.venv\Scripts\python.exe scripts/probe_qlora_31b.py --execute --source data/qlora-31b-source --output data/qlora-31b-256 --training-python .venv-train-qlora-probe/Scripts/python.exe --sequence-tokens 256 --steps 2
```

Successful runs now save a short, hash-bound logit reference. The separate
`scripts/reload_qlora_31b_probe.py` reloads the NF4 base and saved PEFT adapter in
a fresh contained GPU process, using the same 22 GiB allocator and 32 GiB job
limits. It accepts neither incomplete runs nor a changed adapter digest.

After fixing static-placement handling, the 31B adapter reloaded in a fresh
process with **bit-identical logits** for the saved 16-token reference. The
worker took about 249.6 seconds, peaking at 21,059,472,384 allocated Torch bytes
(19.61 GiB), and exited with no remaining processes.

```powershell
.venv\Scripts\python.exe scripts/reload_qlora_31b_probe.py --probe data/qlora-31b-256 --training-python .venv-train-qlora-probe/Scripts/python.exe --output data/qlora-31b-reloaded
```

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

The preparation helper also removes the stale inference `hf_device_map` after
establishing static placement. A full 31B fresh-process PEFT reload exposed why:
leaving that map caused PEFT to infer a new distribution and redispatch the
prepared model, leaving nested NF4 quantization state on the meta device. The
reload checks require materialized parameters and CPU placement for frozen
vision; they do not silently accept automatic redistribution.

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

The 256-token/two-step adapter also passed all 240 exact factor comparisons.

## Read-only published-base audit

`scripts/audit_31b_base.py` hashes the pinned source shards and the published
Q4_K_M file, maps the complete text tensor set, and requantizes bounded source
rows in RAM. It uses `ggml_quantize_chunk` from the pinned native build, without
an importance matrix, following its
[C declaration](https://github.com/ggml-org/llama.cpp/blob/4df29be4f4c3673f428170fda944a5b19f743bb8/ggml/include/ggml.h).
It writes only small reports; neither model is modified or duplicated.

The sampled audit passed: all 421 source-backed F32 tensors matched exactly,
as did the first, middle and last rows of all 355 Q4_K and 56 Q6_K matrices.
The generated rotary-frequency tensor also matched. The comparison covered
1,654 source rows and 11,718,044 bytes, under a 2 GiB CPU job limit. All 833 GGUF
tensor names/shapes were accounted for.

The separate `--tokenizer-only` audit matched all **262,144 vocabulary entries**,
scores, token types and special-token IDs/settings. It also found an
actual metadata difference: the source's 18,839-character chat template differs
from the published GGUF's 17,344-character template, including tool formatting
and turn-handling code. It records both template digests and reports
`chat_template_equal=false`; it does not rewrite either template. Metadata is
copied into compact values before loading the HF tokenizer so the check fits
the same 2 GiB CPU limit.

This is useful evidence of the relationship, **not a production provenance
receipt**. Most matrix rows, complete metadata reproduction and the publisher's
conversion pipeline remain unverified. The optional `--full` mode is
for a later complete payload comparison; it was not run in this validation and
would still not reproduce the entire GGUF file. Production gates are unchanged.

## Full-size native wake and restart

`scripts/validate_31b_native.py` uses the pinned published Q4_K_M base, the
converted synthetic adapter at scale 0.1, Q8 KV, compact sliding-window storage
and a 4,096-position disposable context. Separate contained processes validate
each phase. It never constructs a DMN Runtime or opens an instance directory.

The 31B text path passed on Windows/RTX 5090:

- Decode 3,072 synthetic tokens, retire 1,024 while preserving the full recent
  local window, then save 2,053 retained tokens after a short suffix.
- Reject both strict restore and ordinary text reconstruction under changed
  adapter weights, before any decoding.
- Explicitly rebuild those exact retained tokens under the trained adapter,
  preserving the sampler RNG. Rebuild plus checkpoint writing/hashing took
  about 91.2 seconds; this is not an isolated prefill benchmark.
- Start a fresh process, restore the new native checkpoint with **zero replay**,
  and reproduce all 16 continuation token IDs and logit arrays bit-for-bit.

Each worker had a 32 GiB committed-memory limit and 1,200-second deadline.
The base checkpoint's native state was 646,452,262 bytes. This validates a short
retired context on this build; it does not size a 60,000-token production wake,
certify useful learning, or authorize adopting this disposable adapter.

The same harness has separate real-projector image phases. Run it only during
an agreed GPU maintenance window, with a fresh output directory:

```powershell
.venv\Scripts\python.exe scripts/validate_31b_native.py --model <published-base.gguf> --adapter data/qlora-31b-conversion/adapter.gguf --projector <matching-mmproj.gguf> --native-python .venv-gpu/Scripts/python.exe --pillow-site .venv/Lib/site-packages --output data/31b-native-validation
```

The script requires the specific model/projector hashes recorded in its source.
`--pillow-site` supplies Pillow from a separate compatible Python environment;
it does not install or change packages in the inference environment. The vision
projector executes on CPU while decoder layers are on GPU.

All five phases passed, including the matching BF16 projector. The image trial
restored 1,094 positions with zero replay and eight bit-identical continuation
token/logit steps. Rebuild correctly refused retained visual positions; after
image retirement, 1,056 text tokens rebuilt successfully. No pixels were archived.
See [vision measurements and limits](attachment-vision.md#real-31b-native-validation).

The standard suite passed 401 tests with 27 optional skips. The tiny NF4
four-step training/reload regression also passed after the static-placement fix,
retaining exactly the earlier loss values and unchanged frozen state. These
checks used disposable fixtures and left Syllas's state untouched.

## Scope of a successful result

The synthetic gradient steps establish a narrow resource/mechanics result.
They do not establish useful learning, stable long-term updates, longer-example
capacity or complete equivalence between this source revision and the published
GGUF. The
[remaining production gates](reviewed-training.md#remaining-production-gates)
still apply, including instance review, base provenance, failure choices and
service ownership during sleep and wake.
