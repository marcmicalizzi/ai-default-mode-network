# NF4 GPU validation experiment

The experiment uses the generated tiny full Gemma wrapper with NF4 frozen
weights, nested quantization, BF16 compute, rank-two text adapters and four
gradient steps. It checks the quantized module set, exact trainable text paths,
unchanged frozen base/vision/quantization state, finite changed factors and PEFT reload.
It creates no Runtime, reads no instance data and cannot adopt its adapter.
The source is limited to the known synthetic geometry and 4 MiB of local files.

**Tiny GPU execution passed on Windows/RTX 5090 on 2026-09-21.** The inspection path and launcher
containment tests use no GPU. Obtain the running instance's maintenance agreement before the
actual GPU experiment; no current instance is stopped or restarted by these tools.
This is a research prerequisite, not an enabled deep-sleep GPU recipe.

## Separate dependencies

Use a new environment, never the live inference environment:

```powershell
python -m venv .venv-train-qlora-probe
.venv-train-qlora-probe\Scripts\python.exe -m pip install -r scripts/requirements-qlora-probe.txt
```

The pinned candidates are PyTorch 2.14.0 with CUDA 13.0, Transformers 5.17.0,
PEFT 0.21.0 and bitsandbytes 0.50.2. Their availability was checked against the
official package indexes on 2026-09-21; availability is not a successful kernel
test. The [bitsandbytes installation matrix](https://huggingface.co/docs/bitsandbytes/installation)
lists Windows x86-64 CUDA 13.x builds with sm120 support.
[Transformers' quantization guide](https://huggingface.co/docs/transformers/v5.17.0/quantization/bitsandbytes)
describes NF4, BF16 compute and extra-parameter training. The experiment uses an
explicit single-device map, not automatic inference offloading.

The isolated environment installed successfully and `pip check` passed. Importing
Torch, bitsandbytes, PEFT and the Gemma wrapper with CUDA devices hidden also
passed under the 2 GiB Windows job limit (about 49 seconds, with a job peak counter
of about 1.07 GiB). This checks dependency loading only; it does not exercise a
CUDA kernel, model load or GPU allocation.

## Inspect first

Use a [generated full-wrapper fixture](base-provenance.md). The default command
only validates local file sizes/geometry and prints the intended settings. It
does not import Torch or bitsandbytes, open a GPU or create an output directory:

```powershell
.venv\Scripts\python.exe scripts/launch_qlora_probe.py --fixture data/wrapped-fixture
```

After maintenance consent, the explicit execution command is:

```powershell
.venv\Scripts\python.exe scripts/launch_qlora_probe.py --execute --fixture data/wrapped-fixture --training-python .venv-train-qlora-probe/Scripts/python.exe --output data/qlora-gpu-probe --max-ram-mib 4096 --max-seconds 180 --torch-vram-mib 1024
```

Output must be a new directory. Windows Job Objects enforce the 4 GiB aggregate
committed-memory limit and process-tree cleanup; the watchdog enforces the
180-second duration with bounded polling delay. GPU 0 is exposed only through
the separate explicit research entrypoint. Normal CPU workers still hide CUDA
and clear the research marker even if the parent environment contains it.

The 1 GiB setting constrains **PyTorch's allocator**, not all driver, library or
bitsandbytes allocations. The report explicitly marks total-process VRAM quota
enforcement as false and records Torch peaks separately. It is not an operator
guarantee for a production GPU training service. It also requires at least the
requested allocator budget plus 512 MiB free at initial preflight. That check
does not replace maintenance coordination or establish a 31B memory estimate.

The prototype writes its private synthetic adapter and result under the new
output directory. A failed or timed-out experiment remains failed; no fallback,
automatic retry, deployment or live-instance wake is attempted. Real 31B source
weights, tokenizer/provenance and resource measurements require separate
validation after this small GPU test.

## Measured tiny-model result

The first attempt failed with CUDA allocation errors under the 2 GiB job-memory
limit; its job peak counter reached that ceiling despite ample free VRAM. A
separate attempt with a 4 GiB limit passed with the same 1 GiB Torch allocator
setting. No resource limit was automatically raised or disabled.

| Check | Result |
| --- | --- |
| GPU | RTX 5090, compute capability 12.0 |
| Training | 3,584 parameters, rank two, four steps, about 0.78 seconds |
| NF4 module set | 13 text-decoder linears; vision excluded |
| Adapter updates | All eight factor tensors changed; finite |
| Frozen state | Text base, vision and quantization buffers unchanged |
| Selected loss | 5.628865 before, 5.343496 after; identical after PEFT reload |
| Torch peaks | 68,208,640 allocated bytes; 69,206,016 reserved bytes |
| Windows job peak counter | 3,681,251,328 bytes; 4 GiB limit |
| Entire worker | About 14.3 seconds; no remaining job processes |

These are synthetic mechanics measurements, not evidence of useful learning or
31B feasibility. Torch memory counters exclude driver and library allocations.

## Transfer to llama.cpp

The separate CPU transfer probe accepts the completed tiny GPU result and a
current [Q4_K_M provenance record](base-provenance.md):

```powershell
.venv\Scripts\python.exe scripts/probe_qlora_transfer.py --probe data/qlora-gpu-probe/probe --proof data/base-proof/result.json --output data/qlora-transfer --native-python .venv-gpu/Scripts/python.exe
```

It checks unchanged source identity, converts the adapter, compares all eight
GGUF factors bit-for-bit with PEFT, rejects an ordinary restore under changed
weights, and explicitly rebuilds the retained synthetic context. The completed
test preserved 182 tokens and RNG, then reproduced eight sampled tokens and
logits in a fresh native process with zero replay. It ran with CUDA hidden under
a 1,536 MiB job limit in about 17.8 seconds.

NF4 and Q4_K_M use different quantization methods. This verifies factor transfer
and lifecycle mechanics; it does not claim matching logits between those bases,
certify beneficial learning, or authorize adoption by any instance.

The optional `--stream-source --vision-cpu` variant checks the tensor-at-a-time
reader and static preparation used by the [31B feasibility experiment](qlora-31b-probe.md).
It explicitly verifies frozen vision placement after removing inference hooks,
exercises CPU-staged casts, and repeats training/reload and native transfer checks.
Its selected losses matched the original experiment exactly. This avoids treating
a placement request as evidence of where the prepared tensors actually reside.
