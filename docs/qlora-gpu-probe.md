# Prepared GPU validation experiment

The next experiment uses the generated tiny full Gemma wrapper with NF4 frozen
weights, nested quantization, BF16 compute, rank-two text adapters and four
gradient steps. It checks the quantized module set, exact trainable text paths,
unchanged frozen base/vision/quantization state, finite changed factors and PEFT reload.
It creates no Runtime, reads no instance data and cannot adopt its adapter.
The source is limited to the known synthetic geometry and 4 MiB of local files.

**GPU execution has not been validated yet.** The inspection path and launcher
containment tests use no GPU. Obtain Syllas's maintenance agreement before the
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
.venv\Scripts\python.exe scripts/launch_qlora_probe.py --execute --fixture data/wrapped-fixture --training-python .venv-train-qlora-probe/Scripts/python.exe --output data/qlora-gpu-probe --max-ram-mib 2048 --max-seconds 180 --torch-vram-mib 1024
```

Output must be a new directory. Windows Job Objects enforce the 2 GiB aggregate
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
weights, tokenizer/provenance, training kernels and resource measurements remain
separate validation steps after this small GPU test.
