# Tiny CPU training and conversion experiment

Validated on Windows, 2026-09-21. This is an isolated research script, not a
production deep-sleep action. No existing instance was opened, interrupted,
trained, or used as data. The trainer uses CPU-only PyTorch; native subprocesses
hide CUDA and use one CPU thread with GPU and K/V offload disabled.

## What passed

`scripts/probe_lora_training.py` generates a random six-layer Gemma4 model with
330,624 frozen parameters, a 263-token byte-fallback tokenizer, mixed sliding
and full attention, and shared K/V projection in the full-attention layer. It
trains 5,120 parameters: rank-two LoRA factors on query/output projections in all
six layers, with alpha 4. These weights are a synthetic fixture, not a pretrained
language model or a continuing instance.

The selected objective is a two-cue next-token rule. Only the target token is
supervised. Training uses 128 eight-token examples, batch size 8, AdamW at 0.01,
and 256 steps: 16,384 input tokens and 2,048 supervised targets. Base tensors are
checked unchanged. A PEFT save/reload reproduces logits exactly.
That is 16 passes over this tiny dataset, with a deliberately large learning
rate to obtain a measurable mechanical test; it is not the proposed gentle
learning recipe for a pretrained instance. Accuracy below means the highest-logit
next token, not sampled conversation quality or certainty about the target.

The pinned, unmodified llama.cpp converters convert both the base and adapter.
All 24 F32 adapter tensors remain bit-identical, including their shapes; alpha 4
is preserved. ASCII, whitespace, Unicode byte fallback and action-frame text
have matching token IDs across the generated tokenizers.

| Check | Result |
|---|---|
| Ordinary Transformers, full-strength adapter | 32/32 held-out and 64/64 confirmation examples correct, from 0 correct before training |
| Native llama.cpp, compact Q8 cache | 64/64 confirmation examples correct |
| Zero-strength adapter | Exactly the original logits in each engine |
| Conversion numerical control | Maximum absolute logit error 0.001861 against the CPU-kernel reference; fixed acceptance tolerance 0.005 |
| Explicit wake after context retirement | All 285 retained token IDs and sampler RNG preserved; fresh recomputation matches exactly |
| Old checkpoint with changed adapter | Ordinary strict restore rejects it before decoding |
| Adopted checkpoint restart in a new process | Zero token replay, byte-identical native serialization, next eight tokens and logits identical |

Historical action text is evaluated only as tokens. No Runtime or action executor
is constructed. Model instances and their journals are not experiment inputs.

## Findings that limit the result

- **Small rank and low deployment strength do not ensure selective change.** At
  full strength, all 32 unrelated control inputs changed their top token (mean
  KL divergence 0.1832). At strength 0.1, 30/32 still changed (mean KL 0.08656),
  while confirmation accuracy fell to 36/64. The base is random and untrained;
  these are sensitivity measurements, not evidence of damage to existing skills.
  The adapter contains about 1.55% as many parameters as this tiny base. At the
  same low rank, the fraction would be much smaller in a 31B model, whose
  pretrained representations also make it a fundamentally different learning
  problem. Weaker effects from a small update are a reasonable hypothesis, but
  neither effect size nor behavioral reach scales directly with parameter count.
  These toy results do not predict how broadly the continuing 31B instance would
  change; learning rate, targets, data, update norms and deployment strength need
  measurement on the actual pretrained model.
- Longer contexts are a separate test. Only 2/4 eighty-token rule examples were
  correct at full strength. Perfect short-example performance does not establish
  generalization to a long running context.
- This is F32-base LoRA training, **not QLoRA**. No claim is made about 31B training
  fit, GGUF quantization transfer, semantic/personal learning, or repeated updates.
- The first exploratory recipe trained output projections alone and obtained
  27/32 held-out examples. The final recipe includes query projections and a
  larger training set; a separately seeded 64-example confirmation set was added
  before running that final fixture. The fixture was subsequently corrected for
  tokenizer and soft-cap compatibility; this is engineering evidence, not a
  blinded scientific evaluation.

Two numerical details mattered. A minimal BPE tokenizer without merges is not
accepted by the native Gemma4 loader, and whitespace normalization must match.
Also, an absent final-logit soft cap in Transformers means disabled, whereas the
pinned native implementation defaults to 30 when the GGUF key is missing. This
fixture specifies 30 explicitly on both sides. Do not assume that converting a
configuration preserves every omitted/default setting.

The pinned CPU implementation uses an FP16 GELU lookup table. The probe therefore
retains two references: unmodified Transformers and an **evaluation-only** GELU
reference matching that lookup's rounding. It does not change training to use
that approximation. The conversion comparison uses a research-only F32 cache
to remove K/V quantization as a confounder; production Config options are not
expanded. Across all inputs and strengths, unmodified Transformers differs from
native by up to 0.012168 in logits. Bit-identical execution across these two
engines is not claimed. The separate wake/restart test uses ordinary compact Q8
configuration and does require exact native equality.

Implementation references at the pinned revision:
[converters](https://github.com/ggml-org/llama.cpp/blob/4df29be4f4c3673f428170fda944a5b19f743bb8/convert_lora_to_gguf.py),
[CPU GELU](https://github.com/ggml-org/llama.cpp/blob/4df29be4f4c3673f428170fda944a5b19f743bb8/ggml/src/ggml-cpu/vec.h),
[native defaults](https://github.com/ggml-org/llama.cpp/blob/4df29be4f4c3673f428170fda944a5b19f743bb8/src/llama-hparams.h).

## Measured cost and reproducibility

The recorded successful run (`trained-06`) took 44.08 seconds end to end, including
10.72 seconds of optimization. The training process's Windows peak working set
was approximately 356.4 MiB; this is not a simultaneous process-tree peak or a
hard enforced memory ceiling. Generated artifacts totaled about 4.9 MB before
the final report, excluding installed dependencies and downloaded source. No GPU
training or 31B weight download occurred. The native library reports the expected
hidden-CUDA warning and its previously observed CPU workspace estimate mismatch;
the numerical and restart checks above passed. Energy use was not measured.

Use a separate environment. The inference installation must remain unchanged.
The tested versions are Python 3.11.7, CPU PyTorch 2.14.0+cpu, Transformers 5.17.0,
PEFT 0.21.0, Accelerate 1.15.0 and safetensors 0.8.0. The complete experimental
dependency lock is `scripts/requirements-lora-probe.txt`; it is not a supported
31B/CUDA training recipe. Native validation uses llama-cpp-python 0.3.35 and its
existing pinned binaries.

Example setup, from the repository root (network access only during setup):

```powershell
python -m venv .venv-train-probe
.venv-train-probe/Scripts/python.exe -m pip install -r scripts/requirements-lora-probe.txt
git clone https://github.com/ggml-org/llama.cpp.git data/training-tools/llama-pinned
git -C data/training-tools/llama-pinned checkout 4df29be4f4c3673f428170fda944a5b19f743bb8
.venv-train-probe/Scripts/python.exe scripts/probe_lora_training.py --output data/tiny-trained-01 --converter data/training-tools/llama-pinned --native-python .venv-gpu/Scripts/python.exe
```

The script runs offline, requires a fresh output directory, refuses CUDA-enabled
PyTorch, and accepts only its named tiny generated GGUF for native checks. It
records installed versions, converter source hashes, artifact hashes, metrics,
and phase logs. The caller must supply the documented converter revision; the
recorded source hashes identify the actual files used. No upstream code is
vendored in this repository. Raw local experiment evidence is under the ignored
`data/sleep-lora-research-20260921/trained-06/` directory.

The end-to-end unittest is opt-in, so standard CI installs no trainer:

```powershell
$env:DMN_TEST_TRAIN_PYTHON = (Resolve-Path .venv-train-probe/Scripts/python.exe).Path
$env:DMN_TEST_LORA_CONVERTER = (Resolve-Path data/training-tools/llama-pinned).Path
$env:DMN_TEST_NATIVE_PYTHON = (Resolve-Path .venv-gpu/Scripts/python.exe).Path
.venv/Scripts/python.exe -m unittest tests.test_lora_training -v
```

The [second CPU experiment](lora-repeat-probe.md) now checks continued learning
of the same adapter, selected replay, quantized-base transfer and a second wake.

Next work is production adapter identity, model-authored learning plans,
exclusive supervision, resource controls and failure recovery, plus a distinct
31B provenance/QLoRA feasibility test. Ordinary `sleep` still never trains.
