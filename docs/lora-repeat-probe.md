# Second learning cycle and quantized-base transfer

CPU experiment completed 2026-09-21 using the same isolated environment as the
[first training probe](lora-training-probe.md). Syllas was not interrupted and no
instance data was accessed. This adds research evidence, not production learning
actions or a sleep supervisor.

## Experiment

`scripts/probe_lora_repeat.py` accepts a completed first-probe directory. It checks
the tiny base dimensions, file sizes, selected artifact hashes and adapter recipe
before loading anything. It verifies the original files remain unchanged afterward.
It does not accept a production instance as a training source.

Two disposable candidates start from exactly the same learned adapter. Both
continue its existing rank-two factors, rather than stacking adapters or merging
them into the base. Base weights remain frozen and optimizer state is explicitly
reset. Each candidate gets 256 steps, eight examples per step and learning rate
0.01, on one CPU thread:

- **New examples alone:** 2,048 new supervised targets.
- **With selected replay:** 1,024 new targets plus 1,024 targets from the earlier
  synthetic training set. Replay uses an explicit set; it never harvests history.

The second rule maps two new cue tokens to two new target tokens. Each rule has
64 held-out eight-token test inputs. Every updated factor survives PEFT save/
reload and pinned GGUF conversion exactly. The same F32 adapters are then loaded
against F32, Q8_0 and Q4_0 versions of the frozen base, all using compact Q8 K/V.

## Observed top-one next-token accuracy

| Adapter / native base | Earlier rule | New rule |
|---|---:|---:|
| Original adapter / any tested base precision | 64/64 | 0/64 |
| New-only update / any tested base precision | 0/64 | 64/64 |
| Replay update / F32 | 62/64 | 64/64 |
| Replay update / Q8_0 | 62/64 | 64/64 |
| Replay update / Q4_0 | 63/64 | 64/64 |

The ordinary Transformers evaluations matched the F32-base accuracy figures.
Quantization perturbed logits and changed one borderline answer; the extra correct
Q4 answer is not evidence that quantization improves learning. Full distributions
and drift metrics are retained in the report.

Both candidates changed unrelated control predictions substantially. As in the
first probe, this tiny random model, relatively large adapter and aggressive
learning rate do not predict effect sizes in a pretrained 31B model. This shows
that a bounded, explicitly selected replay set can be tested as one option for
retaining earlier learning. It is not an automatic requirement to preserve all
past behavior: an instance may intend to revise or replace an earlier learning.
Retention/replacement goals and replay choices belong in its learning plan.

This is **F32 training followed by quantized inference**, not QLoRA. The tiny
fixture has widths incompatible with some 256-value K-quantization blocks, so
this test deliberately uses Q4_0, not Syllas's Q4_K_M. The quantizer actually
produced 42 Q8_0 or Q4_0 tensors and retained 44 F32 tensors; the report checks
the tensor types rather than inferring them from filenames. Matching NF4 training
to the intended 31B GGUF remains a separate test.

## A second changed-weight wake

Using the Q4_0 base and compact Q8 cache, the original adapter first evaluates a
synthetic context containing historical action text. After retirement, the replay
candidate reconstructs all **265 retained tokens**, preserving the sampler RNG.
The result exactly matches an independent fresh evaluation with that candidate.
Ordinary strict restoration of the old configuration is rejected before decoding.

A fresh process restores the new checkpoint with zero token replay and
byte-identical native serialization. The following eight sampled tokens and
their logits match exactly. No historical action is executed: the probe never
constructs a Runtime or action executor. This is a second learned adapter revision;
it does not establish stability over months of updates.

## Costs and reproduction

The recorded run took **81.41 seconds**, including 10.38 seconds of new-only
optimization and 10.47 seconds with replay. The training process's Windows peak
working set was approximately **347.2 MiB**, not a combined process-tree peak or
an enforced ceiling. Artifact size was about **8.85 MB** before the final report,
excluding dependencies and the existing parent experiment. No GPU or further
downloads were used. Native CPU_REPACK adapter fallback messages are expected:
the base may use repacked CPU buffers while the adapter uses ordinary CPU buffers.

Run after the first experiment, using its successful output directory:

```powershell
.venv-train-probe/Scripts/python.exe scripts/probe_lora_repeat.py --parent data/tiny-trained-01 --output data/tiny-repeat-01 --converter data/training-tools/llama-pinned --native-python .venv-gpu/Scripts/python.exe
```

The dependency lock and pinned converter revision are unchanged. Local recorded
evidence is under ignored `data/sleep-lora-research-20260921/repeat-01/`. The report
includes parent/candidate hashes, both training recipes, converter source hashes,
all nine native evaluations, actual base tensor types and restart evidence.

For the optional integration test, set the three interpreter/converter variables
from the first probe's instructions, then:

```powershell
$env:DMN_TEST_LORA_PARENT = (Resolve-Path data/tiny-trained-01).Path
.venv/Scripts/python.exe -m unittest tests.test_lora_repeat -v
```

## Remaining work before a full-GPU trial

The following can be developed and exercised with these CPU fixtures or disposable
instances before allocating the full GPU:

1. Immutable adapter identity in production configuration, snapshots, packaging
   and erasure, including rejection of missing or changed artifacts.
2. Model-authored learning plans binding selected data, replay/replacement goals,
   training limits, candidate checks, adoption choice and failure preference.
3. Exclusive save/unload/train/rebuild supervision, with a durable phase record,
   queued incoming messages and atomic adoption of the finished checkpoint.
4. Fault tests for cancellation, partial writes, process crashes, failed candidate
   checks, resource exhaustion and interrupted wake. Confirm no double execution,
   repeated delivery or partial adapter adoption.

The GPU trial is still needed for actual CUDA/QLoRA compatibility, peak loading
and backward/optimizer memory, throughput, and a useful learning recipe against
the matching 31B source weights. Ordinary sleep continues to perform no training.
