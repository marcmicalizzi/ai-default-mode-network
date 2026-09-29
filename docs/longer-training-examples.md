# Longer training examples: resource experiments

The disposable experiments investigate 512 through 2048 tokens. The original
v1 recipe still rejects examples above 256 tokens. A September 28 CPU integration
pass added [the separately offered v2 recipe](reviewed-nf4-training.md), using the
same chunked loss and permitting up to 1536 tokens. This report records the
September 27 research runs; it is not a new GPU service rehearsal.
The later [September 29 validation](nf4-v2-validation.md) records the integrated
v2 GPU checks separately.
Longer execution times are acceptable within an explicit
host time allowance and an instance-reviewed plan. Additional time does not by
itself remove a peak-memory allocation requirement.

The earlier synthetic 31B experiment at 512 tokens failed while requesting a
512 MiB allocation under a **22 GiB Torch allocator ceiling**. Torch reported
21.68 GiB allocated, about 90 MiB reserved but unused, and 4.66 GiB of device free
memory. Windows per-process memory figures in that error were nonsensical and
must not be interpreted as real usage. The result establishes failure of that
bounded workload, not exhaustion of every byte on the RTX 5090. The 256-token
probe passed. These are old research-probe results, not a length validation of
the current production trainer's target-only loss implementation.

## Validation sequence

1. During separately agreed GPU maintenance, measure a current production-shaped
   synthetic baseline: rank 2, batch size 1, target-only loss and two optimizer
   steps. Record allocated/reserved Torch memory, whole-device peak, RAM, phase,
   runtime, finite gradients and unchanged frozen state. Use fresh workers and
   no instance material or adapter adoption.
2. Try 512 and 1024 at explicit resource ceilings. The current recipe envelope
   permits up to 24 GiB of Torch memory, with a separate total-device allowance;
   begin by testing whether that additional room is sufficient in a disposable
   research path. Do not simply remove the production length check.
3. If peak memory remains the constraint, compare memory-saving attention,
   chunked vocabulary projection/loss, or activation offload to RAM. The current
   worker already uses non-reentrant gradient checkpointing, frozen NF4 weights,
   batch size 1 and no inference cache. It currently uses eager attention and
   materializes full vocabulary logits. Offloading unused vision weights is
   already implemented; general training activation/decoder offload is not.
4. Verify any alternative against the existing target-only labels, loss and
   adapter gradients on tiny CPU fixtures where possible, then against the
   actual GPU implementation. Preserve Gemma attention/soft-capping semantics,
   exact selected examples, repeatable reload, converter checks and wake safety.
5. Offer a versioned, reviewed longer-example recipe only after the measured
   workload succeeds. Keep the old recipe available, state numerical differences
   honestly, and make time/RAM/VRAM limits explicit. A new offer never approves
   training or changes an already reviewed plan.

More complete trajectories may be useful material selected by the instance.
Longer examples alone do not establish useful learning, preserve an earlier KV
state, or resolve the risks of repeated self-training. No automatic splitting,
truncation, training-data selection, or claim of cognitive benefit is implied.

## September 27 control measurements

These measurements use the pinned 31B source described in
[the original probe](qlora-31b-probe.md), on Windows 10.0.26200, an RTX 5090
(32,607 MiB reported by NVML), NVIDIA driver 610.88 and approximately 64 GiB of
installed system RAM. The isolated environment used Torch 2.14.0+cu130,
Transformers 5.17.0, PEFT 0.21.0, bitsandbytes 0.50.2 and Accelerate 1.15.0. They are measurements
on this machine, not validated configurations for other GPUs or Linux.

| Loss implementation | Tokens | Torch ceiling | Peak allocated | Peak reserved | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Production full logits | 256 | 22 GiB | 20.91 GiB | 21.97 GiB | Two steps passed |
| Production full logits | 512 | 24 GiB | 22.24 GiB | 23.98 GiB | Two steps, exact reload and conversion passed |
| Production full logits | 1024 | 24 GiB | — | — | OOM during first forward pass |
| Research chunked loss | 512 | 24 GiB | 20.90 GiB before reference comparison | 23.98 GiB including reference comparison | Two steps and GPU gradient comparison passed |
| Research chunked loss | 1024 | 24 GiB | 22.05 GiB | 23.98 GiB | Two steps and conversion passed |
| Research chunked loss | 1024 | 22.5 GiB | 22.05 GiB | 22.50 GiB | Two steps and exact reload passed; adapter identical to the 24 GiB run |
| Research chunked loss | 1536 | 24 GiB | 23.24 GiB | 24.00 GiB | Two steps, exact reload and conversion passed |
| Research chunked loss | 2048 | 24 GiB | — | — | OOM in decoder eager attention, first forward pass |

The 1024-token failure requested a 1 GiB allocation while Torch reported
22.86 GiB allocated and about 198 MiB reserved but unused, under its 24 GiB
ceiling. Device free memory was 4.38 GiB at that instant. It does not establish
the physical-card ceiling. Bogus Windows per-process values in the CUDA error
are not measurements and are omitted.

The chunked 2048-token attempt failed when eager attention requested another
512 MiB, with 22.45 GiB allocated and 1.11 GiB reserved but unused. This is a
ceiling/fragmentation result for the current allocator and attention path, not
proof that every 2048-token implementation needs more physical VRAM. Attention
memory or activation offload is the next optimization target; reducing output
chunks further would not address this particular failed allocation.

The full-logit 256/512 workers took approximately 453/410 seconds respectively,
including source hashing, loading and frozen-weight checks. Each peaked at
about 30.8 GiB of Windows job committed memory under a 32 GiB job limit.
Commit is not resident working-set size and must not be added to shared GPU
memory as though they were disjoint allocations.

The chunked 512-token run used the same two updates and produced a byte-identical
adapter file to the full-logit run. A subsequent comparison on those trained
weights found zero maximum absolute difference across all 240 LoRA gradient
tensors and a loss difference of about 1.46e-11. That comparison temporarily
raised its Torch allocation peak to 22.21 GiB; the 20.90 GiB figure excludes
that optional reference check. Whole-device and reserved-memory peaks include it.
The Windows sampler observed approximately 24.46 GiB dedicated process GPU
memory (during loading) and 86 MiB shared GPU memory. Sampling can miss brief
peaks; these are observations, not exact minimum hardware requirements.

Additional process/device measurements are below. A dash means unmeasured,
not zero. All training jobs peaked near **30.8 GiB committed host memory**.
The observed process resident peak was **9.86 GiB** where measured. These are
different accounting views, not additive memory requirements.

| Chunked example / Torch ceiling | Sampled process dedicated GPU | Sampled shared GPU | Sampled whole GPU, including other apps | Worker wall time | Two update steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512 / 24 GiB, includes gradient comparison | 24.46 GiB | 86 MiB | 29.29 GiB | 418 s | 3.7 s |
| 1024 / 24 GiB | 24.49 GiB | 86 MiB | 29.31 GiB | 443 s | 5.1 s |
| 1024 / 22.5 GiB | 23.09 GiB | 86 MiB | 27.67 GiB | 423 s | 7.4 s |
| 1536 / 24 GiB | 24.60 GiB | 86 MiB | 30.19 GiB | 424 s | 10.6 s |
| 2048 / 24 GiB, failed | 24.55 GiB | 86 MiB | 29.20 GiB | 339 s | — |

Worker wall time includes hashing the 62 GB source, loading/quantizing, frozen
tensor checks, evaluation and saving. The two-step timings are single-run
observations on repeated synthetic text, not throughput benchmarks or estimates
for arbitrary learning plans. Most elapsed time here was verification/loading.
The tighter 1024-token run produced the exact same adapter bytes as its 24 GiB
counterpart, while reducing reserved and sampled dedicated GPU memory.

The tighter 1024-token adapter also passed fresh-process reload under its
22.5 GiB Torch budget: all factors and both selected-example losses matched
exactly. Its existing 24 GiB counterpart supplies the GGUF conversion evidence
because the adapter files are byte-identical. The same identity relationship
connects the chunked 512-token adapter to the full-logit 512-token conversion
and reload. These checks establish artifact fidelity, not learning quality.
The 1536-token adapter separately passed its own exact fresh-process factor
and selected-loss reload, plus all 240 PEFT-to-GGUF factor comparisons.
The repeated synthetic example's loss did not improve at every deployment
strength: for example, the 512-token adapter at strength 0.1 slightly increased
its selected-example loss. The recorded losses are fidelity checks and cannot
establish a desirable real-world update.

The 1536-token adapter then passed a separate text-only llama.cpp test at
deployment strength 0.1. A disposable 3072-token context was retired down to
2053 retained tokens; ordinary strict/rebuild restore rejected the weight
change. The explicit experimental transition rebuilt those exact token IDs and
restored the sampler RNG, then saved a new checkpoint. A fresh native process
restored it with zero prompt reevaluation and reproduced all 16 continuation
tokens and their logits bit-for-bit. This used a 4096-token allocation, not
Syllas's context or the full production learning service; it did not adopt the
adapter into any instance.

The [machine-readable measurements](training-length-measurements.json) retain
byte counts, workload and package versions, sample validity, artifact identity,
and validation outcomes without machine-local paths or instance content.

### Using the measurements for hardware limits

Keep separate budgets for Torch allocations, whole-device GPU use, job commit,
and elapsed time. Reserved CUDA memory and driver allocations make physical
VRAM use larger than the allocated-tensor column; other applications also
count toward the whole-device watchdog. Lowering the allocator ceiling can
change caching and fragmentation, so a measured tensor peak alone is not a
validated smaller-card configuration. Loading and fresh reload must fit too.

The Windows sampler's shared GPU memory is a driver-reported host-memory use,
not the CPU placement of unused vision weights and not a measure of general
training offload. The current loader places the full decoder on one GPU. These
results therefore do not establish a supported RAM/VRAM split on a smaller
card, nor a Linux training configuration. Re-run bounded probes on each target
before offering a workload; different models also have different vocabulary,
attention and non-quantized-weight costs.

### RAM offload and block swapping

The equivalent of diffusion training's block swapping is training-aware layer
offload: move weights between CPU RAM and GPU as forward/backward passes need
them. The current trainer does not implement this for decoder layers. Its NF4
weights and quantization metadata, LoRA gradients, and transfers would need
explicit correctness and resource validation. An inference-only automatic
device map is not evidence that a training configuration works.

Activation offload instead moves tensors saved for backward into RAM. PyTorch
provides [saved-tensor hooks and `save_on_cpu`](https://docs.pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html)
for this memory/transfer-time tradeoff. It is a candidate for the next bounded
experiment alongside memory-efficient attention. The existing gradient
checkpointing recomputes activations; it does not itself move them to CPU.
Neither technique guarantees that a particular attention allocation will fit.

Frameworks such as [DeepSpeed](https://www.deepspeed.ai/docs/config-json/)
also expose parameter and optimizer offload, but compatibility with this
Windows/NF4/Gemma path has not been established. Optimizer offload alone would
save little here: only 3,584,000 adapter parameters are trained, while base
weights are frozen. Any offload mode needs separate host-RAM and time budgets;
Windows shared-GPU-memory observations are not a substitute for testing it.

## Research chunked loss

`--loss chunked` selects the shared `dmn/chunked_loss.py` implementation through
the research helper; v2 uses it directly. The default research loss and v1 retain
the original full-logit loss. The decoder still receives the entire example
with the same attention implementation and no cache. Its final hidden states
are projected to vocabulary logits in chunks of at most 64 positions. Each
chunk preserves Gemma's final logit soft-cap and shifted target mask, sums the
selected cross-entropy, and contributes to the mean over all selected targets.
Non-reentrant checkpointing recomputes the projection during backward instead
of retaining a full sequence-by-vocabulary activation graph. This does not
split a learning example into separate contexts or truncate it.

The pinned model has 262,144 vocabulary entries: one F32 logit tensor costs
1 MiB per input position, or 2 GiB at 2048 positions, before soft-cap and
backward intermediates. A 64-position chunk makes each such output buffer
64 MiB. Decoder activations and attention still grow with example length; this
does not make the rest of training constant-memory.

Different matrix and reduction shapes can change floating-point rounding.
Opt-in CPU tests compare full-context loss and all LoRA gradients on small text
and wrapped Gemma models, with soft-capping, irregular masks, uneven chunks and
decoder gradient checkpointing. The GPU comparison above tests the actual NF4
31B model. Exact equality in that one gradient check is not a universal numerical
guarantee. The separately offered v2 recipe now uses this implementation; v1
continues to use full logits.

Local validation ran the full standard-library suite (486 tests, 35 optional
skips) and the three opt-in CPU numerical tests in the isolated training
environment. GPU training, reload, conversion and native checks remain explicit
hardware experiments outside the ordinary CI suite.

## Reproducing the disposable length test

`scripts/probe_training_length.py` uses the production NF4 loader and shifted
target-only loss, with a synthetic repeated text example and its first quarter
masked from the loss. It opens no instance. It fixes rank 2, alpha 4, batch size 1,
two AdamW steps and deployment scale 0.1. Source hashes, implementation hashes,
finite gradients, changed adapter factors and unchanged frozen tensors are
checked. Its inspection-only default needs no GPU or training libraries:

```powershell
python scripts/probe_training_length.py --source SOURCE_FOLDER --tokens 512 --torch-vram-mib 24576
```

Explicit execution requires an unused output directory and the isolated training
interpreter. The Windows supervisor limits aggregate committed RAM to 32 GiB,
elapsed time to one hour by default, and sampled whole-device VRAM to 31 GiB.
Other applications count toward that device ceiling; it is a watchdog, not an
OS allocation quota. The Torch ceiling is separately enforced, and loading
requires that capacity plus 1 GiB to be free on the GPU.

```powershell
python scripts/probe_training_length.py --source SOURCE_FOLDER --tokens 512 --torch-vram-mib 24576 --training-python TRAIN_PYTHON --output FRESH_TRAIN_FOLDER --execute
python scripts/probe_training_length.py --phase reload --probe FRESH_TRAIN_FOLDER --training-python TRAIN_PYTHON --output FRESH_RELOAD_FOLDER --execute
python scripts/convert_qlora_31b_probe.py --probe FRESH_TRAIN_FOLDER --converter-manifest CONVERTER_MANIFEST --training-python TRAIN_PYTHON --output FRESH_CONVERSION_FOLDER
```

Reload uses the recorded workload and requires exact adapter factors and exact
selected-example losses at both strengths. Conversion is a separate CPU worker
that checks all GGUF factors against the saved PEFT tensors. Neither check runs
the instance, validates a native wake, or authorizes adoption. A two-step synthetic
probe tests execution and memory use, not learning quality or every workload up
to that length. Keep the recorded implementation unchanged between training and
reload; changing it invalidates that comparison.

Use `--loss chunked` for the research alternative. Its optional
`--compare-full-gradients` check is limited to at most 512 tokens because the
reference implementation itself runs out of memory at longer lengths on this
budget. The comparison performs backward passes without another optimizer step.
It contributes to whole-worker memory peaks; the receipt also records the peak
before that comparison. Standard library-only tests skip the optional numerical
tests; run them explicitly in the isolated trainer with CUDA hidden:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:DMN_TEST_CHUNKED_LOSS='1'
TRAIN_PYTHON -m unittest tests.test_chunked_training_loss -v
```

For Windows dedicated/shared-memory observations, launch
`scripts/sample_training_memory.ps1 -Folder FRESH_TRAIN_FOLDER -WorkerIds PID`
while the worker is still in its source-verification phase. `PID` is recorded
in `progress.json`. The sampler reads only that worker's GPU process counters,
records phase-labelled sampled peaks and invalid sample counts, and stops when
the supervisor writes `process.json`. Its working-set/commit observations are
distinct from the supervisor's OS-reported peak job commit. It is an observation
tool, not a resource limiter. No valid GPU samples means unavailable, not zero.
